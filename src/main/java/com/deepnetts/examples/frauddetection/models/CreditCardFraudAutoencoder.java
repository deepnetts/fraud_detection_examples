package com.deepnetts.examples.frauddetection.models;

import com.deepnetts.examples.frauddetection.data.DatasetConfig;
import com.deepnetts.examples.frauddetection.data.FraudDatasetConfigs;
import com.deepnetts.examples.frauddetection.data.DataPreparation;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSetItem;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.TrainingEvent;
import deepnetts.net.train.TrainingListener;
import deepnetts.util.DeepNettsException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import com.deepnetts.examples.frauddetection.util.TrainingLossChart;

public class CreditCardFraudAutoencoder {

    private static final double AUTOENCODER_TRAIN_RATIO = 0.9;
    private static final long RANDOM_SEED = 42;

    public static void main(String[] args) throws DeepNettsException, IOException {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        dataPreparation.printDataSetShape();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.printSelectedColumns();
        dataPreparation.printTargetDistribution();

        String[] csvPaths = dataPreparation.createStratifiedTrainTestCsvFiles(
                "credit_card_fraud_autoencoder", 0.8, RANDOM_SEED);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet = DataSets.readCsv(csvPaths[0], numInputs, numOutputs, true);
        TabularDataSet<MLDataItem> testSet = DataSets.readCsv(csvPaths[1], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);
        scaler.apply(testSet);

        List<MLDataItem> normalItems = new ArrayList<>();
        List<MLDataItem> fraudItems = new ArrayList<>();

        for (MLDataItem item : trainingSet.getItems()) {
            if (item.getTargetOutput().getValues()[0] == 0) normalItems.add(item);
            else fraudItems.add(item);
        }

        Collections.shuffle(normalItems, new Random(RANDOM_SEED));
        Collections.shuffle(fraudItems, new Random(RANDOM_SEED));

        int aeTrainSize = (int) (normalItems.size() * AUTOENCODER_TRAIN_RATIO);
        int calibrationNormalCount = normalItems.size() - aeTrainSize;
        double fraudRatio = fraudItems.size() / (double) trainingSet.size();
        int calibrationFraudCount = (int) Math.round(
                calibrationNormalCount * fraudRatio / (1.0 - fraudRatio));

        TabularDataSet<MLDataItem> aeTrainingSet = new TabularDataSet<>(numInputs, numInputs);
        List<MLDataItem> calibrationSet = new ArrayList<>();

        for (int i = 0; i < aeTrainSize; i++) {
            MLDataItem item = normalItems.get(i);
            aeTrainingSet.add(new DataSetItem(item.getInput(), item.getInput()));
        }

        for (int i = aeTrainSize; i < normalItems.size(); i++) calibrationSet.add(normalItems.get(i));
        for (int i = 0; i < calibrationFraudCount; i++) calibrationSet.add(fraudItems.get(i));

        Collections.shuffle(calibrationSet, new Random(RANDOM_SEED));

        System.out.println("\n=== AUTOENCODER DATA SPLIT ===");
        System.out.println("Autoencoder training: " + aeTrainingSet.size());
        System.out.println("Calibration set: " + calibrationSet.size());
        System.out.println("Final test set: " + testSet.size());

        // Enable Vector API to compare training performance:
        // DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);

        int hiddenSize = Math.max(2, (int) Math.ceil(numInputs * 0.75));
        int bottleneckSize = Math.max(1, (int) Math.ceil(numInputs * 0.5));

        FeedForwardNetwork autoencoder = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(hiddenSize, ActivationType.RELU)
                .addFullyConnectedLayer(bottleneckSize, ActivationType.RELU)
                .addFullyConnectedLayer(hiddenSize, ActivationType.RELU)
                .addOutputLayer(numInputs, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        autoencoder.getTrainer()
                .setStopError(0.001f)
                .setStopEpochs(20)
                .setLearningRate(0.01f)
                .setEarlyStopping(true)
                .setEarlyStoppingPatience(3)
                .setEarlyStoppingMinLossChange(0.00001f);

        System.out.println("\n=== AUTOENCODER ARCHITECTURE ===");
        System.out.println(numInputs + " -> " + hiddenSize + " -> " + bottleneckSize
                + " -> " + hiddenSize + " -> " + numInputs);

        
        List<Integer> epochs = new ArrayList<>();
        List<Float> trainingLosses = new ArrayList<>();
        
        autoencoder.getTrainer().addListener(new TrainingListener() {
            @Override
            public void handleEvent(TrainingEvent event) {
                if (event.getType() == TrainingEvent.Type.EPOCH_FINISHED) {
                    epochs.add(event.getSource().getCurrentEpoch());
                    trainingLosses.add(event.getSource().getTrainingLoss());
                }
            }
        });
        
        // Train only on normal transactions.
        autoencoder.train(aeTrainingSet);

        TrainingLossChart.save(epochs, trainingLosses, "autoencoder-training-loss");

        // Select the anomaly threshold on a separate calibration set.
        ThresholdResult calibration = findBestThreshold(autoencoder, calibrationSet);

        System.out.println("\n=== THRESHOLD CALIBRATION ===");
        System.out.println("Threshold: " + calibration.threshold);
        System.out.println("Calibration F1: " + calibration.f1);

        // Evaluate once on the untouched test set.
        EvaluationResult result = evaluate(autoencoder, testSet, calibration.threshold);

        System.out.println("\n=== TEST CONFUSION MATRIX ===");
        System.out.println("True positives: " + result.truePositive);
        System.out.println("True negatives: " + result.trueNegative);
        System.out.println("False positives: " + result.falsePositive);
        System.out.println("False negatives: " + result.falseNegative);

        System.out.println("\n=== FINAL TEST METRICS ===");
        System.out.println("Accuracy: " + result.accuracy);
        System.out.println("Precision: " + result.precision);
        System.out.println("Recall: " + result.recall);
        System.out.println("F1 score: " + result.f1);
        System.out.println("False positive rate: " + result.falsePositiveRate);
        System.out.println("ROC-AUC: " + rocAuc(result.normalErrors, result.fraudErrors));
    }

    private static ThresholdResult findBestThreshold(
            FeedForwardNetwork autoencoder, List<MLDataItem> calibrationSet) {

        List<ScoredTransaction> transactions = new ArrayList<>();

        for (MLDataItem item : calibrationSet) {
            double error = reconstructionError(autoencoder, item);
            int label = item.getTargetOutput().getValues()[0] == 1 ? 1 : 0;
            transactions.add(new ScoredTransaction(error, label));
        }

        transactions.sort(Comparator.comparingDouble(
                (ScoredTransaction t) -> t.error).reversed());

        int totalFraud = 0;
        for (ScoredTransaction t : transactions) if (t.label == 1) totalFraud++;

        int truePositive = 0, falsePositive = 0, index = 0;
        double bestThreshold = 0, bestF1 = 0;

        while (index < transactions.size()) {
            double error = transactions.get(index).error;

            while (index < transactions.size()
                    && Double.compare(transactions.get(index).error, error) == 0) {
                if (transactions.get(index).label == 1) truePositive++;
                else falsePositive++;
                index++;
            }

            double precision = truePositive / (double) (truePositive + falsePositive);
            double recall = truePositive / (double) totalFraud;
            double f1 = precision + recall == 0 ? 0
                    : 2 * precision * recall / (precision + recall);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestThreshold = index < transactions.size()
                        ? (error + transactions.get(index).error) / 2
                        : Math.nextDown(error);
            }
        }

        return new ThresholdResult(bestThreshold, bestF1);
    }

    private static EvaluationResult evaluate(
            FeedForwardNetwork autoencoder, TabularDataSet<MLDataItem> testSet,
            double threshold) {

        EvaluationResult result = new EvaluationResult();

        for (MLDataItem item : testSet.getItems()) {
            double error = reconstructionError(autoencoder, item);
            int actual = item.getTargetOutput().getValues()[0] == 1 ? 1 : 0;
            int predicted = error > threshold ? 1 : 0;

            if (actual == 0) result.normalErrors.add(error);
            else result.fraudErrors.add(error);

            if (actual == 1 && predicted == 1) result.truePositive++;
            else if (actual == 0 && predicted == 0) result.trueNegative++;
            else if (actual == 0) result.falsePositive++;
            else result.falseNegative++;
        }

        Collections.sort(result.normalErrors);
        Collections.sort(result.fraudErrors);

        int tp = result.truePositive, tn = result.trueNegative;
        int fp = result.falsePositive, fn = result.falseNegative;

        result.accuracy = (tp + tn) / (double) (tp + tn + fp + fn);
        result.precision = tp + fp == 0 ? 0 : tp / (double) (tp + fp);
        result.recall = tp + fn == 0 ? 0 : tp / (double) (tp + fn);
        result.f1 = result.precision + result.recall == 0 ? 0
                : 2 * result.precision * result.recall / (result.precision + result.recall);
        result.falsePositiveRate = tn + fp == 0 ? 0 : fp / (double) (tn + fp);

        return result;
    }

    private static double reconstructionError(FeedForwardNetwork autoencoder, MLDataItem item) {
        float[] input = item.getInput().getValues();
        float[] output = autoencoder.predict(input);
        double error = 0;

        for (int i = 0; i < input.length; i++) {
            double difference = input[i] - output[i];
            error += difference * difference;
        }

        return error / input.length;
    }

    private static double rocAuc(List<Double> normal, List<Double> fraud) {
        double sum = 0;
        int lower = 0, equal = 0;

        for (double fraudError : fraud) {
            while (lower < normal.size() && normal.get(lower) < fraudError) lower++;
            while (equal < normal.size() && normal.get(equal) <= fraudError) equal++;
            sum += lower + 0.5 * (equal - lower);
        }

        return sum / ((double) normal.size() * fraud.size());
    }

    private static class ScoredTransaction {
        final double error;
        final int label;

        ScoredTransaction(double error, int label) {
            this.error = error;
            this.label = label;
        }
    }

    private static class ThresholdResult {
        final double threshold, f1;

        ThresholdResult(double threshold, double f1) {
            this.threshold = threshold;
            this.f1 = f1;
        }
    }

    private static class EvaluationResult {
        int truePositive, trueNegative, falsePositive, falseNegative;
        double accuracy, precision, recall, f1, falsePositiveRate;
        final List<Double> normalErrors = new ArrayList<>();
        final List<Double> fraudErrors = new ArrayList<>();
    }
}