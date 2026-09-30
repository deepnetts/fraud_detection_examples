package com.deepnetts.examples.frauddetection.models;

import com.deepnetts.examples.frauddetection.data.FraudDatasetConfigs;
import com.deepnetts.examples.frauddetection.data.DataPreparation;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;

import java.io.IOException;

import javax.visrec.ml.eval.EvaluationMetrics;

public class CreditCardFraudBalancedFFBatch {

    private static final int BATCH_SIZE = 256;

    public static void main(String[] args) throws DeepNettsException, IOException {

        DataPreparation dataPreparation = new DataPreparation(FraudDatasetConfigs.BALANCED_DATASET_CONFIG);

        dataPreparation.printDataSetShape();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.printSelectedColumns();
        dataPreparation.printTargetDistribution();

        String[] preparedCsvPaths = dataPreparation.createStratifiedTrainTestCsvFilesForBatch(
                "credit_card_fraud_ff_model_batch_2", 0.8, 42, BATCH_SIZE);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet = DataSets.readCsv(
                preparedCsvPaths[0], numInputs, numOutputs, true);

        TabularDataSet<MLDataItem> testSet = DataSets.readCsv(
                preparedCsvPaths[1], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);
        scaler.apply(testSet);

        TabularDataSet<MLDataItem> batchedTrainingSet = DataSets.createBatchedDataset(trainingSet, BATCH_SIZE);
        TabularDataSet<MLDataItem> batchedTestSet = DataSets.createBatchedDataset(testSet, BATCH_SIZE);

        // Enable Vector API to compare training performance:
        DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs, BATCH_SIZE)
                .addFullyConnectedLayer(32, ActivationType.RELU)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addFullyConnectedLayer(8, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopError(0.5f)
                .setStopEpochs(10)
                .setLearningRate(0.1f)
                .setBatchMode(true)
                .setBatchSize(BATCH_SIZE);

        neuralNet.train(batchedTrainingSet);

        EvaluationMetrics evaluation = neuralNet.test(batchedTestSet);

        System.out.println("\n=== BATCH FEEDFORWARD NETWORK 2 TEST RESULTS ===");
        System.out.println(evaluation);
    }
}