package com.deepnetts.examples.frauddetection.models;

import com.deepnetts.examples.frauddetection.data.DatasetConfig;
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

public class CreditCardFraudFFNetwork {

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 42;

    public static void main(String[] args)
            throws DeepNettsException, IOException {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        dataPreparation.printDataSetShape();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.printSelectedColumns();
        dataPreparation.printTargetDistribution();

        String[] csvPaths = dataPreparation.createStratifiedTrainTestCsvFiles(
                "credit_card_fraud_deepnetts_model",
                TRAIN_RATIO,
                RANDOM_SEED
        );

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet =
                DataSets.readCsv(csvPaths[0], numInputs, numOutputs, true);

        TabularDataSet<MLDataItem> testSet =
                DataSets.readCsv(csvPaths[1], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);
        scaler.apply(testSet);
        
        // Enable Vector API to compare training performance:
        DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(32, ActivationType.RELU)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopError(0.05f)
                .setStopEpochs(10)
                .setLearningRate(0.001f);

        neuralNet.train(trainingSet);

        EvaluationMetrics evaluation = neuralNet.test(testSet);

        System.out.println("\n=== FEEDFORWARD NETWORK TEST RESULTS ===");
        System.out.println(evaluation);
    }
}