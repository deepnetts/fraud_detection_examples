package com.deepnetts.examples.frauddetection.bench;

import com.deepnetts.examples.frauddetection.models.*;
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

public class CreditCardFraudFFNetworkBatchBench2 {

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 42;
    private static final int BATCH_SIZE = 128;

    public static void main(String[] args) throws DeepNettsException, IOException {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        String[] csvPaths = dataPreparation.createStratifiedTrainTestCsvFilesForBatch(
                "credit_card_fraud_deepnetts_model_batch", TRAIN_RATIO, RANDOM_SEED, BATCH_SIZE);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        // Enable Vector API to compare training performance:
        DeepNetts.getInstance().setUseVectorAPI(false);
        DeepNetts.getInstance().setMaxThreads(1);

        TabularDataSet<MLDataItem> trainingSet = DataSets.readCsv(csvPaths[0], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);

        TabularDataSet<MLDataItem> batchedTrainingSet = DataSets.createBatchedDataset(trainingSet, BATCH_SIZE);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs, BATCH_SIZE)
                .addFullyConnectedLayer(128, ActivationType.RELU)
                .addFullyConnectedLayer(128, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopError(0.05f)
                .setStopEpochs(5)
                .setLearningRate(0.01f)
                .setBatchMode(true)
                .setBatchSize(BATCH_SIZE);

        neuralNet.train(batchedTrainingSet);

    }
}