package com.deepnetts.examples.frauddetection.models;

import com.deepnetts.examples.frauddetection.data.FraudDatasetConfigs;
import com.deepnetts.examples.frauddetection.data.DataPreparation;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import javax.visrec.ml.eval.EvaluationMetrics;

public class CreditCardFraudBalancedFFBatchGPU {

    private static final int BATCH_SIZE = 102400;
    private static final String MODEL_PATH = "models/balanced-feedforward-network-batch.dnet";

    public static void main(String[] args) throws DeepNettsException, IOException, ClassNotFoundException {

        DataPreparation dataPreparation = new DataPreparation(FraudDatasetConfigs.BALANCED_DATASET_CONFIG);

        String[] preparedCsvPaths = dataPreparation.createStratifiedTrainTestCsvFilesForBatch(
                "credit_card_fraud_ff_model_batch_2", 0.8, 42, BATCH_SIZE);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet = DataSets.readCsv(preparedCsvPaths[0], numInputs, numOutputs, true);

       for (int i =0; i<50; i++) {
            TabularDataSet<MLDataItem> trainingSetAdd = DataSets.readCsv(preparedCsvPaths[0], numInputs, numOutputs, true);
            trainingSet.addAll(trainingSetAdd);

        }

        TabularDataSet<MLDataItem> testSet = DataSets.readCsv(
                preparedCsvPaths[1], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);
        scaler.apply(testSet);

        TabularDataSet<MLDataItem> batchedTrainingSet = DataSets.createBatchedDatasetGPU(trainingSet, BATCH_SIZE);

        // Enable CUDA GPU to compare training performance:
        DeepNetts.getInstance().setUseCuda(true);
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs, BATCH_SIZE)
                .addFullyConnectedLayer(32, ActivationType.TANH)
                .addFullyConnectedLayer(16, ActivationType.TANH)
                .addFullyConnectedLayer(8, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopError(0.5f)
                .setStopEpochs(10)
                .setLearningRate(0.5f)
                .setBatchMode(true)
                .setBatchSize(BATCH_SIZE);

        neuralNet.train(batchedTrainingSet);
        
        Files.createDirectories(Path.of("models"));
        neuralNet.save(MODEL_PATH);

    }
}