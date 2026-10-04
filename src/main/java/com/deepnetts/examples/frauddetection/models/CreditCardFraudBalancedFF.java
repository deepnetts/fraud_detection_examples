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

public class CreditCardFraudBalancedFF {
    
    private static final String MODEL_PATH = "models/balanced-feedforward-network.dnet";

    public static void main(String[] args) throws DeepNettsException, IOException, ClassNotFoundException {

        DataPreparation dataPreparation = new DataPreparation(FraudDatasetConfigs.BALANCED_DATASET_CONFIG);

        dataPreparation.printDataSetShape();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.printSelectedColumns();
        dataPreparation.printTargetDistribution();

        String[] preparedCsvPaths = dataPreparation.createStratifiedTrainTestCsvFiles(
                "credit_card_fraud_ff_model_2", 0.8, 42);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet = DataSets.readCsv(
                preparedCsvPaths[0], numInputs, numOutputs, true);

        TabularDataSet<MLDataItem> testSet = DataSets.readCsv(
                preparedCsvPaths[1], numInputs, numOutputs, true);

        MaxScaler scaler = DataSets.scaleToMax(trainingSet);
        scaler.apply(testSet);

        // Enable Vector API to compare training performance:
        //DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(32, ActivationType.RELU)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopEpochs(30)
                .setLearningRate(0.001f)
                .setEarlyStopping(true)
                .setEarlyStoppingPatience(3)
                .setEarlyStoppingMinLossChange(0.001f);

        neuralNet.train(trainingSet);
        
        Files.createDirectories(Path.of("models"));
        neuralNet.save(MODEL_PATH);

        FeedForwardNetwork loadedModel = NeuralNetwork.load(MODEL_PATH, FeedForwardNetwork.class);

        MLDataItem sample = testSet.getItems().get(0);
        float[] input = sample.getInput().getValues();
        float[] prediction = loadedModel.predict(input);

        System.out.println("\n=== LOADED MODEL PREDICTION ===");
        System.out.println("Predicted fraud probability: " + prediction[0]);
        System.out.println("Actual label: " + sample.getTargetOutput().getValues()[0]);

        EvaluationMetrics evaluation = neuralNet.test(testSet);

        System.out.println("\n=== FEEDFORWARD NETWORK 2 TEST RESULTS ===");
        System.out.println(evaluation);
    }
}