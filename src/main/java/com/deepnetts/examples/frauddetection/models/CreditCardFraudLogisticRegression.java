package com.deepnetts.examples.frauddetection.models;

import com.deepnetts.examples.frauddetection.data.DatasetConfig;
import com.deepnetts.examples.frauddetection.data.FraudDatasetConfigs;
import com.deepnetts.examples.frauddetection.data.DataPreparation;
import com.deepnetts.examples.frauddetection.util.MarkdownReport;
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

public class CreditCardFraudLogisticRegression {

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 42;
    private static final String MODEL_PATH = "models/logistic-regression.dnet";

    public static void main(String[] args)
            throws DeepNettsException, IOException, ClassNotFoundException  {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        dataPreparation.printDataSetShape();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.printSelectedColumns();
        dataPreparation.printTargetDistribution();

        String[] csvPaths = dataPreparation.createStratifiedTrainTestCsvFiles(
                "credit_card_fraud_logistic_regression",
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
        // DeepNetts.getInstance().setUseVectorAPI(true);
        
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork logisticRegression = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        logisticRegression.getTrainer()
                .setLearningRate(0.01f)
                .setStopError(0.13f)
                .setStopEpochs(200)
                .setShuffle(true)
                .setEarlyStopping(true)
                .setEarlyStoppingPatience(15)
                .setEarlyStoppingMinLossChange(0.00001f);

        logisticRegression.train(trainingSet);

        Files.createDirectories(Path.of("models"));
        logisticRegression.save(MODEL_PATH);

        FeedForwardNetwork loadedModel =
                NeuralNetwork.load(MODEL_PATH, FeedForwardNetwork.class);

        MLDataItem sample = testSet.getItems().get(0);
        float[] input = sample.getInput().getValues();
        float[] prediction = loadedModel.predict(input);

        System.out.println("\n=== LOADED MODEL PREDICTION ===");
        System.out.println("Predicted fraud probability: " + prediction[0]);
        System.out.println("Actual label: " + sample.getTargetOutput().getValues()[0]);

        EvaluationMetrics evaluation = logisticRegression.test(testSet);

        System.out.println("\n=== LOGISTIC REGRESSION TEST RESULTS ===");
        System.out.println(evaluation);
        
        String markdown = """
                # Logistic Regression Results

                ## Model

                Logistic Regression implemented as a single Sigmoid output neuron.

                ## Dataset

                Imbalanced credit card fraud dataset.

                ## Architecture

                %d → %d

                ## Training Configuration

                - Learning rate: 0.01
                - Maximum epochs: 200
                - Stop error: 0.13
                - Early stopping: enabled
                - Early stopping patience: 15

                ## Evaluation

                %s
                """.formatted(numInputs, numOutputs, MarkdownReport.evaluationTable(evaluation));

        MarkdownReport.save("logistic-regression-results.md", markdown);
    }
}