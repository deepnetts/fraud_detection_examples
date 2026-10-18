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
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;

import java.io.IOException;

import javax.visrec.ml.eval.EvaluationMetrics;

public class CreditCardFraudFFNetworkBench10M {

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 42;

    public static void main(String[] args)
            throws DeepNettsException, IOException {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
             TabularDataSet<MLDataItem> trainingSet1 =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
             TabularDataSet<MLDataItem> trainingSet2 =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
             TabularDataSet<MLDataItem> trainingSet3 =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
             TabularDataSet<MLDataItem> trainingSet4 =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
             TabularDataSet<MLDataItem> trainingSet5 =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
     
       trainingSet.addAll(trainingSet1);
       trainingSet.addAll(trainingSet2);
       trainingSet.addAll(trainingSet3);
       trainingSet.addAll(trainingSet4);
       trainingSet.addAll(trainingSet5);
        
        MaxScaler scaler = DataSets.scaleToMax(trainingSet);

        
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
                .setStopEpochs(5)
                .setLearningRate(0.001f);

        neuralNet.train(trainingSet);
      
        long start = System.currentTimeMillis();
        for(MLDataItem item : trainingSet) {
            neuralNet.predict(item.getInput());
        }        
        long end = System.currentTimeMillis();
        long inferenceTime = end-start;
        
        System.out.println("Inference time for 1M rows: "+inferenceTime + " ms");
        

    }
}