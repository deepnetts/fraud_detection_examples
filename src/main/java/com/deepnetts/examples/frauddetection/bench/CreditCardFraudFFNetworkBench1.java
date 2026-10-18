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

public class CreditCardFraudFFNetworkBench1 {

    public static void main(String[] args)
            throws DeepNettsException, IOException {

        DatasetConfig config = FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG;
        DataPreparation dataPreparation = new DataPreparation(config);

        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();

        TabularDataSet<MLDataItem> trainingSet =
                DataSets.readCsv(FraudDatasetConfigs.IMBALANCED_DATASET_CONFIG.getCsvFile(), numInputs, numOutputs, true);
     
        MaxScaler scaler = DataSets.scaleToMax(trainingSet);

        
        // Enable Vector API to compare training performance:
        DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(128, ActivationType.RELU)
                .addFullyConnectedLayer(128, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        neuralNet.getTrainer()
                .setStopError(0.05f)
                .setStopEpochs(1)
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