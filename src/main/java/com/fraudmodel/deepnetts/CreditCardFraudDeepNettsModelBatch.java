package com.fraudmodel.deepnetts;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;

import java.io.IOException;

import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;

public class CreditCardFraudDeepNettsModelBatch {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 7;
        int numOutputs = 1;
        boolean hasColumnNames = true;


        DeepNetts.getInstance().setUseVectorAPI(true);
        
        TabularDataSet dataSet = DataSets.readCsv(
                "src/main/resources/data/card_transdata.csv",
                numInputs,
                numOutputs,
                hasColumnNames
        );

        
        DataSets.scaleToMax(dataSet);

        
        DataSet<MLDataItem>[] trainTestSet = dataSet.split(0.8);
        DataSet<MLDataItem> trainingSet = trainTestSet[0];
        DataSet<MLDataItem> testSet = trainTestSet[1];

        
        TabularDataSet batchedDataSet = DataSets.createBatchedDataset(dataSet, 128);
        
        
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs, 128)
                .addFullyConnectedLayer(32, ActivationType.RELU)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addFullyConnectedLayer(8, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        
        neuralNet.getTrainer().setStopError(0.05f)
                .setStopEpochs(10000)
                .setLearningRate(0.01f)
                .setBatchMode(true)
                .setBatchSize(128);

        
        neuralNet.train(batchedDataSet);

        EvaluationMetrics em = neuralNet.test(batchedDataSet);

        System.out.println("Train/test evaluation:");
        System.out.println(em);
    }
}