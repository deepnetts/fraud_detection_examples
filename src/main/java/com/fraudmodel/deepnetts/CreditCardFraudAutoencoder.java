package com.fraudmodel.deepnetts;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSetItem;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;

public class CreditCardFraudAutoencoder {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 7;
        int numOutputs = 1;
        boolean hasColumnNames = true;

        DeepNetts.getInstance().setUseVectorAPI(true);
        DeepNetts.getInstance().setMaxThreads(1);
    
        
        DataSet<MLDataItem> dataSet = DataSets.readCsv(
                "src/main/resources/data/card_transdata.csv",
                numInputs,
                numOutputs,
                hasColumnNames
        );
        
        DataSets.scaleToMax(dataSet);
     //   dataSet = dataSet.split(0.1)[0];

        TabularDataSet<MLDataItem> autoencoderDataSet = new TabularDataSet(numInputs, numInputs);
        for (MLDataItem item : dataSet.getItems()) {
            if (item.getTargetOutput().getValues()[0]==1) {
                MLDataItem newItem = new DataSetItem(item.getInput(), item.getInput());
                autoencoderDataSet.add(newItem);
            }
        }
        

        FeedForwardNetwork autoencoder = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(5, ActivationType.RELU)
                .addFullyConnectedLayer(3, ActivationType.RELU)
                .addFullyConnectedLayer(5, ActivationType.RELU)
                .addOutputLayer(numInputs, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        
        autoencoder.getTrainer()
                    .setStopError(0.001f)
                    .setStopEpochs(1000)
                    .setLearningRate(0.01f);

        
        autoencoder.train(autoencoderDataSet);
        
        
        // todo: predice za prevaru i ne prevaru i greska mora da bud eveca za prevaru
        //  odnsono veca od thresholda


    }
}