package com.fraudmodel.deepnetts;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;

import java.io.IOException;

import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;

public class CreditCardFraudLogisticRegression {

    public static void main(String[] args)
            throws DeepNettsException, IOException {

      
        DataPreparation dataPreparation = new DataPreparation(
                "src/main/resources/data/card_transdata.csv"
        );

        dataPreparation.printBasicInfo();
        dataPreparation.printColumnInfo();
        dataPreparation.previewRows(5);
        dataPreparation.countMissingValues();
        dataPreparation.printTargetDistribution();

        String preparedCsvPath =
                dataPreparation.createPreparedCsvFile(
                        "credit_card_fraud_logistic_regression.csv"
                );

        
        int numInputs = dataPreparation.getInputSize();
        int numOutputs = dataPreparation.getOutputSize();
        boolean hasColumnNames = true;

 
        DataSet<MLDataItem> dataSet = DataSets.readCsv(
                preparedCsvPath,
                numInputs,
                numOutputs,
                hasColumnNames
        );

       
        DataSets.scaleToMax(dataSet);

      
        DataSet<MLDataItem>[] trainTestSet = dataSet.split(0.8);

        DataSet<MLDataItem> trainingSet = trainTestSet[0];
        DataSet<MLDataItem> testSet = trainTestSet[1];

        DeepNetts.getInstance().setUseVectorAPI(true);

        
        FeedForwardNetwork logisticRegression =
                FeedForwardNetwork.builder()
                        .addInputLayer(numInputs)
                        .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                        .lossFunction(LossType.CROSS_ENTROPY)
                        .build();

    
        logisticRegression.getTrainer()
                .setLearningRate(0.01f)
                .setStopError(0.13f)
                .setStopEpochs(200).
                setShuffle(true)
                .setEarlyStopping(true)
                .setEarlyStoppingPatience(15)
                .setEarlyStoppingMinLossChange(0.00001f);

        
        System.out.println();
        
        logisticRegression.train(trainingSet);

        EvaluationMetrics evaluation =
                logisticRegression.test(testSet);

        System.out.println();
        System.out.println(
                "LOGISTIC REGRESSION TEST RESULTS"
        );

        System.out.println(evaluation);
    }
}