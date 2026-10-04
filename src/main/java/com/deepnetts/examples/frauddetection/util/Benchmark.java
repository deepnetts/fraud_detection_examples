package com.deepnetts.examples.frauddetection.util;

import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.FeedForwardNetwork;

public class Benchmark {
    public static void benchmarkInference(FeedForwardNetwork neuralNet, TabularDataSet<MLDataItem> dataSet) {
                long start = System.currentTimeMillis();
        for(MLDataItem item : dataSet) {
            neuralNet.predict(item.getInput());
        }        
        long end = System.currentTimeMillis();
        long inferenceTime = end-start;
        
        System.out.println("Inference time for 1M rows: "+inferenceTime + " ms");
    }
}
