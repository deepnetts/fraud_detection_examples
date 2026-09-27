package com.deepnetts.examples.frauddetection.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

public class TrainingLossChart {

    private static final String OUTPUT_DIRECTORY = "target/training-charts";

    public static void save(List<Integer> epochs, List<Float> trainingLosses, String fileName) throws IOException {
        Path outputDirectory = Path.of(OUTPUT_DIRECTORY);
        Files.createDirectories(outputDirectory);

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(500)
                .title("Autoencoder Training Loss")
                .xAxisTitle("Epoch")
                .yAxisTitle("Mean Squared Error")
                .build();

        chart.addSeries("Training Loss", epochs, trainingLosses);

        String outputPath = outputDirectory.resolve(fileName).toString();
        BitmapEncoder.saveBitmap(chart, outputPath, BitmapEncoder.BitmapFormat.PNG);
    }

    private TrainingLossChart() {
    }
}
