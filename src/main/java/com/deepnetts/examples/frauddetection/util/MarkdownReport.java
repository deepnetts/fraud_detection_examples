/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.deepnetts.examples.frauddetection.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class MarkdownReport {

    private static final String OUTPUT_DIRECTORY = "target/reports";

    public static void save(String fileName, String content) throws IOException {
        Path outputDirectory = Path.of(OUTPUT_DIRECTORY);
        Files.createDirectories(outputDirectory);
        Files.writeString(outputDirectory.resolve(fileName), content);
    }

    public static String evaluationTable(Object evaluation) {
        StringBuilder table = new StringBuilder();
        table.append("| Metric | Value |\n");
        table.append("|---|---:|\n");

        for (String line : evaluation.toString().split("\\R")) {
            if (!line.contains(":")) continue;

            String[] parts = line.split(":", 2);
            String metric = parts[0].trim();
            String value = parts[1].trim().split("\\s+")[0];

            if (metric.equals("Class")) continue;
            if (metric.equals("Total items")) metric = "Total Items";
            if (metric.equals("True positive")) metric = "True Positive";
            if (metric.equals("True negative")) metric = "True Negative";
            if (metric.equals("False positive")) metric = "False Positive";
            if (metric.equals("False negative")) metric = "False Negative";
            if (metric.equals("Accuracy (ACC)")) metric = "Accuracy";
            if (metric.equals("Precision (PPV)")) metric = "Precision";
            if (metric.equals("Fall-out (FPR)")) metric = "False Positive Rate";
            if (metric.equals("False negative rate (FNR)")) metric = "False Negative Rate";

            table.append("| ").append(metric).append(" | ").append(value).append(" |\n");
        }

        return table.toString();
    }

    private MarkdownReport() {
    }
}
