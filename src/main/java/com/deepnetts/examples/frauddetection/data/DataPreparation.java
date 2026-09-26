package com.deepnetts.examples.frauddetection.data;

import java.io.IOException;
import org.dflib.DataFrame;
import org.dflib.Series;
import org.dflib.csv.Csv;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DataPreparation {

    private final DataFrame dataSet;
    private final DatasetConfig config;

    public DataPreparation(DatasetConfig config) {
        this.config = config;
        this.dataSet = Csv.load(config.getCsvFile());
    }

    public DataFrame getData() {
        return dataSet;
    }

    // Prints basic information about the loaded dataset.
    public void printDataSetShape() {
        System.out.println();
        System.out.println("=== DATASET SHAPE ===");
        System.out.println("Rows: " + dataSet.height());
        System.out.println("Columns: " + dataSet.width());
    }

    // Prints column information
    public void printColumnInfo() {

        System.out.println();
        System.out.println("=== COLUMN STRUCTURE ===");

        System.out.printf("%-40s %-15s %-15s %-10s%n", "Column", "Non-Missing", "Missing", "Type");

        for (String columnName : dataSet.getColumnsIndex()) {
            Series<?> column = dataSet.getColumn(columnName);
            int missingCount = 0;
            for (Object value : column) {
                if (isMissingValue(value)) {
                    missingCount++;
                }
            }
            int nonMissingCount = dataSet.height() - missingCount;
            String type = inferColumnType(column);
            System.out.printf(
                    "%-40s %-15d %-15d %-10s%n",
                    columnName,
                    nonMissingCount,
                    missingCount,
                    type
            );
        }
    }

    // Prints the first n rows of the dataset for a quick data preview.
    public void previewRows(int numberOfRows) {
        System.out.println();
        System.out.println("=== FIRST " + numberOfRows + " ROWS ===");

        int rowsToPrint = Math.min(numberOfRows, dataSet.height());

        for (int row = 0; row < rowsToPrint; row++) {

            StringBuilder line = new StringBuilder();

            for (String columnName : dataSet.getColumnsIndex()) {

                if (line.length() > 0) {
                    line.append(" | ");
                }

                line.append(columnName)
                        .append(": ")
                        .append(dataSet.getColumn(columnName).get(row));
            }
            System.out.println(line);
        }
    }
    
    // Checks and reports missing values in each column.
    public void countMissingValues() {
        int totalRows = dataSet.height();

        System.out.println();
        System.out.println("=== MISSING VALUES ===");

        for (String columnName : dataSet.getColumnsIndex()) {

            Series<?> column = dataSet.getColumn(columnName);
            int missing = 0;

            for (int row = 0; row < totalRows; row++) {

                Object value = column.get(row);

                if (isMissingValue(value)) {
                    missing++;
                }
            }

            double percent =
                    totalRows == 0
                            ? 0.0
                            : missing * 100.0 / totalRows;

            System.out.printf( "Column: %-35s | Missing: %d (%.2f%%)%n", columnName, missing, percent);
        }
    }

    // Prints the distribution of fraud and non-fraud transactions.
    public void printTargetDistribution() {
        int totalRows = dataSet.height();

        Series<?> fraudColumn = dataSet.getColumn(getOutputColumn());

        int fraudCount = 0;

        for (int row = 0; row < totalRows; row++) {
            Object value = fraudColumn.get(row);
            if (value != null && Double.parseDouble(value.toString()) == 1.0) {
                fraudCount++;
            }
        }

        int nonFraudCount = totalRows - fraudCount;

        double fraudRate =
                totalRows == 0
                        ? 0.0
                        : fraudCount * 100.0 / totalRows;

        double nonFraudRate =
                totalRows == 0
                        ? 0.0
                        : nonFraudCount * 100.0 / totalRows;

        System.out.println();
        System.out.println("=== TARGET DISTRIBUTION ===");
        System.out.println("Total transactions: " + totalRows);

        System.out.printf("Non-fraud transactions: %d (%.2f%%)%n", nonFraudCount, nonFraudRate);

        System.out.printf("Fraud transactions: %d (%.2f%%)%n", fraudCount, fraudRate);
    }

    // Creates a CSV file containing only the selected input and target columns.
    public String createPreparedCsvFile(String outputFileName) {

        Path outputPath = Path.of("target", "prepared-data", outputFileName);

        String[] selectedColumns = new String[getInputSize() + 1];

        System.arraycopy(getInputColumns(), 0, selectedColumns, 0, getInputSize());

        selectedColumns[getInputSize()] = getOutputColumn();

        DataFrame preparedData = dataSet.cols(selectedColumns).select();

        Csv.saver().createMissingDirs().save(preparedData, outputPath.toString());

        System.out.println();
        System.out.println("=== PREPARED CSV CREATED ===");
        System.out.println("Path: " + outputPath);

        System.out.println("Input columns: " + String.join(", ", getInputColumns()));

        System.out.println("Output column: " + getOutputColumn());

        System.out.println("Rows: " + preparedData.height());

        System.out.println("Columns: " + preparedData.width());

        return outputPath.toString();
    }

    public String[] getInputColumns() {
        return config.getInputColumns();
    }

    public String getOutputColumn() {
        return config.getTargetColumn();
    }

    public int getInputSize() {
        return config.getInputColumns().length;
    }

    public int getOutputSize() {
        return 1;
    }
    
    // Splits the dataset into training and test sets while preserving the class distribution.
    public DataFrame[] stratifiedSplit(double trainRatio, long seed) {

        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException(
                    "trainRatio must be between 0 and 1"
            );
        }

        List<Integer> fraudRows = new ArrayList<>();
        List<Integer> nonFraudRows = new ArrayList<>();

        for (int row = 0; row < dataSet.height(); row++) {

            double label = Double.parseDouble(dataSet.getColumn(getOutputColumn()).get(row).toString());

            if (label == 1.0) {
                fraudRows.add(row);
            } else if (label == 0.0) {
                nonFraudRows.add(row);
            } else {
                throw new IllegalArgumentException(
                        "Unexpected target value: " + label
                );
            }
        }

        Random random = new Random(seed);

        Collections.shuffle(fraudRows, random);
        Collections.shuffle(nonFraudRows, random);

        int fraudTrainSize = (int) (fraudRows.size() * trainRatio);

        int nonFraudTrainSize = (int) (nonFraudRows.size() * trainRatio);

        List<Integer> trainRows = new ArrayList<>();
        List<Integer> testRows = new ArrayList<>();

        trainRows.addAll(fraudRows.subList(0, fraudTrainSize));

        trainRows.addAll(nonFraudRows.subList(0, nonFraudTrainSize));

        testRows.addAll(fraudRows.subList(fraudTrainSize, fraudRows.size()));

        testRows.addAll(nonFraudRows.subList(nonFraudTrainSize, nonFraudRows.size()));

        Collections.shuffle(trainRows, random);
        Collections.shuffle(testRows, random);

        int[] trainIndices = trainRows.stream().mapToInt(Integer::intValue).toArray();

        int[] testIndices = testRows.stream().mapToInt(Integer::intValue).toArray();

        DataFrame trainingData = dataSet.rows(trainIndices).select();

        DataFrame testData = dataSet.rows(testIndices).select();

        int fraudTestSize = fraudRows.size() - fraudTrainSize;

        double trainFraudRate = fraudTrainSize * 100.0 / trainingData.height();

        double testFraudRate = fraudTestSize * 100.0 / testData.height();

        System.out.println();
        System.out.println("=== STRATIFIED TRAIN/TEST SPLIT ===");

        System.out.printf("Training set: %d rows | Fraud: %d (%.2f%%)%n", trainingData.height(), fraudTrainSize, trainFraudRate);

        System.out.printf("Test set: %d rows | Fraud: %d (%.2f%%)%n", testData.height(), fraudTestSize, testFraudRate);
        
        return new DataFrame[]{
                trainingData,
                testData
        };
    }
    
    // Creates stratified training and test sets and saves them as separate CSV files.
    public String[] createStratifiedTrainTestCsvFiles(String baseFileName, double trainRatio, long seed) {

          DataFrame[] split = stratifiedSplit(trainRatio, seed);
          String[] selectedColumns = new String[getInputSize() + 1];

          System.arraycopy(
                  getInputColumns(),
                  0,
                  selectedColumns,
                  0,
                  getInputSize()
          );

          selectedColumns[getInputSize()] = getOutputColumn();

          DataFrame trainingData = split[0].cols(selectedColumns).select();
          DataFrame testData = split[1].cols(selectedColumns).select();

          Path trainPath = Path.of(
                  "target",
                  "prepared-data",
                  baseFileName + "_train.csv"
          );

          Path testPath = Path.of(
                  "target",
                  "prepared-data",
                  baseFileName + "_test.csv"
          );

          Csv.saver().createMissingDirs().save(trainingData, trainPath.toString());
          Csv.saver().createMissingDirs().save(testData, testPath.toString());

          System.out.println();
          System.out.println("=== STRATIFIED TRAIN/TEST CSV CREATED ===");
          System.out.println("Training rows: " + trainingData.height());
          System.out.println("Test rows: " + testData.height());
          System.out.println("Train path: " + trainPath);
          System.out.println("Test path: " + testPath);

          return new String[]{
                  trainPath.toString(),
                  testPath.toString()
          };
      }
    
    // Splits the dataset while preserving class distribution and ensuring
    // that the test set size is divisible by the batch size.
    public DataFrame[] stratifiedSplitForBatch(double trainRatio, long seed, int batchSize) {

          if (trainRatio <= 0 || trainRatio >= 1) {
              throw new IllegalArgumentException("trainRatio must be between 0 and 1");
          }

          if (batchSize <= 0) {
              throw new IllegalArgumentException("batchSize must be greater than 0");
          }

          List<Integer> fraudRows = new ArrayList<>();
          List<Integer> nonFraudRows = new ArrayList<>();

          for (int row = 0; row < dataSet.height(); row++) {
              double label = Double.parseDouble(dataSet.getColumn(getOutputColumn()).get(row).toString());

              if (label == 1.0) {
                  fraudRows.add(row);
              } else if (label == 0.0) {
                  nonFraudRows.add(row);
              } else {
                  throw new IllegalArgumentException("Unexpected target value: " + label);
              }
          }

          Random random = new Random(seed);

          Collections.shuffle(fraudRows, random);
          Collections.shuffle(nonFraudRows, random);

          int desiredTestSize = (int) Math.round(dataSet.height() * (1.0 - trainRatio));
          int testSize =(int) Math.round((double) desiredTestSize / batchSize)* batchSize;
          double fraudRatio = fraudRows.size() / (double) dataSet.height();
          int fraudTestSize = (int) Math.round(testSize * fraudRatio);
          int nonFraudTestSize = testSize - fraudTestSize;

          if (fraudTestSize > fraudRows.size() || nonFraudTestSize > nonFraudRows.size()) {
              throw new IllegalArgumentException("Not enough rows to create the requested batch-compatible split");
          }

          List<Integer> testRows = new ArrayList<>();
          List<Integer> trainRows = new ArrayList<>();

          testRows.addAll(fraudRows.subList(0, fraudTestSize));
          testRows.addAll(nonFraudRows.subList(0, nonFraudTestSize));
          trainRows.addAll(fraudRows.subList(fraudTestSize, fraudRows.size()));
          trainRows.addAll(nonFraudRows.subList(nonFraudTestSize, nonFraudRows.size()));

          Collections.shuffle(trainRows, random);
          Collections.shuffle(testRows, random);

          int[] trainIndices = trainRows.stream().mapToInt(Integer::intValue).toArray();
          int[] testIndices = testRows.stream().mapToInt(Integer::intValue).toArray();

          DataFrame trainingData = dataSet.rows(trainIndices).select();
          DataFrame testData = dataSet.rows(testIndices).select();

          double trainFraudRate = (fraudRows.size() - fraudTestSize)* 100.0 / trainingData.height();
          double testFraudRate = fraudTestSize* 100.0 / testData.height();

          System.out.println();
          System.out.println("=== STRATIFIED BATCH-COMPATIBLE TRAIN/TEST SPLIT ===");
          System.out.printf("Training set: %d rows | Fraud: %d (%.2f%%)%n", trainingData.height(), fraudRows.size() - fraudTestSize, trainFraudRate);
          System.out.printf("Test set: %d rows | Fraud: %d (%.2f%%)%n", testData.height(), fraudTestSize, testFraudRate);
          System.out.printf("Test set batches: %d x %d%n", testData.height() / batchSize, batchSize);

          return new DataFrame[]{
                  trainingData,
                  testData
          };
      }

    // Creates batch-compatible stratified training and test sets.
    // Saves both sets as separate CSV files for model training and evaluation.
    public String[] createStratifiedTrainTestCsvFilesForBatch(String filePrefix, double trainRatio, long seed, int batchSize) throws IOException {

          DataFrame[] split = stratifiedSplitForBatch(trainRatio, seed, batchSize);

          DataFrame trainingData = split[0];
          DataFrame testData = split[1];

          String[] selectedColumns = new String[getInputColumns().length + 1];

          System.arraycopy(getInputColumns(), 0, selectedColumns, 0, getInputColumns().length);

          selectedColumns[selectedColumns.length - 1] = getOutputColumn();

          DataFrame preparedTrainingData = trainingData.cols(selectedColumns).select();
          DataFrame preparedTestData = testData.cols(selectedColumns).select();

          Path outputDirectory = Paths.get("target", "prepared-data");

          Files.createDirectories(outputDirectory);

          Path trainPath = outputDirectory.resolve(filePrefix + "_train.csv");
          Path testPath = outputDirectory.resolve(filePrefix + "_test.csv");

          Csv.save(preparedTrainingData, trainPath.toString());
          Csv.save(preparedTestData, testPath.toString());

          System.out.println();
          System.out.println("=== STRATIFIED BATCH-COMPATIBLE TRAIN/TEST CSV CREATED ===");
          System.out.println("Training rows: " + preparedTrainingData.height());
          System.out.println("Test rows: " + preparedTestData.height());
          System.out.println("Train path: " + trainPath);
          System.out.println("Test path: " + testPath);

          return new String[]{
                  trainPath.toString(),
                  testPath.toString()
          };
      }
    
    //Helper methods
    private boolean isMissingValue(Object value) {
        if (value == null) {
            return true;
        }

        String text = value.toString().trim();

        return text.isEmpty()
                || text.equalsIgnoreCase("NaN")
                || text.equalsIgnoreCase("NULL")
                || text.equals("-");
    }
    
    private String inferColumnType(Series<?> column) {
        boolean hasValue = false;
        boolean isInteger = true;
        boolean isDouble = true;

        for (Object value : column) {
            if (isMissingValue(value)) continue;
            
            hasValue = true;
            String text = value.toString().trim();

            try {
                Integer.parseInt(text);
            } catch (NumberFormatException e) {
                isInteger = false;
            }

            try {
                Double.parseDouble(text);
            } catch (NumberFormatException e) {
                isDouble = false;
            }

            if (!isInteger && !isDouble) return "String";
        }

        if (!hasValue) return "Unknown";
        if (isInteger) return "Integer";
        if (isDouble) return "Double";
        return "String";
    }
    
    // Prints the columns selected for model training.
    public void printSelectedColumns() {
        System.out.println();
        System.out.println("=== SELECTED COLUMNS ===");
        System.out.println("Input columns:");

        String[] inputColumns = getInputColumns();

        for (int i = 0; i < inputColumns.length; i++) {
            System.out.println(" " + (i + 1) + ". " + inputColumns[i]);
        }
        System.out.println("");
        System.out.println("Target column: " + getOutputColumn());
    }
    
}