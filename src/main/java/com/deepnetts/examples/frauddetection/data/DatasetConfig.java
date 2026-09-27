package com.deepnetts.examples.frauddetection.data;

public class DatasetConfig {

    private final String csvFile;
    private final String[] inputColumns;
    private final String targetColumn;

    public DatasetConfig(
            String csvFile,
            String[] inputColumns,
            String targetColumn) {

        this.csvFile = csvFile;
        this.inputColumns = inputColumns;
        this.targetColumn = targetColumn;
    }

    public String getCsvFile() {
        return csvFile;
    }

    public String[] getInputColumns() {
        return inputColumns;
    }

    public String getTargetColumn() {
        return targetColumn;
    }
}
