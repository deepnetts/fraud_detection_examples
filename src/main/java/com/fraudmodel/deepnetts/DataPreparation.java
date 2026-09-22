package com.fraudmodel.deepnetts;

import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class DataPreparation {

    private final Table dataSet;

    public DataPreparation(String csvFile) {
        CsvReadOptions options = CsvReadOptions.builder(csvFile)
                .header(true)
                .sample(false)
                .missingValueIndicator(
                        "NaN",
                        "",
                        " ",
                        "-",
                        "nan",
                        "NULL"
                )
                .build();

        this.dataSet = Table.read().usingOptions(options);
    }

    public Table getData() {
        return dataSet;
    }

    public void printBasicInfo() {
        System.out.println();
        System.out.println("=== DATASET BASIC INFO ===");
        System.out.println("Rows: " + dataSet.rowCount());
        System.out.println("Columns: " + dataSet.columnCount());
        System.out.println("Column names: " + dataSet.columnNames());
    }

    public void printColumnInfo() {
        System.out.println();
        System.out.println("=== COLUMN STRUCTURE ===");
        System.out.println(dataSet.structure());
    }

    public void previewRows(int numberOfRows) {
        System.out.println();
        System.out.println("=== FIRST " + numberOfRows + " ROWS ===");
        System.out.println(dataSet.first(numberOfRows));
    }

    public void countMissingValues() {
        int totalRows = dataSet.rowCount();

        System.out.println();
        System.out.println("=== MISSING VALUES ===");

        for (Column<?> column : dataSet.columns()) {
            int missing = column.countMissing();
            double percent = missing * 100.0 / totalRows;

            System.out.printf(
                    "Column: %-35s | Missing: %d (%.2f%%)%n",
                    column.name(),
                    missing,
                    percent
            );
        }
    }

    public void printTargetDistribution() {
        int totalRows = dataSet.rowCount();

        int fraudCount =
                (int) dataSet.numberColumn("fraud").sum();

        int nonFraudCount =
                totalRows - fraudCount;

        double fraudRate =
                fraudCount * 100.0 / totalRows;

        double nonFraudRate =
                nonFraudCount * 100.0 / totalRows;

        System.out.println();
        System.out.println("=== TARGET DISTRIBUTION ===");
        System.out.println("Total transactions: " + totalRows);

        System.out.printf(
                "Non-fraud transactions: %d (%.2f%%)%n",
                nonFraudCount,
                nonFraudRate
        );

        System.out.printf(
                "Fraud transactions: %d (%.2f%%)%n",
                fraudCount,
                fraudRate
        );
    }

    public String createPreparedCsvFile(String outputFileName) {
        try {
            Path outputDirectory =
                    Path.of("target", "prepared-data");

            Files.createDirectories(outputDirectory);

            Path outputPath =
                    outputDirectory.resolve(outputFileName);

            Table preparedTable =
                    Table.create("credit_card_fraud");

            for (String inputColumn : getInputColumns()) {
                preparedTable.addColumns(
                        dataSet.column(inputColumn).copy()
                );
            }

            preparedTable.addColumns(
                    dataSet.column(getOutputColumn()).copy()
            );

            preparedTable.write().csv(outputPath.toString());

            System.out.println();
            System.out.println("=== PREPARED CSV CREATED ===");
            System.out.println("Path: " + outputPath);

            System.out.println(
                    "Input columns: " +
                    String.join(", ", getInputColumns())
            );

            System.out.println(
                    "Output column: " + getOutputColumn()
            );

            System.out.println(
                    "Rows: " + preparedTable.rowCount()
            );

            System.out.println(
                    "Columns: " + preparedTable.columnCount()
            );

            return outputPath.toString();

        } catch (IOException e) {
            throw new RuntimeException(
                    "Failed to create prepared CSV file.",
                    e
            );
        }
    }

    public String[] getInputColumns() {
        return new String[]{
                "distance_from_home",
                "distance_from_last_transaction",
                "ratio_to_median_purchase_price",
                "repeat_retailer",
                "used_chip",
                "used_pin_number",
                "online_order"
        };
    }

    public String getOutputColumn() {
        return "fraud";
    }

    public int getInputSize() {
        return getInputColumns().length;
    }

    public int getOutputSize() {
        return 1;
    }
}