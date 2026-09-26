/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.deepnetts.examples.frauddetection.data;

/**
 *
 * @author HP
 */

public class FraudDatasetConfigs {

    public static final DatasetConfig IMBALANCED_DATASET_CONFIG =
            new DatasetConfig(
                    "src/main/resources/data/credit_card_fraud_imbalanced.csv",
                    new String[]{
                            "distance_from_home",
                            "distance_from_last_transaction",
                            "ratio_to_median_purchase_price",
                            "repeat_retailer",
                            "used_chip",
                            "used_pin_number",
                            "online_order"
                    },
                    "fraud"
            );

    public static final DatasetConfig BALANCED_DATASET_CONFIG =
            new DatasetConfig(
                    "src/main/resources/data/credit_card_fraud_balanced.csv",
                    new String[]{
                            "amount",
                            "is_cross_border",
                            "attempts_last_30min",
                            "minutes_since_previous_transaction",
                            "distance_from_previous_transaction_km",
                            "different_merchants_last_30min"
                    },
                    "fraud_label"
            );

    private FraudDatasetConfigs() {
    }
}