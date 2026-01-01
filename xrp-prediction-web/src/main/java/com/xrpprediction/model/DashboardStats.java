package com.xrpprediction.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DashboardStats {

    private Double currentPrice;
    private Double priceChange24h;
    private Double priceChangePercent;
    private Double highPrice24h;
    private Double lowPrice24h;
    private Long volume24h;

    // Prediction metrics
    private Double mae1D;  // Mean Absolute Error 1 Day
    private Double mae3D;  // Mean Absolute Error 3 Days
    private Double mae5D;  // Mean Absolute Error 5 Days
    private Double mae7D;  // Mean Absolute Error 7 Days

    private Double directionAccuracy; // Percentage

    private Integer totalPredictions;
    private Integer accuratePredictions;

    // Latest predictions
    private Double predictedPrice1D;
    private Double predictedPrice3D;
    private Double predictedPrice5D;
    private Double predictedPrice7D;

    private Double changePercent1D;
    private Double changePercent3D;
    private Double changePercent5D;
    private Double changePercent7D;

}
