package com.xrpprediction.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Layer1Stats {

    private Double mae;              // Mean Absolute Error
    private Double rmse;             // Root Mean Squared Error
    private Double r2Score;          // R² Score
    private Double directionAccuracy; // Direction Accuracy %
    
    private Double meanError;
    private Double maxError;
    private Double minError;
    private Double stdError;
    
    private Integer totalSamples;
    private Double meanErrorPct;

}
