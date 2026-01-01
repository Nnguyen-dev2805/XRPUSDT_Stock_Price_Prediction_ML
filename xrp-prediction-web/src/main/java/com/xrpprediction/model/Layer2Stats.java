package com.xrpprediction.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Layer2Stats {

    private Double mae;              // Mean Absolute Error
    private Double directionAccuracy; // Direction Accuracy %
    
    private Double meanError;
    private Double maxError;
    private Double minError;
    private Double stdError;
    private Double meanErrorPct;
    private Double meanAbsErrorPct;
    
    private Integer totalSamples;

}
