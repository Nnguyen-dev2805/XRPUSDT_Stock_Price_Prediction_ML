package com.xrpprediction.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResult {

    private LocalDate date;
    private Double actualPrice;
    private Double predictedPrice1D;
    private Double predictedPrice3D;
    private Double predictedPrice5D;
    private Double predictedPrice7D;

    private Double error;
    private Double errorPercentage;

    private String direction; // UP, DOWN, NEUTRAL

    public PredictionResult(LocalDate date, Double actualPrice, 
                           Double pred1D, Double pred3D, Double pred5D, Double pred7D) {
        this.date = date;
        this.actualPrice = actualPrice;
        this.predictedPrice1D = pred1D;
        this.predictedPrice3D = pred3D;
        this.predictedPrice5D = pred5D;
        this.predictedPrice7D = pred7D;

        if (pred1D != null && actualPrice != null) {
            this.error = actualPrice - pred1D;
            this.errorPercentage = (error / actualPrice) * 100;
        }
    }

}
