package com.xrpprediction.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Layer2Coefficients {

    private Double intercept;
    private Double rfCoeff;      // Coefficient for RF_Pred_Today
    private Double openCoeff;    // Coefficient for Open
    private Double volCoeff;     // Coefficient for Vol

}
