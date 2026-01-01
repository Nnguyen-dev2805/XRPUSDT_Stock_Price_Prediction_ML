package com.xrpprediction.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "price_data")
public class PriceData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDate date;

    @Column(nullable = false)
    private Double open;

    @Column(nullable = false)
    private Double high;

    @Column(nullable = false)
    private Double low;

    @Column(nullable = false)
    private Double close;

    @Column(nullable = false)
    private Long volume;

    @Column(name = "predicted_price_1d")
    private Double predictedPrice1D;

    @Column(name = "predicted_price_3d")
    private Double predictedPrice3D;

    @Column(name = "predicted_price_5d")
    private Double predictedPrice5D;

    @Column(name = "predicted_price_7d")
    private Double predictedPrice7D;

    @Column(name = "prediction_error_1d")
    private Double predictionError1D;

    @Column(name = "created_at")
    private LocalDate createdAt;

}
