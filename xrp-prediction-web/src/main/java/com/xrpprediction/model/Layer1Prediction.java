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
@Table(name = "layer1_predictions")
public class Layer1Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDate date;

    @Column(nullable = false)
    private Double priceToday;

    @Column(nullable = false)
    private Double rfPredTomorrow;

    @Column
    private Double actualTomorrow;

    @Column
    private Double error;

    @Column
    private Double errorPercentage;

    @Column(name = "created_at")
    private LocalDate createdAt;

}
