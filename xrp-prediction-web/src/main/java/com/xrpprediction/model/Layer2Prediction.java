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
@Table(name = "layer2_predictions")
public class Layer2Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDate date;

    @Column(nullable = false)
    private Double open;

    @Column(nullable = false)
    private Double vol;

    @Column(nullable = false)
    private Double rfPredToday;

    @Column(nullable = false)
    private Double actualClose;

    @Column(nullable = false)
    private Double ridgePredicted;

    @Column
    private Double error;

    @Column
    private Double errorPercentage;

    @Column(name = "created_at")
    private LocalDate createdAt;

}
