package com.xrpprediction.repository;

import com.xrpprediction.model.Layer1Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface Layer1PredictionRepository extends JpaRepository<Layer1Prediction, Long> {

    Optional<Layer1Prediction> findByDate(LocalDate date);

    List<Layer1Prediction> findByDateBetweenOrderByDateAsc(LocalDate startDate, LocalDate endDate);

    List<Layer1Prediction> findAllByOrderByDateDesc();

    List<Layer1Prediction> findTop30ByOrderByDateDesc();

    @Query("SELECT p FROM Layer1Prediction p ORDER BY p.date DESC LIMIT 1")
    Optional<Layer1Prediction> findLatest();

}
