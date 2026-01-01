package com.xrpprediction.repository;

import com.xrpprediction.model.Layer2Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface Layer2PredictionRepository extends JpaRepository<Layer2Prediction, Long> {

    Optional<Layer2Prediction> findByDate(LocalDate date);

    List<Layer2Prediction> findByDateBetweenOrderByDateAsc(LocalDate startDate, LocalDate endDate);

    List<Layer2Prediction> findAllByOrderByDateDesc();

    List<Layer2Prediction> findTop30ByOrderByDateDesc();

    @Query("SELECT p FROM Layer2Prediction p ORDER BY p.date DESC LIMIT 1")
    Optional<Layer2Prediction> findLatest();

}
