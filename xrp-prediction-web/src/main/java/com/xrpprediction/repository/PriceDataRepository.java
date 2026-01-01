package com.xrpprediction.repository;

import com.xrpprediction.model.PriceData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface PriceDataRepository extends JpaRepository<PriceData, Long> {

    Optional<PriceData> findByDate(LocalDate date);

    List<PriceData> findByDateBetweenOrderByDateAsc(LocalDate startDate, LocalDate endDate);

    List<PriceData> findAllByOrderByDateDesc();

    @Query("SELECT p FROM PriceData p ORDER BY p.date DESC LIMIT 1")
    Optional<PriceData> findLatest();

    List<PriceData> findTop30ByOrderByDateDesc();

}
