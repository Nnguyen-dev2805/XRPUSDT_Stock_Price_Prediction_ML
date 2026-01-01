package com.xrpprediction.util;

import com.xrpprediction.model.Layer1Prediction;
import com.xrpprediction.model.Layer2Prediction;
import com.xrpprediction.repository.Layer1PredictionRepository;
import com.xrpprediction.repository.Layer2PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Initialize test data for Layer 1 and Layer 2 predictions
 * Runs automatically on application startup
 */
@Component
public class DataInitializer implements CommandLineRunner {

    @Autowired
    private Layer1PredictionRepository layer1Repository;

    @Autowired
    private Layer2PredictionRepository layer2Repository;

    @Override
    public void run(String... args) throws Exception {
        // Only initialize if database is empty
        if (layer1Repository.count() > 0) {
            System.out.println("✓ Test data already exists. Skipping initialization.");
            return;
        }

        System.out.println("\n📊 Initializing test data for Layer 1 and Layer 2...\n");

        // Generate test data for last 100 days
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(100);

        initializeLayer1Data(startDate, endDate);
        initializeLayer2Data(startDate, endDate);

        System.out.println("✓ Test data initialization complete!\n");
    }

    private void initializeLayer1Data(LocalDate startDate, LocalDate endDate) {
        List<Layer1Prediction> predictions = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility

        LocalDate currentDate = startDate;
        double basePrice = 0.52; // Starting XRP price

        while (!currentDate.isAfter(endDate)) {
            // Generate realistic price data
            double priceToday = basePrice + (random.nextDouble() - 0.5) * 0.02; // ±1% variation
            double rfPredTomorrow = priceToday + (random.nextDouble() - 0.5) * 0.015; // ±0.75%
            double actualTomorrow = rfPredTomorrow + (random.nextDouble() - 0.5) * 0.01; // ±0.5% error
            double error = Math.abs(rfPredTomorrow - actualTomorrow);
            double errorPercentage = (error / actualTomorrow) * 100;

            Layer1Prediction pred = new Layer1Prediction();
            pred.setDate(currentDate);
            pred.setPriceToday(priceToday);
            pred.setRfPredTomorrow(rfPredTomorrow);
            pred.setActualTomorrow(actualTomorrow);
            pred.setError(error);
            pred.setErrorPercentage(errorPercentage);

            predictions.add(pred);

            basePrice = actualTomorrow; // Update base price for next iteration
            currentDate = currentDate.plusDays(1);
        }

        layer1Repository.saveAll(predictions);
        System.out.println("✓ Layer 1: " + predictions.size() + " predictions saved");
    }

    private void initializeLayer2Data(LocalDate startDate, LocalDate endDate) {
        List<Layer2Prediction> predictions = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility

        LocalDate currentDate = startDate;
        double basePrice = 0.52; // Starting XRP price

        while (!currentDate.isAfter(endDate)) {
            // Generate realistic data
            double open = basePrice;
            double vol = 100000000 + random.nextDouble() * 50000000; // 100M-150M
            double rfPredToday = open + (random.nextDouble() - 0.5) * 0.015;
            double actualClose = rfPredToday + (random.nextDouble() - 0.5) * 0.01;

            // Ridge prediction: intercept + 0.8543*rf_pred + 0.0234*open - 0.0001*vol
            double ridgePredicted = 0.5234 + (0.8543 * rfPredToday) + (0.0234 * open) + (-0.0001 * vol);

            double error = Math.abs(ridgePredicted - actualClose);
            double errorPercentage = (error / actualClose) * 100;

            Layer2Prediction pred = new Layer2Prediction();
            pred.setDate(currentDate);
            pred.setOpen(open);
            pred.setVol(vol);
            pred.setRfPredToday(rfPredToday);
            pred.setActualClose(actualClose);
            pred.setRidgePredicted(ridgePredicted);
            pred.setError(error);
            pred.setErrorPercentage(errorPercentage);

            predictions.add(pred);

            basePrice = actualClose; // Update base price for next iteration
            currentDate = currentDate.plusDays(1);
        }

        layer2Repository.saveAll(predictions);
        System.out.println("✓ Layer 2: " + predictions.size() + " predictions saved");
    }
}
