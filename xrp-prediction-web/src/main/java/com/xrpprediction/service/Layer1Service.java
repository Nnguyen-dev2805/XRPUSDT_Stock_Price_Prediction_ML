package com.xrpprediction.service;

import com.xrpprediction.model.Layer1Prediction;
import com.xrpprediction.model.Layer1Stats;
import com.xrpprediction.repository.Layer1PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class Layer1Service {

    @Autowired
    private Layer1PredictionRepository layer1PredictionRepository;

    /**
     * Get all Layer 1 predictions
     */
    public List<Layer1Prediction> getAllPredictions() {
        return layer1PredictionRepository.findAllByOrderByDateDesc();
    }

    /**
     * Get latest 30 predictions for chart
     */
    public List<Layer1Prediction> getLatestPredictions(int limit) {
        return layer1PredictionRepository.findTop30ByOrderByDateDesc()
                .stream()
                .sorted((a, b) -> a.getDate().compareTo(b.getDate()))
                .collect(Collectors.toList());
    }

    /**
     * Calculate statistics for Layer 1
     */
    public Layer1Stats calculateStats() {
        List<Layer1Prediction> predictions = getAllPredictions();

        if (predictions.isEmpty()) {
            return new Layer1Stats();
        }

        Layer1Stats stats = new Layer1Stats();
        stats.setTotalSamples(predictions.size());

        // Calculate MAE
        Double mae = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .mapToDouble(p -> Math.abs(p.getRfPredTomorrow() - p.getActualTomorrow()))
                .average()
                .orElse(0.0);
        stats.setMae(mae);

        // Calculate RMSE
        Double rmse = Math.sqrt(
                predictions.stream()
                        .filter(p -> p.getActualTomorrow() != null)
                        .mapToDouble(p -> Math.pow(p.getRfPredTomorrow() - p.getActualTomorrow(), 2))
                        .average()
                        .orElse(0.0)
        );
        stats.setRmse(rmse);

        // Calculate R² Score
        Double meanActual = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .mapToDouble(Layer1Prediction::getActualTomorrow)
                .average()
                .orElse(0.0);

        Double ssRes = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .mapToDouble(p -> Math.pow(p.getActualTomorrow() - p.getRfPredTomorrow(), 2))
                .sum();

        Double ssTot = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .mapToDouble(p -> Math.pow(p.getActualTomorrow() - meanActual, 2))
                .sum();

        Double r2 = 1 - (ssRes / ssTot);
        stats.setR2Score(r2);

        // Calculate Direction Accuracy
        long correctDirection = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .filter(p -> {
                    boolean predUp = p.getRfPredTomorrow() > p.getPriceToday();
                    boolean actualUp = p.getActualTomorrow() > p.getPriceToday();
                    return predUp == actualUp;
                })
                .count();
        
        Double directionAccuracy = (double) correctDirection / predictions.size() * 100;
        stats.setDirectionAccuracy(directionAccuracy);

        // Error statistics
        List<Double> errors = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .map(p -> p.getRfPredTomorrow() - p.getActualTomorrow())
                .collect(Collectors.toList());

        if (!errors.isEmpty()) {
            stats.setMeanError(errors.stream().mapToDouble(Double::doubleValue).average().orElse(0.0));
            stats.setMaxError(errors.stream().mapToDouble(Double::doubleValue).max().orElse(0.0));
            stats.setMinError(errors.stream().mapToDouble(Double::doubleValue).min().orElse(0.0));

            Double meanErrorSq = errors.stream()
                    .mapToDouble(e -> Math.pow(e - stats.getMeanError(), 2))
                    .average()
                    .orElse(0.0);
            stats.setStdError(Math.sqrt(meanErrorSq));
        }

        // Error percentage
        Double meanErrorPct = predictions.stream()
                .filter(p -> p.getActualTomorrow() != null)
                .mapToDouble(p -> Math.abs((p.getRfPredTomorrow() - p.getActualTomorrow()) / p.getActualTomorrow() * 100))
                .average()
                .orElse(0.0);
        stats.setMeanErrorPct(meanErrorPct);

        return stats;
    }

    /**
     * Save Layer 1 prediction
     */
    public Layer1Prediction savePrediction(Layer1Prediction prediction) {
        return layer1PredictionRepository.save(prediction);
    }

    /**
     * Get chart data (dates, actual prices, predicted prices)
     */
    public ChartData getChartData() {
        List<Layer1Prediction> predictions = getLatestPredictions(30);

        ChartData chartData = new ChartData();
        chartData.setDates(
                predictions.stream()
                        .map(p -> p.getDate().toString())
                        .collect(Collectors.toList())
        );
        chartData.setActualPrices(
                predictions.stream()
                        .map(Layer1Prediction::getPriceToday)
                        .collect(Collectors.toList())
        );
        chartData.setPredictedPrices(
                predictions.stream()
                        .map(Layer1Prediction::getRfPredTomorrow)
                        .collect(Collectors.toList())
        );

        return chartData;
    }

    /**
     * DTO for chart data
     */
    public static class ChartData {
        private java.util.List<String> dates;
        private java.util.List<Double> actualPrices;
        private java.util.List<Double> predictedPrices;

        // Getters and Setters
        public java.util.List<String> getDates() { return dates; }
        public void setDates(java.util.List<String> dates) { this.dates = dates; }

        public java.util.List<Double> getActualPrices() { return actualPrices; }
        public void setActualPrices(java.util.List<Double> actualPrices) { this.actualPrices = actualPrices; }

        public java.util.List<Double> getPredictedPrices() { return predictedPrices; }
        public void setPredictedPrices(java.util.List<Double> predictedPrices) { this.predictedPrices = predictedPrices; }
    }

}
