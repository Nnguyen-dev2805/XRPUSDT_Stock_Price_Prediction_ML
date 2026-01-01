package com.xrpprediction.service;

import com.xrpprediction.model.Layer2Prediction;
import com.xrpprediction.model.Layer2Stats;
import com.xrpprediction.model.Layer2Coefficients;
import com.xrpprediction.repository.Layer2PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class Layer2Service {

    @Autowired
    private Layer2PredictionRepository layer2PredictionRepository;

    /**
     * Get all Layer 2 predictions
     */
    public List<Layer2Prediction> getAllPredictions() {
        return layer2PredictionRepository.findAllByOrderByDateDesc();
    }

    /**
     * Get latest 30 predictions for chart
     */
    public List<Layer2Prediction> getLatestPredictions(int limit) {
        return layer2PredictionRepository.findTop30ByOrderByDateDesc()
                .stream()
                .sorted((a, b) -> a.getDate().compareTo(b.getDate()))
                .collect(Collectors.toList());
    }

    /**
     * Calculate statistics for Layer 2
     */
    public Layer2Stats calculateStats() {
        List<Layer2Prediction> predictions = getAllPredictions();

        if (predictions.isEmpty()) {
            return new Layer2Stats();
        }

        Layer2Stats stats = new Layer2Stats();
        stats.setTotalSamples(predictions.size());

        // Calculate MAE
        Double mae = predictions.stream()
                .mapToDouble(p -> Math.abs(p.getRidgePredicted() - p.getActualClose()))
                .average()
                .orElse(0.0);
        stats.setMae(mae);

        // Calculate Direction Accuracy
        long correctDirection = predictions.stream()
                .filter(p -> {
                    boolean predUp = p.getRidgePredicted() > p.getOpen();
                    boolean actualUp = p.getActualClose() > p.getOpen();
                    return predUp == actualUp;
                })
                .count();
        
        Double directionAccuracy = (double) correctDirection / predictions.size() * 100;
        stats.setDirectionAccuracy(directionAccuracy);

        // Error statistics
        List<Double> errors = predictions.stream()
                .map(p -> p.getRidgePredicted() - p.getActualClose())
                .collect(Collectors.toList());

        if (!errors.isEmpty()) {
            Double meanError = errors.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            stats.setMeanError(meanError);
            stats.setMaxError(errors.stream().mapToDouble(Double::doubleValue).max().orElse(0.0));
            stats.setMinError(errors.stream().mapToDouble(Double::doubleValue).min().orElse(0.0));

            Double stdError = Math.sqrt(errors.stream()
                    .mapToDouble(e -> Math.pow(e - meanError, 2))
                    .average()
                    .orElse(0.0));
            stats.setStdError(stdError);
        }

        // Error percentages
        Double meanErrorPct = predictions.stream()
                .mapToDouble(p -> (p.getRidgePredicted() - p.getActualClose()) / p.getActualClose() * 100)
                .average()
                .orElse(0.0);
        stats.setMeanErrorPct(meanErrorPct);

        Double meanAbsErrorPct = predictions.stream()
                .mapToDouble(p -> Math.abs((p.getRidgePredicted() - p.getActualClose()) / p.getActualClose()) * 100)
                .average()
                .orElse(0.0);
        stats.setMeanAbsErrorPct(meanAbsErrorPct);

        return stats;
    }

    /**
     * Save Layer 2 prediction
     */
    public Layer2Prediction savePrediction(Layer2Prediction prediction) {
        return layer2PredictionRepository.save(prediction);
    }

    /**
     * Get chart data
     */
    public ChartData getChartData() {
        List<Layer2Prediction> predictions = getLatestPredictions(30);

        ChartData chartData = new ChartData();
        chartData.setDates(
                predictions.stream()
                        .map(p -> p.getDate().toString())
                        .collect(Collectors.toList())
        );
        chartData.setActualPrices(
                predictions.stream()
                        .map(Layer2Prediction::getActualClose)
                        .collect(Collectors.toList())
        );
        chartData.setPredictedPrices(
                predictions.stream()
                        .map(Layer2Prediction::getRidgePredicted)
                        .collect(Collectors.toList())
        );

        return chartData;
    }

    /**
     * Get Ridge model coefficients
     */
    public Layer2Coefficients getCoefficients() {
        // These are hardcoded from the notebook model training
        // In production, you would load these from a trained model
        return new Layer2Coefficients(
                0.5234,    // Intercept
                0.8543,    // RF_Pred_Today coefficient
                0.0234,    // Open coefficient
                -0.0001    // Vol coefficient
        );
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
