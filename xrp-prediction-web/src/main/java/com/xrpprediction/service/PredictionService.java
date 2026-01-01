package com.xrpprediction.service;

import com.xrpprediction.model.DashboardStats;
import com.xrpprediction.model.PriceData;
import com.xrpprediction.model.PredictionResult;
import com.xrpprediction.repository.PriceDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class PredictionService {

    @Autowired
    private PriceDataRepository priceDataRepository;

    public DashboardStats getDashboardStats() {
        DashboardStats stats = new DashboardStats();

        // Get latest price data
        Optional<PriceData> latestOpt = priceDataRepository.findLatest();

        if (latestOpt.isPresent()) {
            PriceData latest = latestOpt.get();
            stats.setCurrentPrice(latest.getClose());
            stats.setHighPrice24h(latest.getHigh());
            stats.setLowPrice24h(latest.getLow());
            stats.setVolume24h(latest.getVolume());

            // Predicted prices
            stats.setPredictedPrice1D(latest.getPredictedPrice1D());
            stats.setPredictedPrice3D(latest.getPredictedPrice3D());
            stats.setPredictedPrice5D(latest.getPredictedPrice5D());
            stats.setPredictedPrice7D(latest.getPredictedPrice7D());

            // Calculate change percentages
            if (latest.getClose() != null && latest.getPredictedPrice1D() != null) {
                stats.setChangePercent1D(((latest.getPredictedPrice1D() - latest.getClose()) / latest.getClose()) * 100);
            }
            if (latest.getClose() != null && latest.getPredictedPrice3D() != null) {
                stats.setChangePercent3D(((latest.getPredictedPrice3D() - latest.getClose()) / latest.getClose()) * 100);
            }
            if (latest.getClose() != null && latest.getPredictedPrice5D() != null) {
                stats.setChangePercent5D(((latest.getPredictedPrice5D() - latest.getClose()) / latest.getClose()) * 100);
            }
            if (latest.getClose() != null && latest.getPredictedPrice7D() != null) {
                stats.setChangePercent7D(((latest.getPredictedPrice7D() - latest.getClose()) / latest.getClose()) * 100);
            }
        }

        // Calculate metrics
        List<PriceData> allData = priceDataRepository.findAllByOrderByDateDesc();
        stats.setTotalPredictions(allData.size());

        // Calculate MAE
        stats.setMae1D(calculateMAE(allData, "1D"));
        stats.setMae3D(calculateMAE(allData, "3D"));
        stats.setMae5D(calculateMAE(allData, "5D"));
        stats.setMae7D(calculateMAE(allData, "7D"));

        // Calculate direction accuracy
        stats.setDirectionAccuracy(calculateDirectionAccuracy(allData));

        return stats;
    }

    public List<PredictionResult> getPredictionHistory(int days) {
        List<PriceData> dataList = priceDataRepository.findTop30ByOrderByDateDesc();

        return dataList.stream()
                .sorted(Comparator.comparing(PriceData::getDate))
                .map(p -> new PredictionResult(
                        p.getDate(),
                        p.getClose(),
                        p.getPredictedPrice1D(),
                        p.getPredictedPrice3D(),
                        p.getPredictedPrice5D(),
                        p.getPredictedPrice7D()
                ))
                .collect(Collectors.toList());
    }

    public List<Map<String, Object>> getChartData() {
        List<PriceData> dataList = priceDataRepository.findTop30ByOrderByDateDesc();

        return dataList.stream()
                .sorted(Comparator.comparing(PriceData::getDate))
                .map(p -> {
                    Map<String, Object> data = new LinkedHashMap<>();
                    data.put("date", p.getDate().toString());
                    data.put("actual", p.getClose());
                    data.put("pred1d", p.getPredictedPrice1D());
                    data.put("pred3d", p.getPredictedPrice3D());
                    data.put("pred5d", p.getPredictedPrice5D());
                    data.put("pred7d", p.getPredictedPrice7D());
                    return data;
                })
                .collect(Collectors.toList());
    }

    private Double calculateMAE(List<PriceData> data, String horizon) {
        return data.stream()
                .filter(p -> {
                    if ("1D".equals(horizon)) return p.getPredictedPrice1D() != null;
                    if ("3D".equals(horizon)) return p.getPredictedPrice3D() != null;
                    if ("5D".equals(horizon)) return p.getPredictedPrice5D() != null;
                    if ("7D".equals(horizon)) return p.getPredictedPrice7D() != null;
                    return false;
                })
                .mapToDouble(p -> {
                    double predicted, actual = p.getClose();
                    if ("1D".equals(horizon)) predicted = p.getPredictedPrice1D();
                    else if ("3D".equals(horizon)) predicted = p.getPredictedPrice3D();
                    else if ("5D".equals(horizon)) predicted = p.getPredictedPrice5D();
                    else predicted = p.getPredictedPrice7D();

                    return Math.abs(predicted - actual);
                })
                .average()
                .orElse(0.0);
    }

    private Double calculateDirectionAccuracy(List<PriceData> data) {
        if (data.size() < 2) return 0.0;

        long correct = 0;
        for (int i = 0; i < data.size() - 1; i++) {
            PriceData curr = data.get(i);
            PriceData next = data.get(i + 1);

            if (curr.getPredictedPrice1D() != null && next.getClose() != null) {
                boolean predicted_up = curr.getPredictedPrice1D() > curr.getClose();
                boolean actual_up = next.getClose() > curr.getClose();
                if (predicted_up == actual_up) {
                    correct++;
                }
            }
        }

        return (double) correct / (data.size() - 1) * 100;
    }

    public void loadPredictionsFromCSV(String csvPath) {
        // This will be implemented to load data from CSV files
        // For now, it's a placeholder
    }

}
