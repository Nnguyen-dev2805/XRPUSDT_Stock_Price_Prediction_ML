package com.xrpprediction.controller;

import com.xrpprediction.model.Layer1Prediction;
import com.xrpprediction.model.Layer1Stats;
import com.xrpprediction.service.Layer1Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@Controller
@RequestMapping("/layer1")
public class Layer1Controller {

    @Autowired
    private Layer1Service layer1Service;

    /**
     * Display Layer 1 Dashboard
     */
    @GetMapping("")
    public String dashboard(Model model) {
        // Get statistics
        Layer1Stats stats = layer1Service.calculateStats();
        model.addAttribute("layer1Stats", stats);

        // Get chart data
        Layer1Service.ChartData chartData = layer1Service.getChartData();
        model.addAttribute("layer1ChartData", chartData);

        // Get predictions for table
        List<Layer1Prediction> predictions = layer1Service.getLatestPredictions(30);
        model.addAttribute("layer1Predictions", predictions);

        // Get test period
        if (!predictions.isEmpty()) {
            model.addAttribute("testStartDate", predictions.get(0).getDate());
            model.addAttribute("testEndDate", predictions.get(predictions.size() - 1).getDate());
        }

        return "layer1";
    }

    /**
     * API: Get Layer 1 Statistics
     */
    @GetMapping("/api/stats")
    @ResponseBody
    public ResponseEntity<Layer1Stats> getStats() {
        Layer1Stats stats = layer1Service.calculateStats();
        return ResponseEntity.ok(stats);
    }

    /**
     * API: Get Layer 1 Chart Data
     */
    @GetMapping("/api/chart-data")
    @ResponseBody
    public ResponseEntity<Layer1Service.ChartData> getChartData() {
        Layer1Service.ChartData chartData = layer1Service.getChartData();
        return ResponseEntity.ok(chartData);
    }

    /**
     * API: Get all predictions
     */
    @GetMapping("/api/predictions")
    @ResponseBody
    public ResponseEntity<List<Layer1Prediction>> getAllPredictions() {
        List<Layer1Prediction> predictions = layer1Service.getAllPredictions();
        return ResponseEntity.ok(predictions);
    }

    /**
     * API: Create new prediction
     */
    @PostMapping("/api/predictions")
    @ResponseBody
    public ResponseEntity<Layer1Prediction> createPrediction(@RequestBody Layer1Prediction prediction) {
        prediction.setCreatedAt(LocalDate.now());
        Layer1Prediction saved = layer1Service.savePrediction(prediction);
        return ResponseEntity.ok(saved);
    }

    /**
     * API: Export CSV
     */
    @GetMapping("/api/export-csv")
    public ResponseEntity<?> exportCsv() {
        List<Layer1Prediction> predictions = layer1Service.getAllPredictions();
        
        StringBuilder csv = new StringBuilder();
        csv.append("Date,Price_Today,RF_Pred_Tomorrow,Actual_Tomorrow,Error,Error_Percentage\n");
        
        for (Layer1Prediction p : predictions) {
            csv.append(p.getDate()).append(",")
                    .append(p.getPriceToday()).append(",")
                    .append(p.getRfPredTomorrow()).append(",")
                    .append(p.getActualTomorrow()).append(",")
                    .append(p.getError()).append(",")
                    .append(p.getErrorPercentage()).append("\n");
        }

        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=\"layer1_predictions.csv\"")
                .body(csv.toString());
    }

}
