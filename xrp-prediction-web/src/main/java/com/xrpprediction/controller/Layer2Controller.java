package com.xrpprediction.controller;

import com.xrpprediction.model.Layer2Prediction;
import com.xrpprediction.model.Layer2Stats;
import com.xrpprediction.model.Layer2Coefficients;
import com.xrpprediction.service.Layer2Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@Controller
@RequestMapping("/layer2")
public class Layer2Controller {

    @Autowired
    private Layer2Service layer2Service;

    /**
     * Display Layer 2 Dashboard
     */
    @GetMapping("")
    public String dashboard(Model model) {
        // Get statistics
        Layer2Stats stats = layer2Service.calculateStats();
        model.addAttribute("layer2Stats", stats);

        // Get chart data
        Layer2Service.ChartData chartData = layer2Service.getChartData();
        model.addAttribute("layer2ChartData", chartData);

        // Get coefficients
        Layer2Coefficients coefficients = layer2Service.getCoefficients();
        model.addAttribute("layer2Coeff", coefficients);

        // Get predictions for table
        List<Layer2Prediction> predictions = layer2Service.getLatestPredictions(30);
        model.addAttribute("layer2Predictions", predictions);

        // Get test period
        if (!predictions.isEmpty()) {
            model.addAttribute("testStartDate", predictions.get(0).getDate());
            model.addAttribute("testEndDate", predictions.get(predictions.size() - 1).getDate());
        }

        return "layer2";
    }

    /**
     * API: Get Layer 2 Statistics
     */
    @GetMapping("/api/stats")
    @ResponseBody
    public ResponseEntity<Layer2Stats> getStats() {
        Layer2Stats stats = layer2Service.calculateStats();
        return ResponseEntity.ok(stats);
    }

    /**
     * API: Get Layer 2 Chart Data
     */
    @GetMapping("/api/chart-data")
    @ResponseBody
    public ResponseEntity<Layer2Service.ChartData> getChartData() {
        Layer2Service.ChartData chartData = layer2Service.getChartData();
        return ResponseEntity.ok(chartData);
    }

    /**
     * API: Get all predictions
     */
    @GetMapping("/api/predictions")
    @ResponseBody
    public ResponseEntity<List<Layer2Prediction>> getAllPredictions() {
        List<Layer2Prediction> predictions = layer2Service.getAllPredictions();
        return ResponseEntity.ok(predictions);
    }

    /**
     * API: Get coefficients
     */
    @GetMapping("/api/coefficients")
    @ResponseBody
    public ResponseEntity<Layer2Coefficients> getCoefficients() {
        Layer2Coefficients coefficients = layer2Service.getCoefficients();
        return ResponseEntity.ok(coefficients);
    }

    /**
     * API: Create new prediction
     */
    @PostMapping("/api/predictions")
    @ResponseBody
    public ResponseEntity<Layer2Prediction> createPrediction(@RequestBody Layer2Prediction prediction) {
        prediction.setCreatedAt(LocalDate.now());
        Layer2Prediction saved = layer2Service.savePrediction(prediction);
        return ResponseEntity.ok(saved);
    }

    /**
     * API: Export CSV
     */
    @GetMapping("/api/export-csv")
    public ResponseEntity<?> exportCsv() {
        List<Layer2Prediction> predictions = layer2Service.getAllPredictions();
        
        StringBuilder csv = new StringBuilder();
        csv.append("Date,Open,Vol,RF_Pred_Today,Actual_Close,Ridge_Predicted,Error,Error_Percentage\n");
        
        for (Layer2Prediction p : predictions) {
            csv.append(p.getDate()).append(",")
                    .append(p.getOpen()).append(",")
                    .append(p.getVol()).append(",")
                    .append(p.getRfPredToday()).append(",")
                    .append(p.getActualClose()).append(",")
                    .append(p.getRidgePredicted()).append(",")
                    .append(p.getError()).append(",")
                    .append(p.getErrorPercentage()).append("\n");
        }

        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=\"layer2_predictions.csv\"")
                .body(csv.toString());
    }

}
