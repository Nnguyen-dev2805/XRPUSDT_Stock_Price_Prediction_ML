package com.xrpprediction.controller;

import com.xrpprediction.model.DashboardStats;
import com.xrpprediction.model.PredictionResult;
import com.xrpprediction.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;

@Controller
public class DashboardController {

    @Autowired
    private PredictionService predictionService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @GetMapping("/")
    public String dashboard(Model model) {
        DashboardStats stats = predictionService.getDashboardStats();
        model.addAttribute("stats", stats);

        List<PredictionResult> history = predictionService.getPredictionHistory(30);
        model.addAttribute("predictions", history);

        return "dashboard";
    }

    @GetMapping("/api/dashboard-stats")
    @ResponseBody
    public DashboardStats getStats() {
        return predictionService.getDashboardStats();
    }

    @GetMapping("/api/chart-data")
    @ResponseBody
    public List<Map<String, Object>> getChartData() {
        return predictionService.getChartData();
    }

    @GetMapping("/predictions")
    public String predictions(Model model) {
        List<PredictionResult> history = predictionService.getPredictionHistory(30);
        model.addAttribute("predictions", history);
        return "predictions";
    }

    @GetMapping("/analysis")
    public String analysis(Model model) {
        DashboardStats stats = predictionService.getDashboardStats();
        model.addAttribute("stats", stats);
        return "analysis";
    }

}
