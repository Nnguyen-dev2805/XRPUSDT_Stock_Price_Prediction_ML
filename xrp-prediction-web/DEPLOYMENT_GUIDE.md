# 🎯 XRP Price Prediction - Spring Boot Website

Website dự đoán giá **XRP/USDT** sử dụng 2-Layer Ensemble Model (Random Forest + Ridge Regression)

## 📋 Cấu trúc Dự án

```
xrp-prediction-web/
├── src/main/java/com/xrpprediction/
│   ├── controller/        # REST Controllers (Layer1, Layer2)
│   ├── model/             # JPA Entities + DTOs
│   ├── repository/        # JPA Repository Interfaces
│   ├── service/           # Business Logic (Stats, Chart Data)
│   └── util/              # DataInitializer
├── src/main/resources/
│   ├── templates/         # Thymeleaf HTML Templates
│   ├── application.properties
│   └── static/            # CSS, JS, Images
├── pom.xml               # Maven Configuration
└── target/               # Compiled JAR
```

## 🚀 Chạy Ứng dụng

### 1️⃣ Build Project
```bash
mvn clean install -DskipTests
```

### 2️⃣ Run Application
```bash
java -jar target/xrp-prediction-web-1.0.0.jar --server.port=5555
```

Ứng dụng sẽ tự động:
- ✅ Tạo H2 Database
- ✅ Khởi tạo test data (101 predictions cho Layer 1 & 2)

### 3️⃣ Truy cập Website

- **Layer 1 Dashboard**: http://localhost:5555/layer1
- **Layer 2 Dashboard**: http://localhost:5555/layer2
- **H2 Database Console**: http://localhost:5555/h2-console

## 📊 API Endpoints

### Layer 1 (Random Forest)
- `GET /layer1/` - Dashboard HTML
- `GET /layer1/api/stats` - JSON statistics (MAE, RMSE, R², Accuracy)
- `GET /layer1/api/chart-data` - JSON chart data
- `GET /layer1/api/predictions` - JSON all predictions
- `POST /layer1/api/predictions` - Create new prediction
- `GET /layer1/api/export-csv` - Export as CSV

### Layer 2 (Ridge Regression)
- `GET /layer2/` - Dashboard HTML
- `GET /layer2/api/stats` - JSON statistics
- `GET /layer2/api/chart-data` - JSON chart data
- `GET /layer2/api/coefficients` - Ridge coefficients
- `GET /layer2/api/predictions` - JSON all predictions
- `POST /layer2/api/predictions` - Create new prediction
- `GET /layer2/api/export-csv` - Export as CSV

## 📈 Dữ liệu

### Layer 1 Predictions
- **Input**: Price Today + 100+ technical features
- **Output**: RF Predicted Tomorrow
- **Metrics**: MAE, RMSE, R² Score, Direction Accuracy

### Layer 2 Predictions
- **Input**: RF_Pred_Today + Open + Volume
- **Output**: Ridge Predicted Close
- **Model**: `Predicted = 0.5234 + 0.8543×RF_Pred + 0.0234×Open - 0.0001×Vol`

## 🛠️ Công nghệ

- **Backend**: Spring Boot 3.2.0, Java 17, Spring Data JPA
- **Frontend**: Thymeleaf, Bootstrap 5, Chart.js
- **Database**: H2 In-Memory (Production: PostgreSQL)
- **Build**: Maven 3.8.9+

## 📝 File Quan trọng

| File | Mô tả |
|------|-------|
| `Layer1Prediction.java` | Entity cho Layer 1 predictions |
| `Layer2Prediction.java` | Entity cho Layer 2 predictions |
| `Layer1Service.java` | Logic tính toán stats Layer 1 |
| `Layer2Service.java` | Logic tính toán stats Layer 2 |
| `DataInitializer.java` | Tự động khởi tạo test data |
| `layer1.html` | Thymeleaf template Layer 1 |
| `layer2.html` | Thymeleaf template Layer 2 |

## ✨ Đặc Điểm

- ✅ Dashboard với Real-time Charts (Chart.js)
- ✅ Responsive Design (Bootstrap 5)
- ✅ Automatic Test Data Loading
- ✅ CSV Export Functionality
- ✅ Statistical Analysis (Error metrics)
- ✅ RESTful API endpoints
- ✅ H2 Database Console

## 🔗 Kết nối Dữ liệu CSV

Để load dữ liệu từ file CSV thực tế:

```python
# data_exporter.py
import pandas as pd
predictions_df = pd.read_csv('predictions.csv')
# Convert to JSON và POST đến /layer1/api/predictions
```

## 📞 Liên Hệ

Tạo bởi GitHub Copilot - Dự đoán Giá XRP/USDT với Machine Learning
