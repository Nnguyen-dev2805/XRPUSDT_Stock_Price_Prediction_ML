# README - XRP Price Prediction Web Application

## 🚀 Project Overview

XRP Price Prediction Web Application is a Spring Boot + Thymeleaf web interface for displaying AI-powered price predictions for XRP/USDT cryptocurrency.

## 📋 Features

- **Dashboard**: Real-time statistics and current price predictions
- **Predictions Page**: Detailed prediction history with error analysis
- **Analysis Page**: Model performance metrics and information
- **Interactive Charts**: Price charts with actual vs predicted prices
- **Responsive Design**: Mobile-friendly UI

## 🛠️ Tech Stack

- **Backend**: Spring Boot 3.2.0
- **Frontend**: Thymeleaf, HTML5, CSS3, JavaScript
- **Database**: H2 (In-memory, can be configured for production)
- **Build Tool**: Maven
- **Java Version**: 17

## 📂 Project Structure

```
xrp-prediction-web/
├── src/
│   ├── main/
│   │   ├── java/com/xrpprediction/
│   │   │   ├── XrpPredictionApplication.java
│   │   │   ├── controller/
│   │   │   │   └── DashboardController.java
│   │   │   ├── service/
│   │   │   │   └── PredictionService.java
│   │   │   ├── model/
│   │   │   │   ├── PriceData.java
│   │   │   │   ├── PredictionResult.java
│   │   │   │   └── DashboardStats.java
│   │   │   └── repository/
│   │   │       └── PriceDataRepository.java
│   │   ├── resources/
│   │   │   ├── templates/
│   │   │   │   ├── dashboard.html
│   │   │   │   ├── predictions.html
│   │   │   │   └── analysis.html
│   │   │   ├── static/
│   │   │   │   ├── css/
│   │   │   │   │   └── style.css
│   │   │   │   └── js/
│   │   │   │       └── dashboard.js
│   │   │   └── application.properties
│   └── test/
└── pom.xml
```

## 🚀 Getting Started

### Prerequisites
- Java 17 or higher
- Maven 3.8.9 or higher

### Installation

1. Navigate to the project directory:
```bash
cd xrp-prediction-web
```

2. Build the project:
```bash
mvn clean install
```

3. Run the application:
```bash
mvn spring-boot:run
```

4. Open your browser and navigate to:
```
http://localhost:8080
```

## 📊 Pages Overview

### Dashboard (/)
- Current XRP price and market statistics
- Multi-horizon price predictions (1D, 3D, 5D, 7D)
- Model performance metrics (MAE, Direction Accuracy)
- Interactive price chart with predictions
- Recent predictions table

### Predictions (/predictions)
- Detailed prediction history
- Error analysis (absolute and percentage)
- Sortable/filterable predictions table

### Analysis (/analysis)
- Model performance metrics by horizon
- Direction accuracy statistics
- Model architecture information
- Features used by the model
- Historical data information

## 🔌 API Endpoints

- `GET /api/dashboard-stats` - Get current dashboard statistics (JSON)
- `GET /api/chart-data` - Get chart data for the last 30 days (JSON)

## 📝 Configuration

Configuration is managed through `application.properties`:

```properties
server.port=8080
spring.jpa.hibernate.ddl-auto=create-drop
spring.datasource.url=jdbc:h2:mem:xrppredictiondb
```

For production, consider:
- Using a persistent database (PostgreSQL, MySQL)
- Enabling caching
- Setting up proper logging

## 🔄 Data Integration

To integrate your Python predictions:

1. Export predictions as CSV from your Jupyter notebooks
2. Create a CSV importer service in `PredictionService`
3. Load data into the H2 database
4. The web interface will automatically display predictions

Example CSV format:
```
date,open,high,low,close,volume,predicted_price_1d,predicted_price_3d,predicted_price_5d,predicted_price_7d
2024-01-01,0.5200,0.5300,0.5100,0.5250,1000000,0.5300,0.5350,0.5400,0.5450
```

## 📈 Model Information

- **Architecture**: 2-Layer Ensemble Learning
  - Layer 1: Random Forest Regressor (500 estimators)
  - Layer 2: Ridge Regression
- **Features**: Advanced technical indicators (SMA, EMA, Bollinger Bands, RSI, MACD, etc.)
- **Data Period**: 2018-2024 (2,192 days)

## 🎨 UI Features

- Modern, clean design with gradient backgrounds
- Responsive grid layouts
- Smooth animations and transitions
- Real-time updates capability
- Color-coded predictions (1D: Blue, 3D: Green, 5D: Orange, 7D: Purple)
- Interactive Chart.js graphs

## 🧪 Testing

Run tests with:
```bash
mvn test
```

## 📦 Dependencies

- Spring Boot Web Starter
- Spring Data JPA
- Thymeleaf
- H2 Database
- Apache Commons CSV
- Lombok

## 🔐 Security Notes

For production deployment:
- Enable HTTPS
- Implement user authentication
- Use environment variables for sensitive data
- Configure CORS properly
- Implement rate limiting

## 📚 Future Enhancements

- [ ] Real-time data updates via WebSocket
- [ ] User authentication and dashboards
- [ ] Export predictions as PDF/CSV
- [ ] Email notifications for price alerts
- [ ] Integration with external cryptocurrency APIs
- [ ] Advanced analytics and backtesting
- [ ] Model retraining interface
- [ ] Multi-currency support

## 📄 License

This project is part of the XRP Price Prediction analysis.

## 👤 Author

Created as part of AI-Powered Cryptocurrency Analysis project.

## 📞 Support

For issues or questions, please refer to the project documentation or create an issue.

---

**Last Updated**: January 2026
**Status**: Development
