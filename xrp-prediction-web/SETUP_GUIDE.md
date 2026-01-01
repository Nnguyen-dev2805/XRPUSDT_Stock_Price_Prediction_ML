# 📚 Hướng dẫn Chi Tiết - XRP Price Prediction Website

## 1. 📦 Cài đặt và Khởi chạy

### Bước 1: Đảm bảo có Java 17

```bash
java -version
# Output should show: openjdk version "17.x.x"
```

Nếu không có, tải từ: https://adoptium.net/

### Bước 2: Xây dựng Project

```bash
cd xrp-prediction-web
mvn clean install
```

Lần đầu có thể mất 3-5 phút để download dependencies.

### Bước 3: Chạy Application

```bash
mvn spring-boot:run
```

Hoặc:

```bash
java -jar target/xrp-prediction-web-1.0.0.jar
```

Bạn sẽ thấy:
```
2026-01-01 10:00:00.000  INFO 1234 --- [main] XrpPredictionApplication : Started XrpPredictionApplication
```

### Bước 4: Truy cập Website

Mở trình duyệt và vào: **http://localhost:8080**

---

## 2. 📊 Nhập Dữ liệu Dự đoán

### Từ Jupyter Notebook

Sau khi chạy xong notebook dự đoán (`TrainStock2Layer.ipynb`):

```python
# Thêm vào cuối notebook của bạn:

import sys
sys.path.append('../xrp-prediction-web')
from data_exporter import PredictionDataExporter

# Khởi tạo exporter
exporter = PredictionDataExporter(output_dir='../xrp-prediction-web/data/exports')

# Export dữ liệu
export_data = exporter.export_predictions(
    cleaned_data=cleaned_data,  # DataFrame từ notebook của bạn
    df_clean=df_clean,
    predictions_dict={
        '1D': predictions['1D'] if '1D' in predictions else None,
        '3D': predictions['3D'] if '3D' in predictions else None,
        '5D': predictions['5D'] if '5D' in predictions else None,
        '7D': predictions['7D'] if '7D' in predictions else None,
    }
)

# Generate SQL insert statements
exporter.generate_sql_insert(export_data)

print("✓ Data exported successfully!")
```

### Nhập vào Spring Boot

#### Cách 1: Tạo endpoint import (Recommended)

Thêm vào `DashboardController.java`:

```java
@PostMapping("/api/import/csv")
@ResponseBody
public ResponseEntity<String> importCSV(
    @RequestParam("file") MultipartFile file) {
    try {
        predictionService.loadPredictionsFromCSV(file.getInputStream());
        return ResponseEntity.ok("Data imported successfully!");
    } catch (IOException e) {
        return ResponseEntity.badRequest().body("Import failed: " + e.getMessage());
    }
}
```

Rồi upload file CSV qua form.

#### Cách 2: Sử dụng SQL trực tiếp

1. Vào **H2 Console**: http://localhost:8080/h2-console
2. JDBC URL: `jdbc:h2:mem:xrppredictiondb`
3. Paste nội dung từ file `insert_predictions.sql`
4. Click "Run"

---

## 3. 🎨 Tùy chỉnh Website

### Đổi cổng (port)

Sửa `application.properties`:

```properties
server.port=9090
```

Rồi chạy lại: http://localhost:9090

### Thay đổi tiêu đề

Sửa `dashboard.html`:

```html
<h1>🚀 Dự đoán Giá XRP - VN Trading</h1>
```

### Đổi màu sắc

Sửa `style.css`:

```css
:root {
    --primary-color: #FF6B00;  /* Cam thay cho xanh */
    --secondary-color: #00D9FF;
    /* ... các màu khác ... */
}
```

---

## 4. 📈 Thêm Dữ liệu Real-time

Để lấy dữ liệu XRP real-time, hãy tích hợp API:

```python
import requests

def get_xrp_price():
    """Fetch XRP price from CoinGecko"""
    response = requests.get(
        'https://api.coingecko.com/api/v3/simple/price',
        params={'ids': 'ripple', 'vs_currencies': 'usd'}
    )
    return response.json()['ripple']['usd']

# Chạy mỗi giờ để cập nhật giá
```

Hoặc dùng API khác như:
- **Binance API**: https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT
- **CoinMarketCap**: https://coinmarketcap.com/api/

---

## 5. 🗄️ Chuyển sang Database Production

### Để sử dụng PostgreSQL thay vì H2:

1. Thêm dependency vào `pom.xml`:

```xml
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
    <scope>runtime</scope>
</dependency>
```

2. Sửa `application.properties`:

```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/xrp_prediction
spring.datasource.username=postgres
spring.datasource.password=your_password
spring.datasource.driver-class-name=org.postgresql.Driver

spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
spring.jpa.hibernate.ddl-auto=update
```

3. Rebuild:
```bash
mvn clean install
```

---

## 6. 🚀 Deploy lên Server

### Docker Deployment

1. Tạo `Dockerfile`:

```dockerfile
FROM openjdk:17-slim
COPY target/xrp-prediction-web-1.0.0.jar app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
```

2. Build image:

```bash
mvn clean package
docker build -t xrp-prediction-web .
```

3. Chạy container:

```bash
docker run -p 8080:8080 xrp-prediction-web
```

### Heroku Deployment

1. Tạo `Procfile`:

```
web: java -Dserver.port=$PORT $JAVA_OPTS -jar target/*.jar
```

2. Deploy:

```bash
heroku create your-app-name
git push heroku main
```

---

## 7. 🔧 Troubleshooting

### Port 8080 đã được sử dụng

```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### Maven download quá chậm

Sửa `~/.m2/settings.xml`:

```xml
<mirrors>
    <mirror>
        <id>aliyun</id>
        <name>Aliyun Maven Mirror</name>
        <url>https://maven.aliyun.com/repository/public</url>
        <mirrorOf>*</mirrorOf>
    </mirror>
</mirrors>
```

### Không có dữ liệu hiển thị

1. Kiểm tra H2 console: http://localhost:8080/h2-console
2. Chạy SQL insert hoặc upload CSV
3. Refresh trang

---

## 8. 📱 Responsive Testing

Kiểm tra website trên mobile:

```bash
# Chạy trên LAN
mvn spring-boot:run -Dserver.address=0.0.0.0
```

Sau đó vào: `http://<your-ip>:8080` từ điện thoại

---

## 9. 📊 API Documentation

### Endpoints available:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Dashboard page |
| GET | `/predictions` | Predictions history |
| GET | `/analysis` | Model analysis |
| GET | `/api/dashboard-stats` | Stats JSON |
| GET | `/api/chart-data` | Chart data JSON |
| POST | `/api/import/csv` | Import CSV (khi implement) |

### Example API calls:

```bash
# Get dashboard stats
curl http://localhost:8080/api/dashboard-stats | jq

# Get chart data
curl http://localhost:8080/api/chart-data | jq '.[:5]'
```

---

## 10. 🔐 Security Best Practices

Khi deploy production:

```properties
# application.properties

# Disable H2 console
spring.h2.console.enabled=false

# HTTPS
server.ssl.key-store=classpath:keystore.jks
server.ssl.key-store-password=your_password

# CORS
spring.web.cors.allowed-origins=https://yourdomain.com
```

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra logs: Terminal console khi chạy `mvn spring-boot:run`
2. Xem H2 console: http://localhost:8080/h2-console
3. Kiểm tra browser console (F12)
4. Xem file `application.properties` settings

---

**Chúc bạn thành công!** 🎉

Nếu cần thêm tính năng, hãy yêu cầu!
