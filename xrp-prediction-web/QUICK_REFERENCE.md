# ⚡ QUICK REFERENCE - XRP Prediction Web

## 🚀 Bắt đầu nhanh (Windows)

```batch
cd d:\CODE\AI\project_predict\xrp-prediction-web
quickstart.bat
# Chọn option 3: Build and Run
# Mở browser: http://localhost:8080
```

## 🚀 Bắt đầu nhanh (Linux/Mac)

```bash
cd ~/CODE/AI/project_predict/xrp-prediction-web
chmod +x quickstart.sh
./quickstart.sh
# Chọn option 3: Build and Run
# Mở browser: http://localhost:8080
```

## 🚀 Bắt đầu nhanh (Manual)

```bash
cd d:\CODE\AI\project_predict\xrp-prediction-web

# Build
mvn clean install

# Run
mvn spring-boot:run

# Vào browser
http://localhost:8080
```

---

## 📁 Các thư mục quan trọng

| Thư mục | Mục đích |
|---------|---------|
| `src/main/java` | Java source code (Controllers, Services, Models) |
| `src/main/resources/templates` | HTML pages (Thymeleaf) |
| `src/main/resources/static` | CSS, JavaScript, images |
| `src/main/resources` | `application.properties` (config) |
| `data/` | CSV data files |
| `target/` | Build output (auto-generated) |

---

## 🌐 URL Endpoints

| URL | Mô tả |
|-----|--------|
| `http://localhost:8080/` | 🏠 Dashboard |
| `http://localhost:8080/predictions` | 📊 Lịch sử dự đoán |
| `http://localhost:8080/analysis` | 📈 Phân tích model |
| `http://localhost:8080/h2-console` | 🗄️ Database console |

---

## 📊 Nhập dữ liệu từ Jupyter

```python
# 1. Chạy xong notebook TrainStock2Layer.ipynb
# 2. Thêm vào cuối notebook:

from sys import path
path.append('../xrp-prediction-web')
from data_exporter import PredictionDataExporter

exporter = PredictionDataExporter()
data = exporter.export_predictions(cleaned_data)
exporter.generate_sql_insert(data)

# 3. Vào http://localhost:8080/h2-console
# 4. Copy & Paste nội dung insert_predictions.sql
# 5. Refresh dashboard
```

---

## 🔧 Thay đổi cổng (Port)

Sửa `src/main/resources/application.properties`:

```properties
server.port=9090
```

---

## 🗄️ Chuyển sang Database khác

### PostgreSQL

1. Thêm dependency vào `pom.xml`:
```xml
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
</dependency>
```

2. Sửa `application.properties`:
```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/xrp_db
spring.datasource.username=postgres
spring.datasource.password=password
spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
```

3. Rebuild: `mvn clean install`

---

## 🐛 Troubleshooting

### ❌ "Port 8080 is already in use"
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### ❌ "mvn is not recognized"
Cài Maven: https://maven.apache.org/download.cgi

### ❌ "Java version not matching"
Cần Java 17: https://adoptium.net/

### ❌ "No data showing"
1. Vào H2 Console: http://localhost:8080/h2-console
2. Run SQL insert commands
3. Refresh dashboard

---

## 📚 File cấu hình

- `pom.xml` - Maven dependencies
- `application.properties` - Spring Boot config
- `src/main/resources/static/css/style.css` - Styling
- `src/main/resources/templates/*.html` - Pages

---

## 🎨 Tùy chỉnh

### Đổi tiêu đề
`src/main/resources/templates/dashboard.html` - Dòng 32
```html
<h1>🚀 Your Custom Title</h1>
```

### Đổi màu chính
`src/main/resources/static/css/style.css` - Dòng 8-9
```css
--primary-color: #3498db;    /* Thay đổi */
--secondary-color: #2ecc71;
```

### Thêm logo
Copy ảnh vào `src/main/resources/static/` rồi:
```html
<img src="/logo.png" alt="Logo">
```

---

## 🚀 Deploy

### Docker
```bash
mvn clean package
docker build -t xrp-web .
docker run -p 8080:8080 xrp-web
```

### JAR file
```bash
mvn clean package
java -jar target/xrp-prediction-web-1.0.0.jar
```

---

## 📞 Hỗ trợ

- **Maven issues**: `mvn clean -U install`
- **Port conflicts**: Thay port trong `application.properties`
- **Database errors**: Xóa `target/` folder rồi rebuild
- **Chart issues**: Kiểm tra browser console (F12)

---

## 🎯 Next Steps

1. ✅ Build & Run Spring Boot application
2. ✅ Export data from Jupyter notebook
3. ✅ Import data vào database
4. ✅ View dashboard
5. ⏭️ Customize colors & layout
6. ⏭️ Add real-time updates
7. ⏭️ Deploy to production

---

**Good luck!** 🎉
