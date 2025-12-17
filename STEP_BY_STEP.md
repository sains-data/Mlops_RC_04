# 🚀 STEP-BY-STEP EXECUTION GUIDE

## Panduan Lengkap Menjalankan Project MLOps Pothole Detection

---

## 📋 Prerequisites Checklist

- [ ] Python 3.10 atau lebih tinggi terinstall
- [ ] Git terinstall
- [ ] (Optional) Docker & Docker Compose terinstall
- [ ] Dataset pothole dalam format YOLO sudah siap

---

## 🔧 FASE 1: SETUP ENVIRONMENT

### Step 1.1: Clone atau Setup Project

```powershell
# Jika dari GitHub
git clone <your-repo-url>
cd pothole-detection-mlops

# Atau sudah di folder ini
cd "c:\Users\HP 14\Downloads\Semester 7\MLops\New folder (7)"
```

### Step 1.2: Buat Virtual Environment

```powershell
# Buat virtual environment
python -m venv venv

# Aktivasi (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Jika ada error, jalankan:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 1.3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install semua requirements
pip install -r requirements.txt

# Verify installation
python -c "import ultralytics; print('YOLOv8 installed successfully')"
```

✅ **Checkpoint 1**: Environment siap!

---

## 📊 FASE 2: PREPARE DATASET

### Step 2.1: Download Dataset

Download dataset pothole dalam format YOLO:
- https://public.roboflow.com/object-detection/pothole
  

### Step 2.2: Setup Dataset Structure

```powershell
# Struktur yang dibutuhkan:
# dataset/
#   ├── train/
#   │   ├── images/  (gambar .jpg/.png)
#   │   └── labels/  (file .txt format YOLO)
#   ├── val/
#   │   ├── images/
#   │   └── labels/
#   └── test/
#       ├── images/
#       └── labels/

# Buat folder jika belum ada
mkdir -p dataset/train/images dataset/train/labels
mkdir -p dataset/val/images dataset/val/labels
mkdir -p dataset/test/images dataset/test/labels
```

### Step 2.3: Validate Dataset

```powershell
# Validasi dataset
python cli.py validate-data

# Output yang diharapkan:
# ✓ Dataset structure is valid
# ✓ No corrupt images found
# ✓ All images have labels
```

### Step 2.4: Analyze Dataset (EDA)

```powershell
# Jalankan EDA
python cli.py analyze-data

# Hasil akan tersimpan di: outputs/eda/
# - class_distribution.png
# - image_dimensions.png
# - train_sample_annotations.png
# - val_sample_annotations.png
# - test_sample_annotations.png
# - eda_report.json
```

✅ **Checkpoint 2**: Dataset validated & analyzed!

---

## 🎯 FASE 3: MODEL TRAINING

### Step 3.1: Start MLflow Server (Terminal 1)

```powershell
# Buka terminal baru
mlflow ui --port 5000

# MLflow UI available at: http://localhost:5000
# Biarkan terminal ini tetap running
```

### Step 3.2: Train Baseline Model (Terminal 2)

```powershell
# Train YOLOv8n (model kecil, cepat)
python cli.py train --model yolov8n --epochs 100 --batch 16

# Training akan:
# 1. Load dataset dari configs/data.yaml
# 2. Train selama 100 epochs
# 3. Log metrics ke MLflow
# 4. Save best model ke runs/train/yolov8n_experiment/weights/best.pt

# Waktu training: ~30-60 menit (tergantung hardware & dataset size)
```

### Step 3.3: Train Improved Model

```powershell
# Train YOLOv8s (model lebih besar, akurasi lebih tinggi)
python cli.py train --model yolov8s --epochs 100 --batch 16

# Waktu training: ~60-120 menit
```

### Step 3.4: Monitor Training

1. Buka MLflow UI: http://localhost:5000
2. Pilih experiment "pothole-detection"
3. Compare runs
4. Check metrics: mAP@0.5, precision, recall

✅ **Checkpoint 3**: Models trained!

---

## 📈 FASE 4: MODEL EVALUATION

### Step 4.1: Evaluate Models

```powershell
# Evaluate YOLOv8n
python cli.py evaluate runs/train/yolov8n_experiment/weights/best.pt

# Evaluate YOLOv8s
python cli.py evaluate runs/train/yolov8s_experiment/weights/best.pt
```

### Step 4.2: Test on Test Set

```powershell
# Test YOLOv8n on test set
python cli.py test runs/train/yolov8n_experiment/weights/best.pt

# Test YOLOv8s on test set
python cli.py test runs/train/yolov8s_experiment/weights/best.pt
```

### Step 4.3: Compare Models

```powershell
# Compare all trained models
python cli.py compare-models

# Output: outputs/evaluation/model_comparison.png
```

✅ **Checkpoint 4**: Models evaluated!

---

## 🔬 FASE 5: HYPERPARAMETER TUNING (Optional)

### Step 5.1: Run Hyperparameter Tuning

```powershell
# Jalankan tuning dengan Optuna (50 trials)
python cli.py tune --model yolov8n --n-trials 50

# Waktu: ~5-10 jam (tergantung n-trials dan hardware)
# Best parameters akan di-log ke MLflow
```

### Step 5.2: Train with Best Parameters

```powershell
# Check best parameters di MLflow UI
# Train model baru dengan parameters tersebut
python cli.py train --model yolov8n --epochs 150 --batch 32
```

✅ **Checkpoint 5**: Hyperparameters optimized!

---

## 🚀 FASE 6: DEPLOYMENT (LOCAL)

### Step 6.1: Start FastAPI Server (Terminal 3)

```powershell
# Start API server
python cli.py serve

# API available at:
# - Base: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

### Step 6.2: Test API

```powershell
# Test dengan curl (Terminal baru)
curl http://localhost:8000/health

# Atau buka browser:
# http://localhost:8000/docs
```

### Step 6.3: Start User Interface (Terminal 4)

```powershell
# Start user UI
streamlit run ui/user_app.py

# User UI available at: http://localhost:8501
```

### Step 6.4: Start Admin Interface (Terminal 5)

```powershell
# Start admin UI
streamlit run ui/admin_app.py

# Admin UI available at: http://localhost:8502
```

✅ **Checkpoint 6**: All services running!

### Services URLs Summary:
- **MLflow**: http://localhost:5000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **User UI**: http://localhost:8501
- **Admin UI**: http://localhost:8502

---

## 🐳 FASE 7: DEPLOYMENT (DOCKER)

### Step 7.1: Stop All Local Services

```powershell
# Stop semua terminal yang running services
# Ctrl+C di setiap terminal
```

### Step 7.2: Build Docker Images

```powershell
# Build semua images
docker-compose build

# Waktu: ~10-20 menit (first time)
```

### Step 7.3: Start All Services

```powershell
# Start semua services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 7.4: Test Docker Deployment

```powershell
# Test health
curl http://localhost:8000/health

# Test UI
# Browser: http://localhost:8501 (User UI)
# Browser: http://localhost:8502 (Admin UI)
```

### Step 7.5: Stop Docker Services

```powershell
# Stop services
docker-compose down

# Remove volumes (optional)
docker-compose down -v
```

✅ **Checkpoint 7**: Docker deployment working!

---

## 🧪 FASE 8: TESTING & QUALITY

### Step 8.1: Run Unit Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
# Browser: htmlcov/index.html
```

### Step 8.2: Code Quality Checks

```powershell
# Lint code
flake8 src --count --statistics

# Format code
black src tests ui cli.py

# Check formatting
black --check src
```

✅ **Checkpoint 8**: Tests passing!

---

## 📊 FASE 9: USE THE SYSTEM

### For Users:

1. **Upload Image**
   - Open: http://localhost:8501
   - Upload road image
   - Select model (yolov8n/yolov8s)
   - Click "Detect Potholes"
   - View results with bounding boxes

2. **API Usage**
   ```powershell
   # Using curl
   curl -X POST "http://localhost:8000/predict" \
     -F "file=@your_image.jpg" \
     -F "model_name=yolov8n"
   ```

### For Admins:

1. **Monitor System**
   - Open: http://localhost:8502
   - View Overview dashboard
   - Check model status
   - Monitor experiments

2. **Manage Models**
   - Compare model performance
   - View metrics
   - Check experiment history

3. **MLflow Tracking**
   - Open: http://localhost:5000
   - Compare experiments
   - Download artifacts
   - Register models

✅ **Checkpoint 9**: System in production use!

---

## 🔄 FASE 10: CI/CD SETUP

### Step 10.1: Push to GitHub

```powershell
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: Complete MLOps project"

# Create GitHub repo and push
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

### Step 10.2: Setup GitHub Secrets

Go to GitHub repo → Settings → Secrets and variables → Actions

Add secrets:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password
- `DEPLOY_HOST`: Production server IP
- `DEPLOY_USER`: Deployment user
- `DEPLOY_SSH_KEY`: SSH private key
- `SLACK_WEBHOOK`: Slack webhook URL (optional)

### Step 10.3: Trigger CI/CD

```powershell
# Make a change and push
git add .
git commit -m "Test CI/CD"
git push

# Check GitHub Actions tab
# CI/CD akan otomatis running
```

✅ **Checkpoint 10**: CI/CD active!

---

## 🎯 COMMON WORKFLOWS

### Retrain Model with New Data:

```powershell
# 1. Add new data to dataset
# 2. Validate
python cli.py validate-data

# 3. Retrain
python cli.py train --model yolov8n --epochs 50

# 4. Evaluate
python cli.py evaluate runs/train/yolov8n_experiment/weights/best.pt

# 5. Deploy if better
# Copy best.pt to models/yolov8n_best.pt
```

### Update and Redeploy:

```powershell
# 1. Stop services
docker-compose down

# 2. Update code/models

# 3. Rebuild
docker-compose build

# 4. Restart
docker-compose up -d
```

---

## 🐛 TROUBLESHOOTING

### Issue: Dataset not found
```powershell
# Check dataset structure
ls dataset/train/images
ls dataset/train/labels

# Re-run validation
python cli.py validate-data
```

### Issue: MLflow connection error
```powershell
# Check if MLflow server is running
curl http://localhost:5000/health

# Restart MLflow
mlflow ui --port 5000
```

### Issue: API not responding
```powershell
# Check if models exist
ls runs/train/*/weights/best.pt

# Restart API
python cli.py serve
```

### Issue: Docker build fails
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

# ITERA 2025