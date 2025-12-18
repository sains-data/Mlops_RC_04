# MLOps Pothole Detection System
## Implementasi YOLOv8 dengan End-to-End MLOps Pipeline

---

## 👥 Tim Pengembang

- Member 1: Feryadi Yulius (122450087)
- Member 2: Syadza Puspadari Azhar (122450072)
- Member 3: Dinda Nababan (122450120)
- Member 4: Alyya 1224500
---

## 📚 Documentation

**→ [START HERE: Step-by-Step Guide](STEP_BY_STEP.md)** 

## 📋 Latar Belakang

### Permasalahan
- Kerusakan jalan (pothole) menimbulkan risiko kecelakaan
- Deteksi manual tidak efisien dan memakan waktu
- Perlu sistem otomatis untuk deteksi real-time

### Solusi
- Implementasi Deep Learning dengan YOLOv8 untuk deteksi pothole
- Pipeline MLOps untuk automasi training hingga deployment
- Monitoring dan tracking untuk menjaga kualitas model

---

## 🎯 Tujuan Proyek

### Objektif Utama
1. **Data Pipeline** → Automasi pengolahan data
2. **Model Training** → Training dengan experiment tracking
3. **Model Evaluation** → Validasi performa model
4. **Deployment** → REST API & Container Docker
5. **Monitoring** → Tracking performa production
6. **CI/CD** → Automasi testing dan deployment

---

## 🏗️ Arsitektur Sistem

### Komponen Utama
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Dataset   │ --> │   Training   │ --> │  MLflow     │
│  Validation │     │   Pipeline   │     │  Tracking   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User UI    │ <-- │   FastAPI    │ <-- │   Model     │
│  (Streamlit)│     │   Server     │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Monitoring  │
                    │   System     │
                    └──────────────┘
```

---

## 🔧 Teknologi yang Digunakan

### Machine Learning
- **YOLOv8** → Model object detection
- **Ultralytics** → Framework training
- **PyTorch** → Deep learning engine

### MLOps Tools
- **MLflow** → Experiment tracking & model registry
- **Optuna** → Hyperparameter tuning
- **DVC** → Data version control

### Backend & API
- **FastAPI** → REST API server
- **Streamlit** → User interface
- **Docker** → Containerization

### CI/CD & Testing
- **GitHub Actions** → Automation pipeline
- **Pytest** → Unit testing
- **Pre-commit** → Code quality

---

## 📊 Dataset

### Informasi Dataset
- **Sumber**: Roboflow Pothole Detection Dataset
- **Total Images**: 665 gambar
- **Train**: 477 gambar (72%)
- **Validation**: 143 gambar (21%)
- **Test**: 45 gambar (7%)
- **Classes**: 1 (Pothole)

### Preprocessing
- Resize: 640x640
- Normalization
- Augmentation: flip, rotation, brightness

---

## 🚀 Pipeline MLOps

### 1. Data Ingestion
- Validasi struktur dataset
- Exploratory Data Analysis (EDA)
- Quality checks

### 2. Training Pipeline
- Multi-model training (YOLOv8n, YOLOv8s)
- Hyperparameter tuning dengan Optuna
- Experiment tracking dengan MLflow
- Model versioning

### 3. Evaluation
- **Metrics**: Precision, Recall, F1-Score, mAP
- Confusion matrix
- Test set evaluation
- Model comparison

### 4. Deployment
- Model serving via FastAPI
- Docker containerization
- Multi-model support
- Load balancing

### 5. Monitoring
- Inference latency tracking
- Error rate monitoring
- Input drift detection
- Performance metrics

---

## 💻 Implementasi Teknis

### Model Training
```python
# CLI command
python cli.py train --model yolov8n --epochs 100

# Hyperparameter tuning
python cli.py tune --n-trials 50
```

### Model Serving
```python
# Start API server
python cli.py serve

# Access API: http://localhost:8000
```

### Docker Deployment
```bash
docker-compose up --build

# Services:
# - MLflow: http://localhost:5000
# - FastAPI: http://localhost:8000
# - User UI: http://localhost:8501
# - Admin UI: http://localhost:8502
```

---


## 🎨 User Interface

### User App (Streamlit)
- Upload gambar untuk deteksi
- Real-time inference
- Visualisasi hasil deteksi
- Download hasil

### Admin App
- Model management
- Performance monitoring
- Experiment comparison
- System health check

---

## ✅ Testing & Quality Assurance

### Test Coverage
- Unit tests: 85%
- Integration tests
- API endpoint tests
- Data validation tests

### CI/CD Pipeline
```
Push → Tests → Build → Deploy
 ↓       ↓       ↓       ↓
Code   Pytest  Docker  Production
```

---

## 🔍 Monitoring & Observability

### Metrics Tracked
1. **Model Performance**
   - Accuracy, Precision, Recall
   - Inference latency
   
2. **System Metrics**
   - API response time
   - Error rates
   - Resource usage

3. **Data Quality**
   - Input distribution
   - Drift detection

---

---

## 🎓 Lessons Learned

### Technical
- Importance of experiment tracking
- Docker containerization benefits
- CI/CD automation value

### MLOps Best Practices
- Version everything (code, data, model)
- Monitor continuously
- Automate repetitive tasks
- Test thoroughly

---

## 🔮 Future Improvements

### Short Term
- [ ] Model quantization untuk inference lebih cepat
- [ ] Add more augmentation techniques
- [ ] Improve UI/UX

### Long Term
- [ ] Multi-class detection (berbagai jenis kerusakan)
- [ ] Edge deployment (mobile/embedded)
- [ ] Real-time video processing
- [ ] Integration dengan GIS system

---

## 📚 Referensi

1. **YOLOv8 Documentation**: https://docs.ultralytics.com/
2. **MLflow**: https://mlflow.org/
3. **Base Project**: https://github.com/prsdm/mlops-project
4. **FastAPI**: https://fastapi.tiangolo.com/

---

## 📞 Kontak

**Repository**: https://github.com/sains-data/Mlops_RC_04  


---

## 🏗️ Project Structure

```
.
├── src/
│   ├── data/              # Data ingestion, validation, preprocessing
│   ├── training/          # Model training scripts
│   ├── evaluation/        # Model evaluation and testing
│   ├── inference/         # Inference logic
│   ├── api/              # FastAPI endpoints
│   ├── monitoring/       # Monitoring and drift detection
│   └── utils/            # Utility functions
├── ui/                   # Streamlit UI (User & Admin)
├── configs/              # Configuration files
├── tests/                # Unit and integration tests
├── .github/workflows/    # CI/CD workflows
├── dataset/              # Dataset directory
├── models/               # Saved models
├── mlruns/              # MLflow tracking
├── docker-compose.yml   # Docker orchestration
└── cli.py               # CLI interface
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3. Train Model

```bash
python cli.py train --model yolov8n --epochs 100
```

### 4. Start MLflow Server

```bash
mlflow ui --port 5000
```

### 5. Start API Server

```bash
python cli.py serve
```

### 6. Launch UI

```bash
# User Interface
streamlit run ui/user_app.py

# Admin Interface
streamlit run ui/admin_app.py
```

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Services:
- MLflow: http://localhost:5000
- FastAPI: http://localhost:8000
- User UI: http://localhost:8501
- Admin UI: http://localhost:8502

## 📊 Features

### Data Pipeline
- ✅ Data validation and integrity checks
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data preprocessing and augmentation

### Training
- ✅ Multi-model training (YOLOv8n, YOLOv8s)
- ✅ Hyperparameter tuning with Optuna
- ✅ Experiment tracking with MLflow

### Evaluation
- ✅ Comprehensive metrics (Precision, Recall, F1, mAP)
- ✅ Confusion matrix
- ✅ Test set evaluation

### Deployment
- ✅ FastAPI REST API
- ✅ Multi-model serving
- ✅ Docker containerization

### Monitoring
- ✅ Inference latency tracking
- ✅ Error rate monitoring
- ✅ Input drift detection

### UI
- ✅ User interface for image upload and detection
- ✅ Admin interface for model management

### CI/CD
- ✅ Automated testing
- ✅ Model training pipeline
- ✅ Deployment automation

## 📈 MLflow Tracking

All experiments are tracked including:
- Hyperparameters
- Metrics (mAP, loss, precision, recall)
- Model artifacts
- Confusion matrices

## 🔧 CLI Commands

```bash
# Training
python cli.py train --model yolov8n --epochs 100 --batch 16

# Evaluation
python cli.py evaluate --model-path models/best.pt

# Testing
python cli.py test --model-path models/best.pt

# Hyperparameter Tuning
python cli.py tune --n-trials 50

# Start API Server
python cli.py serve

# Data Validation
python cli.py validate-data
```

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📅 Project Timeline

**Due Date**: December 15, 2025  
**Team Size**: Maximum 4 people

## 🔗 References

Based on: [https://github.com/prsdm/mlops-project](https://github.com/prsdm/mlops-project)

## 📝 License
