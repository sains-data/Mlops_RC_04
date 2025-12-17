# 🚧 MLOps Pothole Detection - YOLOv8

End-to-end MLOps pipeline for pothole detection using YOLOv8n and YOLOv8s with complete CI/CD, experiment tracking, and monitoring.

# Team
- Member 1: Feryadi Yulius (122450087)
- Member 2: Syadza Puspadari Azhar (122450072)
- Member 3: Dinda Nababan 1224500
- Member 4: Alyya 1224500

## 📚 Documentation

**→ [START HERE: Step-by-Step Guide](STEP_BY_STEP.md)** ⭐

## 🎯 Project Objectives

- ✅ Build end-to-end MLOps pipeline for pothole detection
- ✅ Implement CI/CD for Machine Learning
- ✅ Experiment tracking and model monitoring
- ✅ Provide User and Admin UI

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
