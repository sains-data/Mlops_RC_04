# Makefile for Pothole Detection MLOps Project

.PHONY: help install setup clean test lint format train serve docker-up docker-down

# Default target
help:
	@echo "🚧 Pothole Detection MLOps - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install all dependencies"
	@echo "  make setup            - Complete project setup"
	@echo ""
	@echo "Development:"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code with black"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo ""
	@echo "Data & Training:"
	@echo "  make validate-data    - Validate dataset"
	@echo "  make analyze-data     - Run EDA"
	@echo "  make train            - Train YOLOv8n model"
	@echo "  make train-all        - Train all models"
	@echo "  make tune             - Run hyperparameter tuning"
	@echo ""
	@echo "Deployment:"
	@echo "  make serve            - Start API server"
	@echo "  make ui-user          - Start user interface"
	@echo "  make ui-admin         - Start admin interface"
	@echo "  make mlflow           - Start MLflow server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start all services with Docker"
	@echo "  make docker-down      - Stop all Docker services"
	@echo "  make docker-logs      - View Docker logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean temporary files"
	@echo "  make clean-all        - Clean everything including models"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

setup: install
	@echo "🔧 Setting up project..."
	python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['dataset/train/images', 'dataset/train/labels', 'dataset/val/images', 'dataset/val/labels', 'dataset/test/images', 'dataset/test/labels', 'models', 'logs', 'outputs']]"
	@echo "✅ Project setup complete!"

# Development
lint:
	@echo "🔍 Running linter..."
	flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "✨ Formatting code..."
	black src tests ui cli.py

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Data & Training
validate-data:
	@echo "✅ Validating dataset..."
	python cli.py validate-data

analyze-data:
	@echo "📊 Analyzing dataset..."
	python cli.py analyze-data

train:
	@echo "🚀 Training YOLOv8n model..."
	python cli.py train --model yolov8n --epochs 100

train-all:
	@echo "🚀 Training all models..."
	python cli.py train-all

tune:
	@echo "🎯 Running hyperparameter tuning..."
	python cli.py tune --model yolov8n --n-trials 50

# Deployment
serve:
	@echo "🚀 Starting API server..."
	python cli.py serve

ui-user:
	@echo "🌐 Starting user interface..."
	streamlit run ui/user_app.py

ui-admin:
	@echo "🛠️ Starting admin interface..."
	streamlit run ui/admin_app.py

mlflow:
	@echo "📊 Starting MLflow server..."
	mlflow ui --port 5000

# Docker
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started:"
	@echo "   - MLflow: http://localhost:5000"
	@echo "   - API: http://localhost:8000"
	@echo "   - User UI: http://localhost:8501"
	@echo "   - Admin UI: http://localhost:8502"

docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "📋 Viewing Docker logs..."
	docker-compose logs -f

# Utilities
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

clean-all: clean
	@echo "🧹 Cleaning everything..."
	rm -rf mlruns/ runs/ models/*.pt logs/ outputs/
	@echo "⚠️  All training artifacts removed!"
