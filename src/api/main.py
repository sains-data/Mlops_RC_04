"""
FastAPI Application for Model Serving
"""
# Import torch configuration FIRST to patch torch.load for PyTorch 2.6+ compatibility
from src.utils.torch_config import configure_torch_for_ultralytics
import logging
from pathlib import Path
from typing import List, Optional
import yaml
import io
from datetime import datetime
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import mlflow

# Configure PyTorch to allow Ultralytics models (PyTorch 2.6+ security)
try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn import modules as ultralytics_modules
    from torch.nn.modules.container import Sequential
    from collections import OrderedDict
    import torch.nn as nn
    
    # Add all necessary safe globals for YOLO models
    safe_classes = [
        DetectionModel,
        Sequential,
        OrderedDict,
    ]
    
    # Add PyTorch nn modules
    safe_classes.extend([
        nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Upsample,
        nn.Module, nn.ModuleList, nn.Identity, nn.MaxPool2d,
        nn.ReLU, nn.LeakyReLU, nn.Dropout, nn.Linear,
    ])
    
    # Add all Ultralytics custom modules
    for name in dir(ultralytics_modules):
        obj = getattr(ultralytics_modules, name)
        if isinstance(obj, type):
            safe_classes.append(obj)
    
    torch.serialization.add_safe_globals(safe_classes)
except Exception:
    pass  # Fallback for older PyTorch versions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pothole Detection API",
    description="MLOps API for Pothole Detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Model cache
models_cache = {}


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_name: str
    num_detections: int
    detections: List[dict]
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    available_models: List[str]


def load_model(model_name: str) -> YOLO:
    """
    Load YOLO model
    
    Args:
        model_name: Name of the model
        
    Returns:
        YOLO model instance
    """
    if model_name in models_cache:
        return models_cache[model_name]
    
    model_path = Path("models") / f"{model_name}_best.pt"
    
    if not model_path.exists():
        # Try loading from runs
        model_path = Path("runs/train") / f"{model_name}_experiment" / "weights" / "best.pt"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found. Available models: {get_available_models()}"
        )
    
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    models_cache[model_name] = model
    
    return model


def get_available_models() -> List[str]:
    """Get list of available models"""
    models = []
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*_best.pt"):
            models.append(model_file.stem.replace("_best", ""))
    
    # Check runs directory
    runs_dir = Path("runs/train")
    if runs_dir.exists():
        for experiment_dir in runs_dir.iterdir():
            if experiment_dir.is_dir():
                best_pt = experiment_dir / "weights" / "best.pt"
                if best_pt.exists():
                    model_name = experiment_dir.name.replace("_experiment", "")
                    if model_name not in models:
                        models.append(model_name)
    
    return models


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_models": get_available_models()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_models": get_available_models()
    }


@app.get("/models")
async def list_models():
    """List all available models"""
    models = get_available_models()
    return {
        "models": models,
        "count": len(models)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query("yolov8n", description="Model to use for prediction"),
    conf_threshold: float = Query(0.25, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, description="IOU threshold")
):
    """
    Perform prediction on uploaded image
    
    Args:
        file: Uploaded image file
        model_name: Model to use
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        
    Returns:
        Prediction results
    """
    try:
        # Record start time
        start_time = datetime.now()
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Load model
        model = load_model(model_name)
        
        # Run inference
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": "pothole",
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Prediction completed: {len(detections)} detections in {inference_time:.2f}ms")
        
        return {
            "model_name": model_name,
            "num_detections": len(detections),
            "detections": detections,
            "inference_time_ms": inference_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/visualize")
async def predict_visualize(
    file: UploadFile = File(...),
    model_name: str = Query("yolov8n", description="Model to use for prediction"),
    conf_threshold: float = Query(0.25, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, description="IOU threshold")
):
    """
    Perform prediction and return image with bounding boxes
    
    Args:
        file: Uploaded image file
        model_name: Model to use
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        
    Returns:
        Image with bounding boxes
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Load model
        model = load_model(model_name)
        
        # Run inference
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Draw results on image
        annotated_img = results[0].plot()
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_bytes = io.BytesIO(buffer.tobytes())
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    model_name: str = Query("yolov8n", description="Model to use for prediction"),
    conf_threshold: float = Query(0.25, description="Confidence threshold")
):
    """
    Perform batch prediction on multiple images
    
    Args:
        files: List of uploaded image files
        model_name: Model to use
        conf_threshold: Confidence threshold
        
    Returns:
        Batch prediction results
    """
    try:
        results = []
        
        # Load model
        model = load_model(model_name)
        
        for file in files:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Run inference
            pred_results = model.predict(img, conf=conf_threshold, verbose=False)
            
            # Process results
            detections = []
            for result in pred_results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "class_name": "pothole",
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x1": float(box.xyxy[0][0]),
                            "y1": float(box.xyxy[0][1]),
                            "x2": float(box.xyxy[0][2]),
                            "y2": float(box.xyxy[0][3])
                        }
                    }
                    detections.append(detection)
            
            results.append({
                "filename": file.filename,
                "num_detections": len(detections),
                "detections": detections
            })
        
        return {
            "total_images": len(files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=config['api']['reload']
    )
