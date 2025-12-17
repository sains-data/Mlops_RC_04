"""
Inference module for making predictions
"""

# Import torch configuration FIRST to patch torch.load for PyTorch 2.6+ compatibility
from src.utils.torch_config import configure_torch_for_ultralytics

import logging
from pathlib import Path
from typing import Dict, List, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PotholeDetector:
    """Pothole detector for inference"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize PotholeDetector
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict on a single image
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with prediction results
        """
        # Read image
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
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
        
        return {
            "image_path": str(image_path),
            "num_detections": len(detections),
            "detections": detections
        }
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "error": str(e)
                })
        
        return results
    
    def visualize_predictions(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """
        Visualize predictions on image and save
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
        """
        # Read image
        img = cv2.imread(str(image_path))
        
        # Run inference
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Draw results
        annotated_img = results[0].plot()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_img)
        
        logger.info(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    # Example usage
    detector = PotholeDetector("runs/train/yolov8n_experiment/weights/best.pt")
    
    # Predict single image
    result = detector.predict_image("path/to/image.jpg")
    print(result)
