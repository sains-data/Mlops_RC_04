"""
Model Evaluation Module
"""
# Import torch configuration FIRST to patch torch.load for PyTorch 2.6+ compatibility
from src.utils.torch_config import configure_torch_for_ultralytics
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ModelEvaluator
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path("outputs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model_path: str,
        data_yaml: str = "configs/data.yaml",
        split: str = "val"
    ) -> Dict:
        """
        Evaluate model on validation or test set
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to data.yaml
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 60)
        logger.info(f"EVALUATING MODEL: {model_path}")
        logger.info("=" * 60)
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        logger.info(f"Running evaluation on {split} set...")
        results = model.val(data=data_yaml, split=split)
        
        # Extract metrics
        metrics = {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "f1_score": float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-8))
        }
        
        logger.info("\nEvaluation Metrics:")
        logger.info(f"  mAP@0.5:      {metrics['map50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        logger.info(f"  Precision:    {metrics['precision']:.4f}")
        logger.info(f"  Recall:       {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:     {metrics['f1_score']:.4f}")
        logger.info("=" * 60)
        
        return metrics
    
    def plot_confusion_matrix(self, model_path: str, data_yaml: str = "configs/data.yaml"):
        """
        Plot confusion matrix
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to data.yaml
        """
        logger.info("Generating confusion matrix...")
        
        model = YOLO(model_path)
        results = model.val(data=data_yaml)
        
        # Confusion matrix is saved automatically by YOLO
        logger.info("✓ Confusion matrix generated")
    
    def compare_models(self, model_paths: Dict[str, str]) -> Dict:
        """
        Compare multiple models
        
        Args:
            model_paths: Dictionary mapping model names to paths
            
        Returns:
            Comparison results
        """
        logger.info("=" * 60)
        logger.info("COMPARING MODELS")
        logger.info("=" * 60)
        
        comparison = {}
        
        for model_name, model_path in model_paths.items():
            logger.info(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model_path)
            comparison[model_name] = metrics
        
        # Create comparison plot
        self._plot_model_comparison(comparison)
        
        return comparison
    
    def _plot_model_comparison(self, comparison: Dict):
        """Plot model comparison"""
        logger.info("Plotting model comparison...")
        
        models = list(comparison.keys())
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for idx, metric in enumerate(metrics):
            values = [comparison[model][metric] for model in models]
            axes[idx].bar(models, values, color='skyblue')
            axes[idx].set_title(metric.upper().replace('_', ' '))
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Model comparison saved to {output_path}")
        plt.close()
    
    def test_model(self, model_path: str, data_yaml: str = "configs/data.yaml") -> Dict:
        """
        Test model on test set
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to data.yaml
            
        Returns:
            Test metrics
        """
        logger.info("=" * 60)
        logger.info("TESTING MODEL ON TEST SET")
        logger.info("=" * 60)
        
        metrics = self.evaluate_model(model_path, data_yaml, split="test")
        
        return metrics


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Example: Compare models
    models = {
        "YOLOv8n": "runs/train/yolov8n_experiment/weights/best.pt",
        "YOLOv8s": "runs/train/yolov8s_experiment/weights/best.pt"
    }
    
    comparison = evaluator.compare_models(models)
    print(comparison)
