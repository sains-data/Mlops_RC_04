"""
Model Training Module with MLflow Integration
"""
# Import torch configuration FIRST to patch torch.load for PyTorch 2.6+ compatibility
from src.utils.torch_config import configure_torch_for_ultralytics
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from datetime import datetime
import torch
import mlflow
import mlflow.pytorch
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
except Exception as e:
    # Fallback for older PyTorch versions
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training with MLflow tracking"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ModelTrainer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup MLflow
        self.mlflow_uri = self.config['mlflow']['tracking_uri']
        self.experiment_name = self.config['mlflow']['experiment_name']
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow tracking URI: {self.mlflow_uri}")
        logger.info(f"MLflow experiment: {self.experiment_name}")
    
    def train_model(
        self,
        model_type: str = "yolov8n",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        data_yaml: str = "configs/data.yaml",
        run_name: Optional[str] = None
    ) -> Dict:
        """
        Train YOLO model with MLflow tracking
        
        Args:
            model_type: Type of YOLO model (yolov8n, yolov8s)
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size
            data_yaml: Path to data.yaml
            run_name: Custom run name for MLflow
            
        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info(f"STARTING TRAINING: {model_type}")
        logger.info("=" * 60)
        
        # Get model config
        model_config = None
        for model in self.config['training']['models']:
            if model['name'] == model_type:
                model_config = model
                break
        
        if model_config is None:
            raise ValueError(f"Model {model_type} not found in configuration")
        
        # Override with provided parameters
        epochs = epochs or model_config['epochs']
        batch_size = batch_size or model_config['batch_size']
        img_size = img_size or model_config['imgsz']
        
        # Start MLflow run
        if run_name is None:
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("img_size", img_size)
            mlflow.log_param("lr0", model_config['lr0'])
            mlflow.log_param("optimizer", model_config['optimizer'])
            
            # Log augmentation parameters
            aug_config = self.config['training']['augmentation']
            for key, value in aug_config.items():
                mlflow.log_param(f"aug_{key}", value)
            
            # Initialize model
            logger.info(f"Initializing {model_type} model...")
            model = YOLO(model_config['type'])
            
            # Train model
            logger.info("Starting training...")
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=model_config['lr0'],
                optimizer=model_config['optimizer'],
                name=run_name,
                project='runs/train',
                exist_ok=True,
                pretrained=True,
                verbose=True,
                save=True,
                save_period=10,
                **aug_config
            )
            
            # Get best model path
            best_model_path = Path('runs/train') / run_name / 'weights' / 'best.pt'
            
            # Log metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        # Sanitize metric name: remove parentheses and other invalid chars
                        clean_key = key.replace('(', '_').replace(')', '').replace(' ', '_')
                        mlflow.log_metric(clean_key, float(value))
            
            # Validate model
            logger.info("Validating model...")
            val_results = model.val()
            
            # Log validation metrics
            mlflow.log_metric("val_map50", val_results.box.map50)
            mlflow.log_metric("val_map50-95", val_results.box.map)
            mlflow.log_metric("val_precision", val_results.box.mp)
            mlflow.log_metric("val_recall", val_results.box.mr)
            
            # Log model weights file to MLflow (not the model object itself)
            logger.info("Logging model weights to MLflow...")
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="model")
            
            # Log artifacts
            results_dir = Path('runs/train') / run_name
            if results_dir.exists():
                # Log key artifacts but skip large files
                for artifact in ['results.png', 'confusion_matrix.png', 'labels.jpg', 'results.csv']:
                    artifact_path = results_dir / artifact
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path), artifact_path="training_results")
            
            # Copy model to models/ directory for easy access
            logger.info("Copying model to models/ directory...")
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            production_model_path = models_dir / f"{model_type}_best.pt"
            if best_model_path.exists():
                import shutil
                shutil.copy2(str(best_model_path), str(production_model_path))
                logger.info(f"✓ Model copied to: {production_model_path}")
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED")
            logger.info(f"Best model saved to: {best_model_path}")
            logger.info(f"Production model: {production_model_path}")
            logger.info(f"mAP@0.5: {val_results.box.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
            logger.info("=" * 60)
            
            return {
                "run_id": run.info.run_id,
                "model_path": str(best_model_path),
                "production_model_path": str(production_model_path),
                "map50": float(val_results.box.map50),
                "map50_95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr)
            }
    
    def train_multiple_models(self):
        """Train all configured models"""
        logger.info("Training multiple models...")
        
        results = {}
        
        for model_config in self.config['training']['models']:
            model_type = model_config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type}...")
            logger.info(f"{'='*60}\n")
            
            try:
                result = self.train_model(model_type=model_type)
                results[model_type] = result
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    # Train single model
    result = trainer.train_model("yolov8n")
    print(result)
