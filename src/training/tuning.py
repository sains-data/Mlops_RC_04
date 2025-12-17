"""
Hyperparameter Tuning Module using Optuna
"""
# Import torch configuration FIRST to patch torch.load for PyTorch 2.6+ compatibility
from src.utils.torch_config import configure_torch_for_ultralytics
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch
import optuna
from optuna.trial import Trial
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


class HyperparameterTuner:
    """Handles hyperparameter tuning using Optuna"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize HyperparameterTuner
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tuning_config = self.config['tuning']
        self.data_yaml = "configs/data.yaml"
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(f"{self.config['mlflow']['experiment_name']}_tuning")
    
    def objective(self, trial: Trial, model_type: str = "yolov8n") -> float:
        """
        Objective function for Optuna
        
        Args:
            trial: Optuna trial object
            model_type: Type of YOLO model
            
        Returns:
            mAP@0.5 score (metric to optimize)
        """
        # Suggest hyperparameters
        lr0 = trial.suggest_float("lr0", 
                                   self.tuning_config['parameters']['lr0'][0],
                                   self.tuning_config['parameters']['lr0'][1],
                                   log=True)
        
        batch_size = trial.suggest_categorical("batch_size", 
                                               self.tuning_config['parameters']['batch_size'])
        
        epochs = trial.suggest_categorical("epochs",
                                          self.tuning_config['parameters']['epochs'])
        
        optimizer = trial.suggest_categorical("optimizer",
                                             self.tuning_config['parameters']['optimizer'])
        
        logger.info(f"\nTrial {trial.number}:")
        logger.info(f"  lr0: {lr0}")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  epochs: {epochs}")
        logger.info(f"  optimizer: {optimizer}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            # Log parameters
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("lr0", lr0)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("model_type", model_type)
            
            try:
                # Initialize model
                model = YOLO(f"{model_type}.pt")
                
                # Train model
                results = model.train(
                    data=self.data_yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=640,
                    lr0=lr0,
                    optimizer=optimizer,
                    name=f"tuning_trial_{trial.number}",
                    project='runs/tuning',
                    exist_ok=True,
                    verbose=False,
                    patience=10,
                    save=False
                )
                
                # Validate model
                val_results = model.val()
                
                # Get mAP@0.5
                map50 = float(val_results.box.map50)
                
                # Log metrics
                mlflow.log_metric("map50", map50)
                mlflow.log_metric("map50_95", float(val_results.box.map))
                mlflow.log_metric("precision", float(val_results.box.mp))
                mlflow.log_metric("recall", float(val_results.box.mr))
                
                logger.info(f"  mAP@0.5: {map50:.4f}")
                
                return map50
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return 0.0
    
    def tune(
        self,
        model_type: str = "yolov8n",
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Run hyperparameter tuning
        
        Args:
            model_type: Type of YOLO model
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info("=" * 60)
        logger.info(f"STARTING HYPERPARAMETER TUNING FOR {model_type}")
        logger.info("=" * 60)
        
        n_trials = n_trials or self.tuning_config['n_trials']
        timeout = timeout or self.tuning_config['timeout']
        
        # Create study
        study = optuna.create_study(
            study_name=f"{model_type}_tuning",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Start parent MLflow run
        with mlflow.start_run(run_name=f"{model_type}_tuning_session"):
            # Optimize
            study.optimize(
                lambda trial: self.objective(trial, model_type),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info("\n" + "=" * 60)
            logger.info("TUNING COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Best mAP@0.5: {best_value:.4f}")
            logger.info("Best parameters:")
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")
            logger.info("=" * 60)
            
            # Log best parameters to MLflow
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            mlflow.log_metric("best_map50", best_value)
            
            # Save optimization history
            self._save_optimization_history(study)
            
            return {
                "best_params": best_params,
                "best_value": best_value,
                "study": study
            }
    
    def _save_optimization_history(self, study: optuna.Study):
        """Save optimization history plot"""
        import matplotlib.pyplot as plt
        
        logger.info("Saving optimization history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot optimization history
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        
        axes[0].plot(values, marker='o')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('mAP@0.5')
        axes[0].set_title('Optimization History')
        axes[0].grid(alpha=0.3)
        
        # Plot parameter importance
        if len(trials) > 1:
            try:
                importances = optuna.importance.get_param_importances(study)
                params = list(importances.keys())
                importance_values = list(importances.values())
                
                axes[1].barh(params, importance_values)
                axes[1].set_xlabel('Importance')
                axes[1].set_title('Parameter Importance')
                axes[1].grid(axis='x', alpha=0.3)
            except:
                axes[1].text(0.5, 0.5, 'Not enough trials for importance analysis',
                           ha='center', va='center')
                axes[1].set_title('Parameter Importance')
        
        plt.tight_layout()
        output_path = Path("outputs/tuning/optimization_history.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Optimization history saved to {output_path}")
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(str(output_path))


if __name__ == "__main__":
    tuner = HyperparameterTuner()
    results = tuner.tune(model_type="yolov8n", n_trials=10)
    print(results)
