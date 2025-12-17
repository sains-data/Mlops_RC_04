"""
Data Preprocessing Module
Handles data preprocessing and augmentation
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for YOLO training"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataPreprocessor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.img_size = self.config['data']['img_size']
        self.augmentation = self.config['training']['augmentation']
    
    def get_augmentation_config(self) -> Dict:
        """
        Get augmentation configuration for YOLO training
        
        Returns:
            Dictionary with augmentation parameters
        """
        logger.info("Getting augmentation configuration...")
        
        aug_config = {
            'hsv_h': self.augmentation['hsv_h'],
            'hsv_s': self.augmentation['hsv_s'],
            'hsv_v': self.augmentation['hsv_v'],
            'degrees': self.augmentation['degrees'],
            'translate': self.augmentation['translate'],
            'scale': self.augmentation['scale'],
            'shear': self.augmentation['shear'],
            'flipud': self.augmentation['flipud'],
            'fliplr': self.augmentation['fliplr'],
            'mosaic': self.augmentation['mosaic']
        }
        
        logger.info(f"Augmentation config: {aug_config}")
        return aug_config
    
    def prepare_training_config(self, model_type: str = "yolov8n") -> Dict:
        """
        Prepare complete training configuration
        
        Args:
            model_type: Type of YOLO model
            
        Returns:
            Training configuration dictionary
        """
        logger.info(f"Preparing training configuration for {model_type}...")
        
        # Find model config
        model_config = None
        for model in self.config['training']['models']:
            if model['name'] == model_type:
                model_config = model
                break
        
        if model_config is None:
            logger.error(f"Model {model_type} not found in config")
            raise ValueError(f"Model {model_type} not found in config")
        
        training_config = {
            'data': 'configs/data.yaml',
            'epochs': model_config['epochs'],
            'batch': model_config['batch_size'],
            'imgsz': model_config['imgsz'],
            'lr0': model_config['lr0'],
            'optimizer': model_config['optimizer'],
            'name': f'{model_type}_experiment',
            'project': 'runs/train',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'save': True,
            'save_period': 10,
            'device': self.config['inference']['device']
        }
        
        # Add augmentation parameters
        training_config.update(self.get_augmentation_config())
        
        logger.info("✓ Training configuration prepared")
        return training_config


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    config = preprocessor.prepare_training_config("yolov8n")
    print(config)
