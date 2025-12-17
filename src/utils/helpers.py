"""
Utility functions for the project
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Any, Dict
import json


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, output_path: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> Dict:
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_model_path(model_name: str, base_dir: str = "models") -> Path:
    """
    Get path for model
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for models
        
    Returns:
        Path to model
    """
    model_path = Path(base_dir) / f"{model_name}.pt"
    return model_path


def ensure_dir(directory: str):
    """
    Ensure directory exists
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
