"""
PyTorch Configuration for compatibility with PyTorch 2.6+
This module must be imported before any model loading operations.
"""

import torch
import os

# Set environment variable to disable weights_only mode
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

def configure_torch_for_ultralytics():
    """
    Configure PyTorch to work with Ultralytics YOLO models in PyTorch 2.6+
    This adds all necessary safe globals for model loading.
    """
    try:
        # Import all necessary modules
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn import modules as ultralytics_modules
        from torch.nn.modules.container import Sequential
        from collections import OrderedDict
        import torch.nn as nn
        
        # Build comprehensive safe globals list
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
            nn.ConvTranspose2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
            nn.Sigmoid, nn.Softmax, nn.LayerNorm, nn.GroupNorm,
        ])
        
        # Add all Ultralytics custom modules dynamically
        for name in dir(ultralytics_modules):
            obj = getattr(ultralytics_modules, name)
            if isinstance(obj, type):
                safe_classes.append(obj)
        
        # Apply safe globals
        torch.serialization.add_safe_globals(safe_classes)
        
        return True
    except Exception as e:
        # If configuration fails, we'll try to use weights_only=False fallback
        import warnings
        warnings.warn(f"Failed to configure safe globals: {e}. Using legacy mode.")
        return False

# Monkey-patch torch.load to use weights_only=False for Ultralytics models
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """Patched torch.load that defaults to weights_only=False for backward compatibility"""
    if weights_only is None:
        # Default to False for backward compatibility with Ultralytics
        weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                 weights_only=weights_only, **kwargs)

# Apply the patch
torch.load = _patched_torch_load

# Run configuration
configure_torch_for_ultralytics()
