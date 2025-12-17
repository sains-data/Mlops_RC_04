"""
Monitoring Module for ML System
Tracks inference latency, error rates, and drift detection
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import deque
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitors ML system performance and health"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize SystemMonitor
        
        Args:
            max_history: Maximum number of records to keep in memory
        """
        self.max_history = max_history
        
        # Metrics storage
        self.latency_history = deque(maxlen=max_history)
        self.error_history = deque(maxlen=max_history)
        self.prediction_history = deque(maxlen=max_history)
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        
        # Output directory
        self.output_dir = Path("outputs/monitoring")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_inference(
        self,
        model_name: str,
        latency_ms: float,
        num_detections: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log an inference request
        
        Args:
            model_name: Name of the model used
            latency_ms: Inference latency in milliseconds
            num_detections: Number of detections made
            success: Whether inference was successful
            error_message: Error message if failed
        """
        timestamp = datetime.now().isoformat()
        
        # Log latency
        self.latency_history.append({
            "timestamp": timestamp,
            "model": model_name,
            "latency_ms": latency_ms
        })
        
        # Log prediction
        self.prediction_history.append({
            "timestamp": timestamp,
            "model": model_name,
            "num_detections": num_detections
        })
        
        # Log errors
        if not success:
            self.error_history.append({
                "timestamp": timestamp,
                "model": model_name,
                "error": error_message
            })
            self.total_errors += 1
        
        self.total_requests += 1
        
        # Log to file periodically
        if self.total_requests % 100 == 0:
            self.save_metrics()
    
    def get_latency_stats(self, model_name: Optional[str] = None) -> Dict:
        """
        Get latency statistics
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = [
            record['latency_ms']
            for record in self.latency_history
            if model_name is None or record['model'] == model_name
        ]
        
        if not latencies:
            return {
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0,
                "min": 0,
                "max": 0
            }
        
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "count": len(latencies)
        }
    
    def get_error_rate(self, model_name: Optional[str] = None) -> float:
        """
        Calculate error rate
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            Error rate as percentage
        """
        if self.total_requests == 0:
            return 0.0
        
        if model_name is None:
            return (self.total_errors / self.total_requests) * 100
        
        model_errors = sum(
            1 for record in self.error_history
            if record['model'] == model_name
        )
        
        model_requests = sum(
            1 for record in self.latency_history
            if record['model'] == model_name
        )
        
        if model_requests == 0:
            return 0.0
        
        return (model_errors / model_requests) * 100
    
    def get_prediction_stats(self, model_name: Optional[str] = None) -> Dict:
        """
        Get prediction statistics
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            Dictionary with prediction statistics
        """
        detections = [
            record['num_detections']
            for record in self.prediction_history
            if model_name is None or record['model'] == model_name
        ]
        
        if not detections:
            return {
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "total": 0
            }
        
        return {
            "mean": np.mean(detections),
            "median": np.median(detections),
            "min": np.min(detections),
            "max": np.max(detections),
            "total": np.sum(detections)
        }
    
    def get_system_stats(self) -> Dict:
        """
        Get system resource statistics
        
        Returns:
            Dictionary with system stats
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Monitoring report dictionary
        """
        logger.info("Generating monitoring report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.get_error_rate(),
            "latency_stats": self.get_latency_stats(),
            "prediction_stats": self.get_prediction_stats(),
            "system_stats": self.get_system_stats()
        }
        
        return report
    
    def save_metrics(self):
        """Save metrics to file"""
        report = self.generate_report()
        
        output_path = self.output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def detect_drift(self, baseline_stats: Dict, current_stats: Dict, threshold: float = 0.2) -> bool:
        """
        Detect if there's drift in predictions
        
        Args:
            baseline_stats: Baseline prediction statistics
            current_stats: Current prediction statistics
            threshold: Drift detection threshold
            
        Returns:
            True if drift detected
        """
        baseline_mean = baseline_stats.get('mean', 0)
        current_mean = current_stats.get('mean', 0)
        
        if baseline_mean == 0:
            return False
        
        drift_ratio = abs(current_mean - baseline_mean) / baseline_mean
        
        if drift_ratio > threshold:
            logger.warning(f"Drift detected! Ratio: {drift_ratio:.3f}")
            return True
        
        return False


# Global monitor instance
monitor = SystemMonitor()


if __name__ == "__main__":
    # Example usage
    monitor = SystemMonitor()
    
    # Simulate some inferences
    for i in range(100):
        monitor.log_inference(
            model_name="yolov8n",
            latency_ms=np.random.uniform(30, 60),
            num_detections=np.random.randint(0, 5),
            success=np.random.random() > 0.05
        )
    
    # Generate report
    report = monitor.generate_report()
    print(json.dumps(report, indent=2))
