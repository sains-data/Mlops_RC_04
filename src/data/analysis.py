"""
Data Analysis and Exploratory Data Analysis (EDA) Module
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class DataAnalyzer:
    """Performs EDA on YOLO format dataset"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize DataAnalyzer
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.test_path = self.dataset_path / "test"
        self.output_dir = Path("outputs/eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_class_distribution(self, split: str = "train") -> Dict:
        """
        Analyze class distribution in dataset
        
        Args:
            split: Dataset split (train/val/test)
            
        Returns:
            Dictionary with class counts
        """
        logger.info(f"Analyzing class distribution in {split} split...")
        
        labels_path = self.dataset_path / split / "labels"
        label_files = list(labels_path.glob("*.txt"))
        
        class_counts = Counter()
        total_annotations = 0
        
        for label_file in tqdm(label_files, desc=f"Reading {split} labels"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        total_annotations += 1
        
        logger.info(f"Total annotations in {split}: {total_annotations}")
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        return {
            "class_counts": dict(class_counts),
            "total_annotations": total_annotations
        }
    
    def plot_class_distribution(self):
        """Plot class distribution for all splits"""
        logger.info("Plotting class distribution...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, split in enumerate(["train", "val", "test"]):
            dist = self.analyze_class_distribution(split)
            
            classes = list(dist["class_counts"].keys())
            counts = list(dist["class_counts"].values())
            
            axes[idx].bar(classes, counts, color='skyblue')
            axes[idx].set_title(f'{split.capitalize()} Class Distribution')
            axes[idx].set_xlabel('Class ID')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved class distribution plot to {output_path}")
        plt.close()
    
    def analyze_image_dimensions(self, split: str = "train") -> Dict:
        """
        Analyze image dimensions
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary with dimension statistics
        """
        logger.info(f"Analyzing image dimensions in {split} split...")
        
        images_path = self.dataset_path / split / "images"
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        widths = []
        heights = []
        aspect_ratios = []
        
        for img_path in tqdm(image_files[:100], desc=f"Reading {split} images"):
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        
        stats = {
            "width": {
                "mean": np.mean(widths),
                "std": np.std(widths),
                "min": np.min(widths),
                "max": np.max(widths)
            },
            "height": {
                "mean": np.mean(heights),
                "std": np.std(heights),
                "min": np.min(heights),
                "max": np.max(heights)
            },
            "aspect_ratio": {
                "mean": np.mean(aspect_ratios),
                "std": np.std(aspect_ratios)
            }
        }
        
        logger.info(f"Average dimensions: {stats['width']['mean']:.0f}x{stats['height']['mean']:.0f}")
        
        return stats
    
    def plot_image_dimensions(self):
        """Plot image dimension statistics"""
        logger.info("Plotting image dimensions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for split in ["train", "val", "test"]:
            images_path = self.dataset_path / split / "images"
            image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
            
            widths = []
            heights = []
            
            for img_path in tqdm(image_files[:100], desc=f"Processing {split}"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
            
            # Width distribution
            axes[0, 0].hist(widths, bins=30, alpha=0.5, label=split)
            axes[0, 0].set_title('Image Width Distribution')
            axes[0, 0].set_xlabel('Width (pixels)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Height distribution
            axes[0, 1].hist(heights, bins=30, alpha=0.5, label=split)
            axes[0, 1].set_title('Image Height Distribution')
            axes[0, 1].set_xlabel('Height (pixels)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # Scatter plot
        stats = self.analyze_image_dimensions("train")
        axes[1, 0].text(0.1, 0.5, f"Average Width: {stats['width']['mean']:.0f}px\n"
                                   f"Average Height: {stats['height']['mean']:.0f}px\n"
                                   f"Aspect Ratio: {stats['aspect_ratio']['mean']:.2f}",
                        fontsize=12, verticalalignment='center')
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Dimension Statistics')
        
        # Remove last subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        output_path = self.output_dir / "image_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved dimension plot to {output_path}")
        plt.close()
    
    def analyze_bbox_sizes(self, split: str = "train") -> Dict:
        """
        Analyze bounding box sizes
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary with bbox statistics
        """
        logger.info(f"Analyzing bounding box sizes in {split} split...")
        
        labels_path = self.dataset_path / split / "labels"
        label_files = list(labels_path.glob("*.txt"))
        
        bbox_widths = []
        bbox_heights = []
        bbox_areas = []
        
        for label_file in tqdm(label_files, desc=f"Processing {split} labels"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        w, h = float(parts[3]), float(parts[4])
                        bbox_widths.append(w)
                        bbox_heights.append(h)
                        bbox_areas.append(w * h)
        
        stats = {
            "width": {
                "mean": np.mean(bbox_widths),
                "std": np.std(bbox_widths),
                "min": np.min(bbox_widths),
                "max": np.max(bbox_widths)
            },
            "height": {
                "mean": np.mean(bbox_heights),
                "std": np.std(bbox_heights),
                "min": np.min(bbox_heights),
                "max": np.max(bbox_heights)
            },
            "area": {
                "mean": np.mean(bbox_areas),
                "std": np.std(bbox_areas),
                "min": np.min(bbox_areas),
                "max": np.max(bbox_areas)
            }
        }
        
        logger.info(f"Average bbox size: {stats['width']['mean']:.3f}x{stats['height']['mean']:.3f}")
        
        return stats
    
    def visualize_sample_annotations(self, split: str = "train", num_samples: int = 6):
        """
        Visualize sample images with annotations
        
        Args:
            split: Dataset split
            num_samples: Number of samples to visualize
        """
        logger.info(f"Visualizing sample annotations from {split}...")
        
        images_path = self.dataset_path / split / "images"
        labels_path = self.dataset_path / split / "labels"
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, img_path in enumerate(sample_images):
            # Read image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read labels
            label_path = labels_path / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            class_id, x_center, y_center, width, height = map(float, parts)
                            
                            # Convert to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # Draw bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(img, f"Pothole", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'Sample {idx+1}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{split}_sample_annotations.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved sample annotations to {output_path}")
        plt.close()
    
    def generate_eda_report(self) -> Dict:
        """
        Generate complete EDA report
        
        Returns:
            Dictionary with all EDA results
        """
        logger.info("=" * 60)
        logger.info("GENERATING COMPLETE EDA REPORT")
        logger.info("=" * 60)
        
        report = {}
        
        # Class distribution
        for split in ["train", "val", "test"]:
            report[f"{split}_class_dist"] = self.analyze_class_distribution(split)
            report[f"{split}_image_dims"] = self.analyze_image_dimensions(split)
            report[f"{split}_bbox_stats"] = self.analyze_bbox_sizes(split)
        
        # Generate plots
        self.plot_class_distribution()
        self.plot_image_dimensions()
        
        # Visualize samples
        for split in ["train", "val", "test"]:
            self.visualize_sample_annotations(split)
        
        # Convert NumPy types to Python types for JSON serialization
        report = convert_numpy_types(report)
        
        # Save report
        report_path = self.output_dir / "eda_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"✓ EDA report saved to {report_path}")
        logger.info("=" * 60)
        
        return report


if __name__ == "__main__":
    analyzer = DataAnalyzer("dataset")
    analyzer.generate_eda_report()
