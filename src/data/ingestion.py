"""
Data Ingestion Module
Validates dataset structure, checks for missing labels and corrupt images
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data validation and ingestion for YOLO format dataset"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize DataIngestion
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.test_path = self.dataset_path / "test"
        
    def validate_dataset_structure(self) -> bool:
        """
        Validate if dataset has correct structure
        
        Returns:
            bool: True if structure is valid
        """
        logger.info("Validating dataset structure...")
        
        required_dirs = [
            self.train_path / "images",
            self.train_path / "labels",
            self.val_path / "images",
            self.val_path / "labels",
            self.test_path / "images",
            self.test_path / "labels"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Missing required directory: {dir_path}")
                return False
            logger.info(f"✓ Found: {dir_path}")
        
        logger.info("✓ Dataset structure is valid")
        return True
    
    def check_corrupt_images(self, split: str = "train") -> List[str]:
        """
        Check for corrupt or unreadable images
        
        Args:
            split: Dataset split (train/val/test)
            
        Returns:
            List of corrupt image paths
        """
        logger.info(f"Checking corrupt images in {split} split...")
        
        split_path = self.dataset_path / split / "images"
        corrupt_images = []
        
        image_files = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Checking {split} images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_images.append(str(img_path))
                    logger.warning(f"Corrupt image: {img_path}")
            except Exception as e:
                corrupt_images.append(str(img_path))
                logger.warning(f"Error reading {img_path}: {e}")
        
        if corrupt_images:
            logger.warning(f"Found {len(corrupt_images)} corrupt images in {split}")
        else:
            logger.info(f"✓ No corrupt images found in {split}")
        
        return corrupt_images
    
    def check_missing_labels(self, split: str = "train") -> List[str]:
        """
        Check for images without corresponding labels
        
        Args:
            split: Dataset split (train/val/test)
            
        Returns:
            List of images without labels
        """
        logger.info(f"Checking missing labels in {split} split...")
        
        images_path = self.dataset_path / split / "images"
        labels_path = self.dataset_path / split / "labels"
        
        missing_labels = []
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Checking {split} labels"):
            label_path = labels_path / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                missing_labels.append(str(img_path))
                logger.warning(f"Missing label for: {img_path}")
        
        if missing_labels:
            logger.warning(f"Found {len(missing_labels)} images without labels in {split}")
        else:
            logger.info(f"✓ All images have labels in {split}")
        
        return missing_labels
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Calculating dataset statistics...")
        
        stats = {
            "train": self._count_split("train"),
            "val": self._count_split("val"),
            "test": self._count_split("test")
        }
        
        # Calculate totals from splits only (not from stats dict which will contain total_images)
        splits = ["train", "val", "test"]
        stats["total_images"] = sum([stats[s]["images"] for s in splits])
        stats["total_labels"] = sum([stats[s]["labels"] for s in splits])
        
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Total labels: {stats['total_labels']}")
        
        return stats
    
    def _count_split(self, split: str) -> Dict:
        """Count images and labels in a split"""
        images_path = self.dataset_path / split / "images"
        labels_path = self.dataset_path / split / "labels"
        
        num_images = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.png")))
        num_labels = len(list(labels_path.glob("*.txt")))
        
        return {
            "images": num_images,
            "labels": num_labels
        }
    
    def validate_label_format(self, split: str = "train", sample_size: int = 10) -> Tuple[bool, List[str]]:
        """
        Validate YOLO label format
        
        Args:
            split: Dataset split
            sample_size: Number of labels to check
            
        Returns:
            Tuple of (is_valid, invalid_files)
        """
        logger.info(f"Validating label format in {split} split (sample: {sample_size})...")
        
        labels_path = self.dataset_path / split / "labels"
        label_files = list(labels_path.glob("*.txt"))[:sample_size]
        
        invalid_files = []
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        
                        if len(parts) != 5:
                            invalid_files.append(str(label_file))
                            logger.warning(f"Invalid format in {label_file}: {line}")
                            break
                        
                        # Check if values are valid
                        class_id, x, y, w, h = map(float, parts)
                        
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            invalid_files.append(str(label_file))
                            logger.warning(f"Invalid coordinates in {label_file}: {line}")
                            break
                            
            except Exception as e:
                invalid_files.append(str(label_file))
                logger.warning(f"Error reading {label_file}: {e}")
        
        is_valid = len(invalid_files) == 0
        
        if is_valid:
            logger.info(f"✓ Label format is valid in {split}")
        else:
            logger.warning(f"Found {len(invalid_files)} files with invalid label format")
        
        return is_valid, invalid_files
    
    def create_data_yaml(self, output_path: str = "configs/data.yaml"):
        """
        Create data.yaml file for YOLO training
        
        Args:
            output_path: Path to save data.yaml
        """
        logger.info("Creating data.yaml...")
        
        data_yaml = {
            "path": str(self.dataset_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["pothole"]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"✓ Created data.yaml at {output_path}")
    
    def run_full_validation(self) -> bool:
        """
        Run complete dataset validation
        
        Returns:
            bool: True if all validations pass
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL DATASET VALIDATION")
        logger.info("=" * 60)
        
        # Check structure
        if not self.validate_dataset_structure():
            return False
        
        # Get statistics
        stats = self.get_dataset_statistics()
        
        # Check each split
        all_valid = True
        
        for split in ["train", "val", "test"]:
            logger.info(f"\n--- Validating {split} split ---")
            
            # Check corrupt images
            corrupt = self.check_corrupt_images(split)
            if corrupt:
                all_valid = False
            
            # Check missing labels
            missing = self.check_missing_labels(split)
            if missing:
                all_valid = False
            
            # Validate label format
            is_valid, invalid = self.validate_label_format(split)
            if not is_valid:
                all_valid = False
        
        # Create data.yaml
        self.create_data_yaml()
        
        logger.info("\n" + "=" * 60)
        if all_valid:
            logger.info("✓ ALL VALIDATIONS PASSED")
        else:
            logger.warning("⚠ SOME VALIDATIONS FAILED - Please fix issues before training")
        logger.info("=" * 60)
        
        return all_valid


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion("dataset")
    ingestion.run_full_validation()
