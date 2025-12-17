"""
Unit tests for data ingestion module
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestion


def test_data_ingestion_init():
    """Test DataIngestion initialization"""
    ingestion = DataIngestion("dataset")
    
    assert ingestion.dataset_path == Path("dataset")
    assert ingestion.train_path == Path("dataset/train")
    assert ingestion.val_path == Path("dataset/val")
    assert ingestion.test_path == Path("dataset/test")


def test_validate_dataset_structure_missing_dirs(tmp_path):
    """Test dataset structure validation with missing directories"""
    ingestion = DataIngestion(tmp_path)
    
    result = ingestion.validate_dataset_structure()
    
    assert result == False


def test_count_split(tmp_path):
    """Test counting files in a split"""
    # Create dummy structure
    train_images = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    
    # Create dummy files
    (train_images / "img1.jpg").touch()
    (train_images / "img2.jpg").touch()
    (train_labels / "img1.txt").touch()
    (train_labels / "img2.txt").touch()
    
    ingestion = DataIngestion(tmp_path)
    result = ingestion._count_split("train")
    
    assert result["images"] == 2
    assert result["labels"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
