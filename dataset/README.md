# Dataset Directory

Place your dataset here with the following structure:

```
dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt in YOLO format)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
├── test/
│   ├── images/     # Test images
│   └── labels/     # Test labels
└── dataset.yaml    # Auto-generated dataset config
```

## YOLO Label Format

Each `.txt` file contains bounding box annotations in the format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: Class index (0 for pothole)
- `x_center`, `y_center`: Center coordinates (normalized 0-1)
- `width`, `height`: Box dimensions (normalized 0-1)

## Example

For an image `image001.jpg`:
```
0 0.5 0.5 0.2 0.3
```

This represents a pothole at the center of the image with 20% width and 30% height.
