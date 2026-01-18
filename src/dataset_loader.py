"""
Dataset loader for training RF-DETR models.
Supports COCO format annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotation file."""
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    with open(annotation_path, 'r') as f:
        return json.load(f)


def get_image_paths(images_dir: str) -> List[str]:
    """Get all image file paths from directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(images_dir).glob(f'*{ext}'))
        image_paths.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    return sorted([str(p) for p in image_paths])


def validate_dataset(dataset_path: str) -> Tuple[bool, List[str]]:
    """Validate dataset structure and return issues."""
    issues = []
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        issues.append(f"Dataset directory does not exist: {dataset_path}")
        return False, issues
    
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations" / "annotations.json"
    
    if not images_dir.exists():
        issues.append(f"Images directory missing: {images_dir}")
    
    if not annotations_file.exists():
        issues.append(f"Annotations file missing: {annotations_file}")
    else:
        try:
            coco_data = load_coco_annotations(str(annotations_file))
            if 'images' not in coco_data:
                issues.append("COCO format: missing 'images' key")
            if 'annotations' not in coco_data:
                issues.append("COCO format: missing 'annotations' key")
            if 'categories' not in coco_data:
                issues.append("COCO format: missing 'categories' key")
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON in annotations: {e}")
    
    return len(issues) == 0, issues


def get_dataset_info(dataset_path: str) -> Dict:
    """Get summary information about the dataset."""
    dataset_path = Path(dataset_path)
    info = {
        'path': str(dataset_path),
        'images_count': 0,
        'annotations_count': 0,
        'categories_count': 0,
        'categories': []
    }
    
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations" / "annotations.json"
    
    if images_dir.exists():
        info['images_count'] = len(get_image_paths(str(images_dir)))
    
    if annotations_file.exists():
        try:
            coco_data = load_coco_annotations(str(annotations_file))
            info['annotations_count'] = len(coco_data.get('annotations', []))
            info['categories_count'] = len(coco_data.get('categories', []))
            info['categories'] = [cat['name'] for cat in coco_data.get('categories', [])]
        except Exception as e:
            info['error'] = str(e)
    
    return info
