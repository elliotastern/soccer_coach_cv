"""
COCO Dataset Loader for DETR Training
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoDataset(Dataset):
    """COCO format dataset for DETR training"""
    
    def __init__(self, dataset_dir: str, transforms=None):
        """
        Initialize COCO dataset
        
        Args:
            dataset_dir: Directory containing images/ and annotations/annotations.json
            transforms: Optional transform pipeline
        """
        self.dataset_dir = Path(dataset_dir)
        self.transforms = transforms
        
        # Load annotations
        annotations_path = self.dataset_dir / "annotations" / "annotations.json"
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Map category names to class IDs (player=0, ball=1)
        self.class_map = {}
        for cat_id, cat_name in self.categories.items():
            if cat_name.lower() == 'player':
                self.class_map[cat_id] = 0
            elif cat_name.lower() == 'ball':
                self.class_map[cat_id] = 1
            else:
                # Default mapping
                self.class_map[cat_id] = int(cat_id)
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Create list of image IDs
        self.image_ids = sorted(self.images.keys())
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item from dataset
        
        Returns:
            image: PIL Image or transformed tensor
            target: Dictionary with 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image with error handling for corrupted files
        image_path = self.dataset_dir / "images" / image_info['file_name']
        try:
            image = Image.open(image_path)
            # Try to load and convert
            try:
                image.load()
                image = image.convert('RGB')
            except (OSError, IOError):
                # If image is corrupted, create a black placeholder
                print(f"Warning: Corrupted image {image_path.name}, using placeholder")
                image = Image.new('RGB', (1920, 1080), (0, 0, 0))
        except Exception as e:
            # Skip corrupted images - return a black image with no annotations
            print(f"Warning: Skipping corrupted image {image_path}: {e}")
            image = Image.new('RGB', (1920, 1080), (0, 0, 0))
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([image_id], dtype=torch.int64),
                'area': areas,
                'iscrowd': iscrowd
            }
            
            if self.transforms:
                image, target = self.transforms(image, target)
            
            return image, target
        
        # Get annotations for this image
        annotations = self.image_annotations.get(image_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            
            # Map category to class ID
            cat_id = ann['category_id']
            labels.append(self.class_map.get(cat_id, 0))
            
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        if len(boxes) == 0:
            # No annotations - create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id], dtype=torch.int64),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target
