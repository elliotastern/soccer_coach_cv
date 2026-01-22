"""
Mosaic Augmentation for Tiny Object Detection
Carefully implemented to avoid cutting off small balls at image edges.
"""

import random
import numpy as np
from typing import Tuple, List, Dict
from PIL import Image
import torch


def apply_mosaic_careful(
    images: List[Image.Image],
    annotations: List[Dict],
    output_size: Tuple[int, int] = (1288, 1288),
    min_bbox_size: int = 5,
    border_margin: int = 10
) -> Tuple[Image.Image, Dict]:
    """
    Apply Mosaic augmentation while carefully preserving tiny objects.
    
    Args:
        images: List of 4 PIL Images to combine
        annotations: List of 4 annotation dicts (each with 'boxes' and 'labels')
        output_size: Output image size (width, height)
        min_bbox_size: Minimum bbox size in pixels to keep (filter out cut-off objects)
        border_margin: Margin from border to ensure objects aren't cut off
    
    Returns:
        Tuple of (mosaic_image, combined_annotations)
    """
    if len(images) != 4 or len(annotations) != 4:
        raise ValueError("Mosaic requires exactly 4 images and 4 annotation dicts")
    
    output_width, output_height = output_size
    
    # Create output image
    mosaic_image = Image.new('RGB', (output_width, output_height), (114, 114, 114))
    
    # Randomly choose split points (but ensure margin from edges)
    split_x = random.randint(border_margin, output_width - border_margin)
    split_y = random.randint(border_margin, output_height - border_margin)
    
    # Define quadrants: top-left, top-right, bottom-left, bottom-right
    quadrants = [
        (0, 0, split_x, split_y),  # top-left
        (split_x, 0, output_width, split_y),  # top-right
        (0, split_y, split_x, output_height),  # bottom-left
        (split_x, split_y, output_width, output_height)  # bottom-right
    ]
    
    combined_boxes = []
    combined_labels = []
    combined_areas = []
    combined_iscrowd = []
    
    for idx, (img, ann) in enumerate(zip(images, annotations)):
        x_min_q, y_min_q, x_max_q, y_max_q = quadrants[idx]
        quad_width = x_max_q - x_min_q
        quad_height = y_max_q - y_min_q
        
        # Resize image to fit quadrant
        img_resized = img.resize((quad_width, quad_height), Image.BILINEAR)
        
        # Paste into mosaic
        mosaic_image.paste(img_resized, (x_min_q, y_min_q))
        
        # Process annotations for this quadrant
        if 'boxes' in ann and len(ann['boxes']) > 0:
            boxes = ann['boxes']
            labels = ann.get('labels', torch.zeros(len(boxes), dtype=torch.int64))
            areas = ann.get('area', torch.zeros(len(boxes), dtype=torch.float32))
            iscrowd = ann.get('iscrowd', torch.zeros(len(boxes), dtype=torch.int64))
            
            # Get original image dimensions
            orig_width, orig_height = img.size
            scale_x = quad_width / orig_width
            scale_y = quad_height / orig_height
            
            for box, label, area, crowd in zip(boxes, labels, areas, iscrowd):
                # Convert box to absolute coordinates if needed
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                
                # COCO format: [x_min, y_min, width, height]
                x_min, y_min, width, height = box
                x_max = x_min + width
                y_max = y_min + height
                
                # Scale to quadrant size
                x_min_scaled = x_min * scale_x
                y_min_scaled = y_min * scale_y
                x_max_scaled = x_max * scale_x
                y_max_scaled = y_max * scale_y
                
                # Translate to quadrant position
                x_min_final = x_min_scaled + x_min_q
                y_min_final = y_min_scaled + y_min_q
                x_max_final = x_max_scaled + x_min_q
                y_max_final = y_max_scaled + y_min_q
                
                # Check if bbox is fully within image bounds (with margin)
                if (x_min_final >= border_margin and 
                    y_min_final >= border_margin and
                    x_max_final <= output_width - border_margin and
                    y_max_final <= output_height - border_margin):
                    
                    # Check minimum size
                    bbox_width = x_max_final - x_min_final
                    bbox_height = y_max_final - y_min_final
                    
                    if bbox_width >= min_bbox_size and bbox_height >= min_bbox_size:
                        # Convert back to COCO format [x_min, y_min, width, height]
                        final_box = [x_min_final, y_min_final, bbox_width, bbox_height]
                        combined_boxes.append(final_box)
                        combined_labels.append(int(label))
                        combined_areas.append(float(bbox_width * bbox_height))
                        combined_iscrowd.append(int(crowd))
    
    # Create combined annotation dict
    if len(combined_boxes) > 0:
        combined_ann = {
            'boxes': torch.tensor(combined_boxes, dtype=torch.float32),
            'labels': torch.tensor(combined_labels, dtype=torch.int64),
            'area': torch.tensor(combined_areas, dtype=torch.float32),
            'iscrowd': torch.tensor(combined_iscrowd, dtype=torch.int64)
        }
    else:
        # No valid annotations
        combined_ann = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.int64)
        }
    
    return mosaic_image, combined_ann


class MosaicDatasetWrapper:
    """
    Wrapper for COCO dataset that applies Mosaic augmentation.
    """
    
    def __init__(self, base_dataset, prob: float = 0.5, min_bbox_size: int = 5, 
                 border_margin: int = 10, output_size: Tuple[int, int] = (1288, 1288)):
        """
        Initialize Mosaic dataset wrapper.
        
        Args:
            base_dataset: Base COCO dataset
            prob: Probability of applying mosaic (0.0 to 1.0)
            min_bbox_size: Minimum bbox size in pixels to keep
            border_margin: Margin from border to ensure objects aren't cut off
            output_size: Output image size (width, height)
        """
        self.base_dataset = base_dataset
        self.prob = prob
        self.min_bbox_size = min_bbox_size
        self.border_margin = border_margin
        self.output_size = output_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        image, target = self.base_dataset[idx]
        
        # Apply mosaic with probability
        if random.random() < self.prob:
            # Get 3 additional random samples for mosaic
            indices = [idx]
            while len(indices) < 4:
                other_idx = random.randint(0, len(self.base_dataset) - 1)
                if other_idx not in indices:
                    indices.append(other_idx)
            
            # Get all 4 samples
            images = []
            annotations = []
            for i in indices:
                img, ann = self.base_dataset[i]
                images.append(img)
                annotations.append(ann)
            
            # Apply mosaic
            try:
                mosaic_img, mosaic_ann = apply_mosaic_careful(
                    images, annotations, self.output_size, 
                    self.min_bbox_size, self.border_margin
                )
                return mosaic_img, mosaic_ann
            except Exception as e:
                # If mosaic fails, return original sample
                print(f"Warning: Mosaic augmentation failed: {e}, using original sample")
                return image, target
        else:
            # No mosaic, return original
            return image, target
