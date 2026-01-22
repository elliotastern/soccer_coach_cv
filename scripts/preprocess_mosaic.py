#!/usr/bin/env python3
"""
Pre-process COCO dataset with Mosaic augmentation.
Creates augmented images and annotations that preserve tiny balls.
"""

import json
import random
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.training.mosaic_augmentation import apply_mosaic_careful


def load_coco_dataset(coco_path: Path) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load COCO format dataset."""
    ann_file = coco_path / "_annotations.coco.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Group annotations by image_id
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    return images, ann_by_image, categories


def coco_ann_to_dict(ann_list: List[Dict], image_info: Dict) -> Dict:
    """Convert COCO annotations to dict format."""
    boxes = []
    labels = []
    areas = []
    iscrowd = []
    
    for ann in ann_list:
        boxes.append(ann['bbox'])  # [x, y, width, height]
        labels.append(ann['category_id'])
        areas.append(ann['area'])
        iscrowd.append(ann['iscrowd'])
    
    return {
        'boxes': boxes,
        'labels': labels,
        'area': areas,
        'iscrowd': iscrowd
    }


def apply_mosaic_to_dataset(
    coco_path: Path,
    output_path: Path,
    prob: float = 0.5,
    num_augmented: int = None,
    min_bbox_size: int = 5,
    border_margin: int = 10,
    output_size: Tuple[int, int] = (1288, 1288)
):
    """
    Apply Mosaic augmentation to COCO dataset.
    
    Args:
        coco_path: Path to input COCO dataset directory
        output_path: Path to output augmented dataset directory
        prob: Probability of applying mosaic (not used here, we create fixed number)
        num_augmented: Number of augmented images to create (None = same as original)
        min_bbox_size: Minimum bbox size to keep
        border_margin: Margin from border
        output_size: Output image size
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load original dataset
    print(f"Loading dataset from {coco_path}...")
    images, ann_by_image, categories = load_coco_dataset(coco_path)
    
    print(f"Original dataset: {len(images)} images")
    
    # Determine number of augmented images
    if num_augmented is None:
        num_augmented = len(images)
    
    # Create augmented dataset
    augmented_images = []
    augmented_annotations = []
    image_id_counter = 1
    annotation_id_counter = 1
    
    # Copy original images first
    print("Copying original images...")
    for img_info in images:
        img_file = coco_path / img_info['file_name']
        if img_file.exists():
            # Copy original image
            output_img = output_path / img_info['file_name']
            shutil.copy2(img_file, output_img)
            
            # Copy annotations
            img_id = img_info['id']
            if img_id in ann_by_image:
                for ann in ann_by_image[img_id]:
                    new_ann = ann.copy()
                    new_ann['id'] = annotation_id_counter
                    new_ann['image_id'] = image_id_counter
                    augmented_annotations.append(new_ann)
                    annotation_id_counter += 1
            
            new_img_info = img_info.copy()
            new_img_info['id'] = image_id_counter
            augmented_images.append(new_img_info)
            image_id_counter += 1
    
    # Create mosaic augmented images
    print(f"Creating {num_augmented} mosaic-augmented images...")
    for i in range(num_augmented):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_augmented}")
        
        # Randomly select 4 images
        selected_indices = random.sample(range(len(images)), min(4, len(images)))
        selected_images = []
        selected_annotations = []
        
        for idx in selected_indices:
            img_info = images[idx]
            img_file = coco_path / img_info['file_name']
            
            if img_file.exists():
                img = Image.open(img_file).convert('RGB')
                selected_images.append(img)
                
                # Get annotations for this image
                img_id = img_info['id']
                ann_list = ann_by_image.get(img_id, [])
                ann_dict = coco_ann_to_dict(ann_list, img_info)
                selected_annotations.append(ann_dict)
        
        if len(selected_images) == 4:
            try:
                # Apply mosaic
                mosaic_img, mosaic_ann = apply_mosaic_careful(
                    selected_images,
                    selected_annotations,
                    output_size,
                    min_bbox_size,
                    border_margin
                )
                
                # Save mosaic image
                mosaic_filename = f"mosaic_{image_id_counter:06d}.png"
                mosaic_path = output_path / mosaic_filename
                mosaic_img.save(mosaic_path)
                
                # Create image info
                mosaic_img_info = {
                    'id': image_id_counter,
                    'width': output_size[0],
                    'height': output_size[1],
                    'file_name': mosaic_filename
                }
                augmented_images.append(mosaic_img_info)
                
                # Create annotations
                if len(mosaic_ann['boxes']) > 0:
                    boxes = mosaic_ann['boxes'].numpy() if hasattr(mosaic_ann['boxes'], 'numpy') else mosaic_ann['boxes']
                    labels = mosaic_ann['labels'].numpy() if hasattr(mosaic_ann['labels'], 'numpy') else mosaic_ann['labels']
                    areas = mosaic_ann['area'].numpy() if hasattr(mosaic_ann['area'], 'numpy') else mosaic_ann['area']
                    
                    for box, label, area in zip(boxes, labels, areas):
                        ann = {
                            'id': annotation_id_counter,
                            'image_id': image_id_counter,
                            'category_id': int(label),
                            'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                            'area': float(area),
                            'iscrowd': 0
                        }
                        augmented_annotations.append(ann)
                        annotation_id_counter += 1
                
                image_id_counter += 1
            except Exception as e:
                print(f"Warning: Failed to create mosaic {i}: {e}")
                continue
    
    # Save augmented COCO JSON
    augmented_coco = {
        'info': {
            'description': 'Mosaic-augmented ball detection dataset',
            'version': '1.0'
        },
        'licenses': [],
        'images': augmented_images,
        'annotations': augmented_annotations,
        'categories': categories
    }
    
    output_ann_file = output_path / "_annotations.coco.json"
    with open(output_ann_file, 'w') as f:
        json.dump(augmented_coco, f, indent=2)
    
    print(f"\n✅ Mosaic augmentation complete!")
    print(f"   Original images: {len(images)}")
    print(f"   Augmented images: {len(augmented_images)}")
    print(f"   Total annotations: {len(augmented_annotations)}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description="Apply Mosaic augmentation to COCO dataset")
    parser.add_argument("--input", type=str, required=True, help="Input COCO dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output augmented dataset directory")
    parser.add_argument("--prob", type=float, default=0.5, help="Probability of applying mosaic")
    parser.add_argument("--num-augmented", type=int, default=None, help="Number of augmented images to create")
    parser.add_argument("--min-bbox-size", type=int, default=5, help="Minimum bbox size to keep")
    parser.add_argument("--border-margin", type=int, default=10, help="Margin from border")
    parser.add_argument("--output-size", type=int, nargs=2, default=[1288, 1288], help="Output image size [width height]")
    
    args = parser.parse_args()
    
    apply_mosaic_to_dataset(
        Path(args.input),
        Path(args.output),
        args.prob,
        args.num_augmented,
        args.min_bbox_size,
        args.border_margin,
        tuple(args.output_size)
    )
