#!/usr/bin/env python3
"""
Train RF-DETR model for ball-only detection using SoccerSynth_sub dataset.
Optimized for tiny ball objects (<15 pixels).
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# MLflow tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Error: RF-DETR not installed. Install with: pip install rfdetr")
    sys.exit(1)


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert YOLO bbox to COCO bbox format [x_min, y_min, width, height]."""
    class_id, x_center, y_center, width, height = yolo_bbox
    
    x_center_abs = float(x_center) * img_width
    y_center_abs = float(y_center) * img_height
    width_abs = float(width) * img_width
    height_abs = float(height) * img_height
    
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)
    
    return [x_min, y_min, width_abs, height_abs]


def load_yolo_annotation(txt_path: Path) -> List[List[float]]:
    """Load YOLO format annotation file."""
    annotations = []
    if not txt_path.exists():
        return annotations
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    annotations.append([float(p) for p in parts])
    
    return annotations


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions."""
    img = Image.open(image_path)
    return img.width, img.height


def convert_yolo_to_coco_ball_only(
    yolo_dir: Path,
    output_dir: Path,
    split_name: str,
    ball_class_id: int,
    category_name: str,
    category_id: int
) -> Tuple[int, int]:
    """
    Convert YOLO format to COCO format, filtering for ball-only annotations.
    
    Args:
        yolo_dir: Directory containing YOLO format images and annotations
        output_dir: Output directory for COCO format dataset
        split_name: Name of the split (train/valid/test)
        ball_class_id: Class ID for ball in YOLO format (typically 1)
        category_name: Category name for COCO format (e.g., "ball")
        category_id: Category ID for COCO format (0 for single class)
    
    Returns:
        Tuple of (num_images, num_annotations)
    """
    print(f"\n🔄 Converting {split_name} split (ball-only)...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RF-DETR expects images directly in the split directory and _annotations.coco.json
    images_dir = output_path
    images_dir.mkdir(exist_ok=True)
    
    coco_data = {
        "info": {
            "description": f"SoccerSynth {split_name} dataset - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            }
        ]
    }
    
    image_files = sorted(yolo_dir.glob("*.png"))
    if not image_files:
        # Try other image formats
        image_files = sorted(list(yolo_dir.glob("*.jpg")) + list(yolo_dir.glob("*.jpeg")))
    
    image_id = 1
    annotation_id = 1
    images_with_balls = 0
    total_ball_annotations = 0
    
    for img_file in image_files:
        txt_file = yolo_dir / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            # Skip images without annotations
            continue
        
        img_width, img_height = get_image_size(img_file)
        
        # Copy image to output directory
        dest_image = images_dir / img_file.name
        shutil.copy2(img_file, dest_image)
        
        image_info = {
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_file.name
        }
        coco_data["images"].append(image_info)
        
        # Load YOLO annotations and filter for ball only
        yolo_annotations = load_yolo_annotation(txt_file)
        has_ball = False
        
        for yolo_bbox in yolo_annotations:
            class_id = int(yolo_bbox[0])
            
            # Only keep ball annotations (class_id == ball_class_id)
            if class_id == ball_class_id:
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                
                # Skip if bbox is too small (likely annotation error)
                if coco_bbox[2] < 1.0 or coco_bbox[3] < 1.0:
                    continue
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # Always 0 for single class
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                total_ball_annotations += 1
                has_ball = True
        
        if has_ball:
            images_with_balls += 1
        else:
            # Remove image entry if it has no ball annotations
            coco_data["images"].pop()
            # Remove copied image if no ball annotations
            if dest_image.exists():
                dest_image.unlink()
            continue
        
        image_id += 1
    
    # Save COCO annotations file (RF-DETR expects _annotations.coco.json)
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {images_with_balls} images with ball annotations")
    print(f"✅ Created {total_ball_annotations} ball annotations")
    print(f"✅ Saved to {annotation_file}")
    
    return images_with_balls, total_ball_annotations


def check_coco_dataset_exists(coco_path: Path) -> bool:
    """Check if COCO format dataset already exists."""
    annotation_file = coco_path / "_annotations.coco.json"
    return annotation_file.exists() and len(list(coco_path.glob("*.png"))) > 0


def merge_coco_datasets(
    dataset1_path: Path,
    dataset2_path: Path,
    output_path: Path,
    category_name: str = "ball",
    category_id: int = 0,
    id_offset: int = 10000
) -> Tuple[int, int]:
    """
    Merge two COCO format datasets, filtering dataset2 for ball-only annotations.
    
    Args:
        dataset1_path: Path to first COCO dataset (SoccerSynth - already ball-only)
        dataset2_path: Path to second COCO dataset (Validation images OFFICIAL)
        output_path: Path to save merged dataset
        category_name: Category name to filter for in dataset2
        category_id: Category ID to use in merged dataset
        id_offset: Offset for image IDs from dataset2 to ensure uniqueness
    
    Returns:
        Tuple of (total_images, total_annotations)
    """
    print(f"\n🔄 Merging COCO datasets...")
    print(f"   Dataset 1: {dataset1_path}")
    print(f"   Dataset 2: {dataset2_path}")
    print(f"   Output: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset 1 (SoccerSynth - already ball-only)
    dataset1_ann_file = dataset1_path / "_annotations.coco.json"
    if not dataset1_ann_file.exists():
        raise FileNotFoundError(f"Dataset 1 annotation file not found: {dataset1_ann_file}")
    
    with open(dataset1_ann_file, 'r') as f:
        dataset1_data = json.load(f)
    
    dataset1_images = dataset1_data.get('images', [])
    dataset1_annotations = dataset1_data.get('annotations', [])
    dataset1_categories = dataset1_data.get('categories', [])
    
    print(f"   Dataset 1: {len(dataset1_images)} images, {len(dataset1_annotations)} annotations")
    
    # Load dataset 2 (Validation images OFFICIAL)
    dataset2_ann_file = dataset2_path / "_annotations.coco.json"
    if not dataset2_ann_file.exists():
        raise FileNotFoundError(f"Dataset 2 annotation file not found: {dataset2_ann_file}")
    
    with open(dataset2_ann_file, 'r') as f:
        dataset2_data = json.load(f)
    
    dataset2_images = dataset2_data.get('images', [])
    dataset2_annotations = dataset2_data.get('annotations', [])
    dataset2_categories = {cat['id']: cat['name'] for cat in dataset2_data.get('categories', [])}
    
    # Find ball category ID in dataset2
    ball_category_id = None
    for cat_id, cat_name in dataset2_categories.items():
        if cat_name.lower() == category_name.lower():
            ball_category_id = int(cat_id)
            break
    
    if ball_category_id is None:
        raise ValueError(f"Ball category not found in dataset2. Available categories: {list(dataset2_categories.values())}")
    
    print(f"   Dataset 2: {len(dataset2_images)} images, {len(dataset2_annotations)} total annotations")
    print(f"   Ball category ID in dataset2: {ball_category_id}")
    
    # Filter dataset2 for ball-only annotations
    dataset2_ball_annotations = [ann for ann in dataset2_annotations 
                                if ann.get('category_id') == ball_category_id]
    
    # Get image IDs that have ball annotations
    dataset2_ball_image_ids = set(ann['image_id'] for ann in dataset2_ball_annotations)
    
    # Filter dataset2 images to only those with ball annotations
    dataset2_ball_images = [img for img in dataset2_images 
                           if img['id'] in dataset2_ball_image_ids]
    
    print(f"   Dataset 2 (filtered): {len(dataset2_ball_images)} images with balls, {len(dataset2_ball_annotations)} ball annotations")
    
    # Create merged dataset structure
    merged_data = {
        "info": {
            "description": "Combined SoccerSynth + Validation images OFFICIAL - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            }
        ]
    }
    
    # Add dataset1 images and annotations (keep original IDs)
    image_id_mapping = {}  # Map old image_id -> new image_id
    
    for img in dataset1_images:
        new_image_id = len(merged_data['images']) + 1
        image_id_mapping[img['id']] = new_image_id
        
        new_img = img.copy()
        new_img['id'] = new_image_id
        
        # Copy image file (handle filename conflicts)
        src_image = dataset1_path / img['file_name']
        if src_image.exists():
            dst_image = output_path / img['file_name']
            # If filename conflict, prefix with synth_
            if dst_image.exists() and dst_image.stat().st_size != src_image.stat().st_size:
                new_filename = f"synth_{img['file_name']}"
                dst_image = output_path / new_filename
                new_img['file_name'] = new_filename
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
        
        merged_data['images'].append(new_img)
    
    # Add dataset2 images with offset IDs
    for img in dataset2_ball_images:
        old_image_id = img['id']
        new_image_id = len(merged_data['images']) + 1
        image_id_mapping[id_offset + old_image_id] = new_image_id  # Use offset for mapping
        
        new_img = img.copy()
        new_img['id'] = new_image_id
        
        # Copy image file (handle filename conflicts)
        src_image = dataset2_path / img['file_name']
        if src_image.exists():
            dst_image = output_path / img['file_name']
            # If filename conflict, prefix with validation_
            if dst_image.exists() and dst_image.stat().st_size != src_image.stat().st_size:
                new_filename = f"validation_{img['file_name']}"
                dst_image = output_path / new_filename
                new_img['file_name'] = new_filename
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
        
        merged_data['images'].append(new_img)
    
    # Add dataset1 annotations (remap image_ids)
    annotation_id = 1
    for ann in dataset1_annotations:
        old_image_id = ann['image_id']
        if old_image_id in image_id_mapping:
            new_ann = ann.copy()
            new_ann['id'] = annotation_id
            new_ann['image_id'] = image_id_mapping[old_image_id]
            new_ann['category_id'] = category_id  # Ensure correct category ID
            merged_data['annotations'].append(new_ann)
            annotation_id += 1
    
    # Add dataset2 ball annotations (remap image_ids and category_id)
    for ann in dataset2_ball_annotations:
        old_image_id = ann['image_id']
        mapped_key = id_offset + old_image_id
        if mapped_key in image_id_mapping:
            new_ann = ann.copy()
            new_ann['id'] = annotation_id
            new_ann['image_id'] = image_id_mapping[mapped_key]
            new_ann['category_id'] = category_id  # Map to ball category
            merged_data['annotations'].append(new_ann)
            annotation_id += 1
    
    # Save merged dataset
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    total_images = len(merged_data['images'])
    total_annotations = len(merged_data['annotations'])
    
    print(f"✅ Merged dataset created:")
    print(f"   Total images: {total_images}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Saved to: {annotation_file}")
    
    return total_images, total_annotations


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: Dict, force_convert: bool = False) -> Tuple[str, str]:
    """
    Prepare COCO format dataset from YOLO format, optionally merging with Validation images OFFICIAL.
    
    Args:
        config: Configuration dictionary
        force_convert: Force conversion even if COCO dataset exists
    
    Returns:
        Tuple of (train_dir, val_dir) paths
    """
    dataset_config = config['dataset']
    ball_class_id = dataset_config['ball_class_id']
    category_name = dataset_config['category_name']
    category_id = dataset_config['category_id']
    
    use_combined = dataset_config.get('use_combined_dataset', False)
    
    yolo_train_path = Path(dataset_config['yolo_train_path'])
    yolo_val_path = Path(dataset_config.get('yolo_val_path', ''))
    coco_train_path = Path(dataset_config['coco_train_path'])
    coco_val_path = Path(dataset_config.get('coco_val_path', ''))
    
    # Convert SoccerSynth from YOLO to COCO first
    if force_convert or not check_coco_dataset_exists(coco_train_path):
        print(f"\n📦 Converting SoccerSynth train dataset from YOLO to COCO format...")
        print(f"   Source: {yolo_train_path}")
        print(f"   Destination: {coco_train_path}")
        
        if not yolo_train_path.exists():
            raise FileNotFoundError(f"YOLO train directory not found: {yolo_train_path}")
        
        train_images, train_annos = convert_yolo_to_coco_ball_only(
            yolo_train_path,
            coco_train_path,
            "train",
            ball_class_id,
            category_name,
            category_id
        )
        
        if train_images == 0:
            raise ValueError(f"No ball annotations found in train dataset: {yolo_train_path}")
    else:
        print(f"\n✅ SoccerSynth train COCO dataset already exists: {coco_train_path}")
    
    # Convert validation split if provided
    if yolo_val_path and Path(yolo_val_path).exists():
        if force_convert or not check_coco_dataset_exists(coco_val_path):
            print(f"\n📦 Converting SoccerSynth validation dataset from YOLO to COCO format...")
            print(f"   Source: {yolo_val_path}")
            print(f"   Destination: {coco_val_path}")
            
            val_images, val_annos = convert_yolo_to_coco_ball_only(
                Path(yolo_val_path),
                coco_val_path,
                "valid",
                ball_class_id,
                category_name,
                category_id
            )
            
            if val_images == 0:
                print(f"⚠️  Warning: No ball annotations found in validation dataset")
        else:
            print(f"✅ SoccerSynth validation COCO dataset already exists: {coco_val_path}")
    
    # Merge with Validation images OFFICIAL if enabled
    if use_combined:
        validation_official_path = Path(dataset_config.get('validation_official_path', ''))
        combined_train_path = Path(dataset_config.get('combined_train_path', ''))
        combined_val_path = Path(dataset_config.get('combined_val_path', ''))
        
        if not validation_official_path or not validation_official_path.exists():
            raise FileNotFoundError(f"Validation images OFFICIAL path not found: {validation_official_path}")
        
        validation_train_path = validation_official_path / "train"
        validation_val_path = validation_official_path / "valid"
        
        if not validation_train_path.exists():
            raise FileNotFoundError(f"Validation images OFFICIAL train not found: {validation_train_path}")
        
        # Merge training datasets
        if force_convert or not check_coco_dataset_exists(combined_train_path):
            print(f"\n🔀 Merging SoccerSynth + Validation images OFFICIAL for training...")
            merge_coco_datasets(
                coco_train_path,
                validation_train_path,
                combined_train_path,
                category_name=category_name,
                category_id=category_id,
                id_offset=10000
            )
        else:
            print(f"\n✅ Combined training dataset already exists: {combined_train_path}")
        
        # Prepare validation dataset (use Validation images OFFICIAL valid split, filter for balls)
        if validation_val_path.exists() and (force_convert or not check_coco_dataset_exists(combined_val_path)):
            print(f"\n📦 Preparing Validation images OFFICIAL validation split (ball-only)...")
            val_ann_file = validation_val_path / "_annotations.coco.json"
            if val_ann_file.exists():
                with open(val_ann_file, 'r') as f:
                    val_data = json.load(f)
                
                val_categories = {cat['id']: cat['name'] for cat in val_data.get('categories', [])}
                ball_cat_id = None
                for cat_id, cat_name in val_categories.items():
                    if cat_name.lower() == 'ball':
                        ball_cat_id = int(cat_id)
                        break
                
                if ball_cat_id:
                    val_images = val_data.get('images', [])
                    val_annos = val_data.get('annotations', [])
                    ball_image_ids = set(ann['image_id'] for ann in val_annos if ann.get('category_id') == ball_cat_id)
                    ball_images = [img for img in val_images if img['id'] in ball_image_ids]
                    ball_annotations = [ann for ann in val_annos if ann.get('category_id') == ball_cat_id]
                    
                    combined_val_path.mkdir(parents=True, exist_ok=True)
                    
                    # Copy images
                    for img in ball_images:
                        src = validation_val_path / img['file_name']
                        dst = combined_val_path / img['file_name']
                        if src.exists() and not dst.exists():
                            shutil.copy2(src, dst)
                    
                    # Remap image IDs to be sequential
                    image_id_map = {old_id: new_id for new_id, old_id in enumerate([img['id'] for img in ball_images], 1)}
                    
                    # Create filtered COCO data
                    filtered_data = {
                        "info": {"description": "Validation images OFFICIAL valid - Ball only", "version": "1.0"},
                        "licenses": [],
                        "images": [{"id": image_id_map[img['id']], "width": img['width'], "height": img['height'],
                                   "file_name": img['file_name']} for img in ball_images],
                        "annotations": [{"id": i+1, "image_id": image_id_map[ann['image_id']], 
                                       "category_id": category_id, "bbox": ann['bbox'], 
                                       "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3] if len(ann['bbox']) >= 4 else 0), 
                                       "iscrowd": ann.get('iscrowd', 0)}
                                      for i, ann in enumerate(ball_annotations)],
                        "categories": [{"id": category_id, "name": category_name, "supercategory": "object"}]
                    }
                    
                    with open(combined_val_path / "_annotations.coco.json", 'w') as f:
                        json.dump(filtered_data, f, indent=2)
                    
                    print(f"✅ Validation dataset prepared: {len(ball_images)} images, {len(ball_annotations)} annotations")
            else:
                print(f"⚠️  Validation images OFFICIAL valid annotation file not found")
        else:
            if validation_val_path.exists():
                print(f"✅ Combined validation dataset already exists: {combined_val_path}")
            else:
                print(f"⚠️  Validation images OFFICIAL valid path not found: {validation_val_path}")
        
        return str(combined_train_path), str(combined_val_path)
    else:
        # Use SoccerSynth only
        if not yolo_val_path or not Path(yolo_val_path).exists():
            print(f"⚠️  No validation dataset provided, using train for validation")
            coco_val_path = coco_train_path
        
        return str(coco_train_path), str(coco_val_path)


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR model for ball-only detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtdetr_r50vd_ball.yml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force dataset conversion even if COCO format exists"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Load config
    print("📋 Loading configuration...")
    config = load_config(args.config)
    print(f"✅ Loaded config from {args.config}")
    
    # Prepare dataset (convert YOLO to COCO if needed)
    print("\n" + "="*60)
    print("DATASET PREPARATION")
    print("="*60)
    train_dir, val_dir = prepare_dataset(config, force_convert=args.force_convert)
    
    # Get training config early (needed for mosaic augmentation)
    training_config = config['training']
    
    # Apply Mosaic augmentation if enabled
    augmentation_config = config.get('augmentation', {})
    mosaic_config = augmentation_config.get('mosaic', {})
    if mosaic_config.get('enabled', False):
        print("\n" + "="*60)
        print("MOSAIC AUGMENTATION")
        print("="*60)
        
        # Import mosaic preprocessing
        sys.path.insert(0, str(Path(__file__).parent))
        from preprocess_mosaic import apply_mosaic_to_dataset
        
        mosaic_train_dir = Path(train_dir).parent / f"{Path(train_dir).name}_mosaic"
        mosaic_val_dir = Path(val_dir).parent / f"{Path(val_dir).name}_mosaic"
        
        resolution = training_config.get('resolution', 1288)
        output_size = (resolution, resolution)
        
        print(f"Applying Mosaic augmentation to training set...")
        print(f"  Input: {train_dir}")
        print(f"  Output: {mosaic_train_dir}")
        print(f"  Output size: {output_size}")
        print(f"  Min bbox size: {mosaic_config.get('min_bbox_size', 5)}")
        print(f"  Border margin: {mosaic_config.get('border_margin', 10)}")
        
        if not (mosaic_train_dir / "_annotations.coco.json").exists() or args.force_convert:
            apply_mosaic_to_dataset(
                Path(train_dir),
                mosaic_train_dir,
                prob=mosaic_config.get('prob', 0.5),
                num_augmented=None,  # Create same number as original
                min_bbox_size=mosaic_config.get('min_bbox_size', 5),
                border_margin=mosaic_config.get('border_margin', 10),
                output_size=output_size
            )
            train_dir = str(mosaic_train_dir)
        else:
            print(f"✅ Mosaic-augmented training set already exists: {mosaic_train_dir}")
            train_dir = str(mosaic_train_dir)
        
        # Apply to validation set (optional, usually not needed)
        if Path(val_dir).exists() and val_dir != train_dir:
            if not (mosaic_val_dir / "_annotations.coco.json").exists() or args.force_convert:
                print(f"\nApplying Mosaic augmentation to validation set...")
                apply_mosaic_to_dataset(
                    Path(val_dir),
                    mosaic_val_dir,
                    prob=0.3,  # Lower prob for validation
                    num_augmented=len(list(Path(val_dir).glob("*.png"))),  # Same number
                    min_bbox_size=mosaic_config.get('min_bbox_size', 5),
                    border_margin=mosaic_config.get('border_margin', 10),
                    output_size=output_size
                )
                val_dir = str(mosaic_val_dir)
            else:
                print(f"✅ Mosaic-augmented validation set already exists: {mosaic_val_dir}")
                val_dir = str(mosaic_val_dir)
    
    # Initialize RF-DETR model
    print("\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    model_config = config['model']
    
    print(f"Initializing RF-DETR {model_config['rfdetr_size']} model...")
    print(f"  Architecture: {model_config['architecture']}")
    print(f"  Backbone: {model_config['backbone']}")
    print(f"  Num classes: {model_config['num_classes']}")
    print(f"  Remap MS COCO categories: {model_config['remap_mscoco_category']}")
    
    # Initialize model based on size
    size_map = {
        'nano': 'RFDETRNano',
        'small': 'RFDETRSmall',
        'medium': 'RFDETRMedium',
        'base': 'RFDETRBase',
        'large': 'RFDETRLarge'
    }
    
    rfdetr_size = model_config['rfdetr_size'].lower()
    if rfdetr_size not in size_map:
        print(f"⚠️  Warning: Unknown RF-DETR size '{rfdetr_size}', using 'base'")
        rfdetr_size = 'base'
    
    # For now, we'll use RFDETRBase (can be extended to support other sizes)
    if rfdetr_size == 'base':
        model = RFDETRBase()
    else:
        # Import other sizes if needed
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        size_class_map = {
            'nano': RFDETRNano,
            'small': RFDETRSmall,
            'medium': RFDETRMedium,
            'large': RFDETRLarge
        }
        model_class = size_class_map[rfdetr_size]
        model = model_class()
    
    print(f"✅ Model initialized: RF-DETR {rfdetr_size}")
    
    # Get training parameters (already loaded above, but ensure it's available)
    if 'training_config' not in locals():
        training_config = config['training']
    output_config = config.get('output', {})
    
    output_dir = args.output_dir or output_config.get('output_dir', 'models/ball_detection')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataset directory structure for RF-DETR
    # RF-DETR expects: dataset_dir/train/, dataset_dir/valid/, and dataset_dir/test/
    dataset_base = Path(output_dir) / "dataset"
    train_dataset_dir = dataset_base / "train"
    valid_dataset_dir = dataset_base / "valid"
    test_dataset_dir = dataset_base / "test"
    
    # Create symlinks or copy data to expected structure
    train_dataset_dir.mkdir(parents=True, exist_ok=True)
    valid_dataset_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train data
    if not (train_dataset_dir / "_annotations.coco.json").exists():
        print(f"\n📁 Setting up dataset structure...")
        # Copy images and annotations (both PNG and JPG)
        for img_file in Path(train_dir).glob("*.png"):
            shutil.copy2(img_file, train_dataset_dir / img_file.name)
        for img_file in Path(train_dir).glob("*.jpg"):
            shutil.copy2(img_file, train_dataset_dir / img_file.name)
        for img_file in Path(train_dir).glob("*.jpeg"):
            shutil.copy2(img_file, train_dataset_dir / img_file.name)
        shutil.copy2(Path(train_dir) / "_annotations.coco.json", 
                    train_dataset_dir / "_annotations.coco.json")
        print(f"✅ Train dataset ready: {train_dataset_dir}")
    
    # Copy validation data
    if val_dir != train_dir and not (valid_dataset_dir / "_annotations.coco.json").exists():
        for img_file in Path(val_dir).glob("*.png"):
            shutil.copy2(img_file, valid_dataset_dir / img_file.name)
        for img_file in Path(val_dir).glob("*.jpg"):
            shutil.copy2(img_file, valid_dataset_dir / img_file.name)
        for img_file in Path(val_dir).glob("*.jpeg"):
            shutil.copy2(img_file, valid_dataset_dir / img_file.name)
        shutil.copy2(Path(val_dir) / "_annotations.coco.json",
                    valid_dataset_dir / "_annotations.coco.json")
        print(f"✅ Validation dataset ready: {valid_dataset_dir}")
    elif val_dir == train_dir:
        # Use train for validation if no separate val set
        print(f"⚠️  Using train dataset for validation (no separate val set provided)")
        valid_dataset_dir = train_dataset_dir
    
    # RF-DETR also requires a test directory - use validation set for test
    if not (test_dataset_dir / "_annotations.coco.json").exists():
        # Copy validation data to test directory (RF-DETR requirement)
        for img_file in Path(val_dir).glob("*.png"):
            shutil.copy2(img_file, test_dataset_dir / img_file.name)
        shutil.copy2(Path(val_dir) / "_annotations.coco.json",
                    test_dataset_dir / "_annotations.coco.json")
        print(f"✅ Test dataset ready: {test_dataset_dir}")
    
    # Setup MLflow tracking
    mlflow_run = None
    logging_config = config.get('logging', {})
    dataset_config = config.get('dataset', {})
    if logging_config.get('mlflow', False) and MLFLOW_AVAILABLE:
        try:
            tracking_uri = logging_config.get('mlflow_tracking_uri', 'file:./mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment_name = logging_config.get('mlflow_experiment_name', 'ball_detection_training')
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)
            
            mlflow_run = mlflow.start_run(experiment_id=experiment_id)
            
            # Get dataset sizes for logging
            train_dataset_size = 0
            val_dataset_size = 0
            try:
                train_ann_file = Path(train_dir) / "_annotations.coco.json"
                if train_ann_file.exists():
                    with open(train_ann_file, 'r') as f:
                        train_data = json.load(f)
                        train_dataset_size = len(train_data.get('images', []))
                val_ann_file = Path(val_dir) / "_annotations.coco.json"
                if val_ann_file.exists():
                    with open(val_ann_file, 'r') as f:
                        val_data = json.load(f)
                        val_dataset_size = len(val_data.get('images', []))
            except Exception as e:
                print(f"⚠️  Could not read dataset sizes for MLflow: {e}")
            
            # Log hyperparameters
            mlflow.log_params({
                'epochs': training_config['epochs'],
                'batch_size': training_config['batch_size'],
                'learning_rate': training_config['learning_rate'],
                'grad_accum_steps': training_config['grad_accum_steps'],
                'resolution': training_config.get('resolution', 1288),
                'num_classes': model_config['num_classes'],
                'model_size': model_config['rfdetr_size'],
                'mosaic_enabled': mosaic_config.get('enabled', False) if 'mosaic_config' in locals() else False,
                'use_combined_dataset': dataset_config.get('use_combined_dataset', False),
                'train_dataset_size': train_dataset_size,
                'val_dataset_size': val_dataset_size,
            })
            
            print(f"✅ MLflow tracking enabled. Run ID: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"⚠️  Warning: MLflow setup failed: {e}")
            mlflow_run = None
    
    # Start training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Train dataset: {train_dataset_dir}")
    print(f"Validation dataset: {valid_dataset_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get dataset sizes
    try:
        with open(Path(train_dir) / "_annotations.coco.json", 'r') as f:
            train_data = json.load(f)
            train_size = len(train_data.get('images', []))
            train_annos = len(train_data.get('annotations', []))
        with open(Path(val_dir) / "_annotations.coco.json", 'r') as f:
            val_data = json.load(f)
            val_size = len(val_data.get('images', []))
            val_annos = len(val_data.get('annotations', []))
        print(f"\nDataset sizes:")
        print(f"  Training: {train_size} images, {train_annos} annotations")
        print(f"  Validation: {val_size} images, {val_annos} annotations")
    except Exception as e:
        print(f"⚠️  Could not read dataset sizes: {e}")
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Gradient accumulation steps: {training_config['grad_accum_steps']}")
    print(f"  Resolution: {training_config.get('resolution', 1288)}")
    if dataset_config.get('use_combined_dataset', False):
        print(f"  Combined dataset: SoccerSynth + Validation images OFFICIAL")
    
    # Test video path for evaluation
    test_video_path = config.get('evaluation', {}).get('test_video_path', 
        'data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4')
    
    try:
        # Train epoch by epoch to enable evaluation after each epoch
        num_epochs = training_config['epochs']
        eval_config = config.get('evaluation', {})
        eval_frequency = eval_config.get('eval_frequency', 1)
        test_video_path = eval_config.get('test_video_path', test_video_path)
        num_test_frames = eval_config.get('num_test_frames', 100)
        
        print(f"\nTraining for {num_epochs} epochs with evaluation every {eval_frequency} epoch(s)...")
        
        # Check for existing checkpoint to resume from
        checkpoint_files = list(Path(output_dir).glob("*.pth"))
        start_epoch = 0
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            try:
                import torch
                checkpoint = torch.load(str(latest_checkpoint), map_location='cpu', weights_only=False)
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"📁 Found checkpoint from epoch {checkpoint['epoch']}")
                    print(f"🔄 Resuming training from epoch {start_epoch}/{num_epochs}")
                else:
                    print(f"⚠️  Checkpoint found but no epoch info, starting from epoch 0")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint info: {e}")
                print(f"   Starting from epoch 0")
        
        # Track last checkpoint to detect new ones
        last_checkpoint_time = 0
        checkpoint_files_before = set(Path(output_dir).glob("*.pth"))
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train for 1 epoch (catch RF-DETR evaluation bug)
            try:
                model.train(
                    dataset_dir=str(dataset_base),
                    epochs=1,  # Train one epoch at a time
                    batch_size=training_config['batch_size'],
                    grad_accum_steps=training_config['grad_accum_steps'],
                    lr=training_config['learning_rate'],
                    output_dir=output_dir,
                    resolution=training_config.get('resolution', 1288),
                    device=training_config.get('device', 'cuda'),
                    num_workers=training_config.get('num_workers', 4)
                )
            except TypeError as e:
                # RF-DETR has a known bug in evaluation code (non-critical)
                if "only 0-dimensional arrays" in str(e):
                    print(f"⚠️  RF-DETR evaluation bug encountered (non-critical): {e}")
                    print("   Training completed, but evaluation had an error. Continuing...")
                else:
                    raise
            
            # Check for new checkpoint
            checkpoint_files_after = set(Path(output_dir).glob("*.pth"))
            new_checkpoints = checkpoint_files_after - checkpoint_files_before
            if new_checkpoints:
                checkpoint_files_before = checkpoint_files_after
                print(f"✅ New checkpoint created")
            
            # Log basic progress metrics to MLflow (even if evaluation fails)
            if mlflow_run:
                try:
                    mlflow.log_metric('epoch', epoch + 1, step=epoch + 1)
                    mlflow.log_metric('checkpoint_saved', 1 if new_checkpoints else 0, step=epoch + 1)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to log basic metrics to MLflow: {e}")
            
            # Evaluate after each epoch (or based on frequency)
            if (epoch + 1) % eval_frequency == 0:
                try:
                    print(f"\n{'='*60}")
                    print(f"EVALUATION (Epoch {epoch + 1})")
                    print(f"{'='*60}")
                    
                    # Use training dataset for evaluation (has ground truth)
                    from scripts.evaluate_ball_model import evaluate_on_training_dataset, load_model
                    
                    # Load latest checkpoint
                    checkpoint_path = None
                    checkpoint_files = list(Path(output_dir).glob("*.pth"))
                    if checkpoint_files:
                        checkpoint_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
                    
                    eval_model = load_model(checkpoint_path)
                    
                    # Evaluate on training dataset (subset of images)
                    num_eval_images = eval_config.get('num_eval_images', 100)
                    eval_metrics = evaluate_on_training_dataset(
                        eval_model,
                        train_dataset_dir,
                        num_images=num_eval_images,
                        confidence_threshold=0.3
                    )
                    
                    print(f"\n📊 Evaluation Results (Epoch {epoch + 1}):")
                    print(f"  Ball Recall: {eval_metrics['ball_recall']:.4f}")
                    print(f"  Ball Precision: {eval_metrics['ball_precision']:.4f}")
                    print(f"  Center Distance: {eval_metrics['ball_center_distance']:.2f} pixels")
                    print(f"  Avg Detections/Frame: {eval_metrics['avg_detections_per_frame']:.2f}")
                    
                    # Log to MLflow
                    if mlflow_run:
                        try:
                            mlflow.log_metrics({
                                'ball_recall': eval_metrics['ball_recall'],
                                'ball_precision': eval_metrics['ball_precision'],
                                'ball_center_distance': eval_metrics['ball_center_distance'],
                                'avg_detections_per_frame': eval_metrics['avg_detections_per_frame']
                            }, step=epoch + 1)
                            print("✅ Metrics logged to MLflow")
                        except Exception as e:
                            print(f"⚠️  Warning: Failed to log metrics to MLflow: {e}")
                except Exception as eval_error:
                    print(f"⚠️  Evaluation failed (non-critical): {eval_error}")
                    print("   Training will continue to next epoch...")
                    import traceback
                    traceback.print_exc()
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if mlflow_run:
            try:
                mlflow.end_run()
                print("✅ MLflow run ended")
            except Exception as e:
                print(f"⚠️  Warning: Failed to end MLflow run: {e}")


if __name__ == "__main__":
    main()
