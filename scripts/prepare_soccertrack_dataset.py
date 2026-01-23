#!/usr/bin/env python3
"""
Prepare SoccerTrack_sub dataset for training:
1. Extract frames from videos
2. Generate pseudo-labels using pre-trained model + SAHI
3. Convert to COCO format
4. Create train/val split
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import SAHI and detection model
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("Warning: SAHI not available. Install with: pip install sahi")
    print("Pseudo-labeling will be skipped. Manual annotation required.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[Path]:
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum frames to extract (None = all)
    
    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames")
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"  Extracted {extracted_count} frames")
    
    return frame_paths


def generate_pseudo_labels(
    frame_path: Path,
    detection_model,
    confidence_threshold: float = 0.1
) -> List[Dict]:
    """
    Generate pseudo-labels for a frame using pre-trained model + SAHI
    
    Args:
        frame_path: Path to frame image
        detection_model: SAHI AutoDetectionModel
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detection dicts in COCO format
    """
    if not SAHI_AVAILABLE or detection_model is None:
        return []
    
    try:
        result = get_sliced_prediction(
            str(frame_path),
            detection_model,
            slice_height=1280,
            slice_width=1280,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_metric_threshold=0.5,
            postprocess_class_agnostic=False
        )
        
        detections = []
        for obj_pred in result.object_prediction_list:
            if obj_pred.score.value < confidence_threshold:
                continue
            
            # Get bounding box
            bbox = obj_pred.bbox
            x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            
            # Convert to COCO format [x, y, width, height]
            x = float(x_min)
            y = float(y_min)
            w = float(x_max - x_min)
            h = float(y_max - y_min)
            
            # Map class name to category ID - ONLY keep player detections
            class_name = obj_pred.category.name.lower()
            if 'ball' in class_name:
                continue  # Filter out ball detections
            elif 'player' in class_name or 'person' in class_name:
                category_id = 1  # Player only
            else:
                continue  # Skip unknown classes
            
            detections.append({
                'bbox': [x, y, w, h],
                'category_id': category_id,
                'score': obj_pred.score.value,
                'area': w * h
            })
        
        return detections
    
    except Exception as e:
        print(f"Error generating labels for {frame_path}: {e}")
        return []


def create_coco_annotation(
    image_path: Path,
    image_id: int,
    width: int,
    height: int,
    detections: List[Dict]
) -> Tuple[Dict, List[Dict]]:
    """
    Create COCO format image entry and annotations
    
    Returns:
        Tuple of (image_dict, annotation_list)
    """
    image_dict = {
        'id': image_id,
        'file_name': image_path.name,
        'width': width,
        'height': height
    }
    
    annotations = []
    for ann_id, det in enumerate(detections):
        ann = {
            'id': ann_id + 1,
            'image_id': image_id,
            'category_id': det['category_id'],
            'bbox': det['bbox'],
            'area': det['area'],
            'iscrowd': 0
        }
        annotations.append(ann)
    
    return image_dict, annotations


def create_coco_dataset(
    frame_paths: List[Path],
    output_dir: Path,
    detection_model=None,
    split_ratio: float = 0.8
) -> Tuple[Path, Path]:
    """
    Create COCO format dataset from frames
    
    Args:
        frame_paths: List of frame image paths
        output_dir: Output directory for dataset
        detection_model: Optional SAHI model for pseudo-labeling
        split_ratio: Train/val split ratio (train portion)
    
    Returns:
        Tuple of (train_dir, val_dir)
    """
    # Shuffle frames
    random.shuffle(frame_paths)
    
    # Split train/val
    split_idx = int(len(frame_paths) * split_ratio)
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_frames)} frames")
    print(f"  Val: {len(val_frames)} frames")
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    train_images_dir = train_dir / "images"
    val_images_dir = val_dir / "images"
    train_annos_dir = train_dir / "annotations"
    val_annos_dir = val_dir / "annotations"
    
    for d in [train_images_dir, val_images_dir, train_annos_dir, val_annos_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process train set
    print(f"\nProcessing train set...")
    train_images = []
    train_annotations = []
    image_id = 1
    ann_id = 1
    
    for frame_path in tqdm(train_frames):
        # Copy frame to train images
        new_frame_path = train_images_dir / frame_path.name
        import shutil
        shutil.copy2(frame_path, new_frame_path)
        
        # Get image dimensions
        img = cv2.imread(str(new_frame_path))
        height, width = img.shape[:2]
        
        # Generate pseudo-labels if model provided
        detections = []
        if detection_model:
            detections = generate_pseudo_labels(new_frame_path, detection_model)
        
        # Create COCO entries
        image_dict, anns = create_coco_annotation(
            new_frame_path, image_id, width, height, detections
        )
        
        # Update annotation IDs
        for ann in anns:
            ann['id'] = ann_id
            ann_id += 1
        
        train_images.append(image_dict)
        train_annotations.extend(anns)
        image_id += 1
    
    # Process val set
    print(f"\nProcessing val set...")
    val_images = []
    val_annotations = []
    
    for frame_path in tqdm(val_frames):
        # Copy frame to val images
        new_frame_path = val_images_dir / frame_path.name
        import shutil
        shutil.copy2(frame_path, new_frame_path)
        
        # Get image dimensions
        img = cv2.imread(str(new_frame_path))
        height, width = img.shape[:2]
        
        # Generate pseudo-labels if model provided
        detections = []
        if detection_model:
            detections = generate_pseudo_labels(new_frame_path, detection_model)
        
        # Create COCO entries
        image_dict, anns = create_coco_annotation(
            new_frame_path, image_id, width, height, detections
        )
        
        # Update annotation IDs
        for ann in anns:
            ann['id'] = ann_id
            ann_id += 1
        
        val_images.append(image_dict)
        val_annotations.extend(anns)
        image_id += 1
    
    # Create COCO JSON files - ONLY player category
    categories = [
        {'id': 1, 'name': 'player', 'supercategory': 'object'}
    ]
    
    # Filter annotations to only include player (category_id=1)
    train_annotations = [ann for ann in train_annotations if ann['category_id'] == 1]
    val_annotations = [ann for ann in val_annotations if ann['category_id'] == 1]
    
    train_coco = {
        'info': {'description': 'SoccerTrack_sub Training Set'},
        'licenses': [],
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_coco = {
        'info': {'description': 'SoccerTrack_sub Validation Set'},
        'licenses': [],
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }
    
    # Save JSON files
    train_json_path = train_annos_dir / "annotations.json"
    val_json_path = val_annos_dir / "annotations.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"\n✅ Dataset created:")
    print(f"  Train: {train_json_path}")
    print(f"  Val: {val_json_path}")
    print(f"  Train images: {len(train_images)}, annotations: {len(train_annotations)}")
    print(f"  Val images: {len(val_images)}, annotations: {len(val_annotations)}")
    
    return train_dir, val_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SoccerTrack_sub dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/soccer_coach_cv/data/raw/SoccerTrack_sub",
        help="Path to SoccerTrack_sub directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/soccer_coach_cv/datasets/soccertrack",
        help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30)"
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: all)"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Generate pseudo-labels using pre-trained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model for pseudo-labeling"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for pseudo-labeling (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    videos_dir = data_dir / "videos"
    
    if not videos_dir.exists():
        raise ValueError(f"Videos directory not found: {videos_dir}")
    
    # Extract frames from all videos
    print("=" * 70)
    print("STEP 1: Extracting frames from videos")
    print("=" * 70)
    
    frames_dir = output_dir / "extracted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    all_frame_paths = []
    video_files = list(videos_dir.glob("*.mp4"))
    
    for video_path in video_files:
        frame_paths = extract_frames_from_video(
            video_path,
            frames_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames_per_video
        )
        all_frame_paths.extend(frame_paths)
    
    print(f"\n✅ Extracted {len(all_frame_paths)} total frames")
    
    # Load detection model if pseudo-labeling requested
    detection_model = None
    if args.pseudo_label:
        if not SAHI_AVAILABLE:
            print("\n⚠️  SAHI not available. Skipping pseudo-labeling.")
            print("   Install with: pip install sahi")
        else:
            print("\n" + "=" * 70)
            print("STEP 2: Loading detection model for pseudo-labeling")
            print("=" * 70)
            
            try:
                # Try to load from Roboflow or local path
                if args.model_path:
                    detection_model = AutoDetectionModel.from_pretrained(
                        model_type='rfdetr',
                        model_path=args.model_path,
                        confidence_threshold=args.confidence_threshold,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'
                    )
                else:
                    # Try Roboflow model (requires API key)
                    print("Attempting to load Roboflow RF-DETR model...")
                    detection_model = AutoDetectionModel.from_pretrained(
                        model_type='rfdetr',
                        model_name='rfdetr-base',
                        confidence_threshold=args.confidence_threshold,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'
                    )
                print("✅ Detection model loaded")
            except Exception as e:
                print(f"⚠️  Could not load detection model: {e}")
                print("   Continuing without pseudo-labeling...")
                detection_model = None
    
    # Create COCO dataset
    print("\n" + "=" * 70)
    print("STEP 3: Creating COCO format dataset")
    print("=" * 70)
    
    train_dir, val_dir = create_coco_dataset(
        all_frame_paths,
        output_dir,
        detection_model=detection_model,
        split_ratio=args.split_ratio
    )
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print(f"\nTrain directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    print(f"\nNext steps:")
    print(f"1. Review pseudo-labels (if generated)")
    print(f"2. Manually refine annotations in CVAT or similar tool")
    print(f"3. Start training with:")
    print(f"   python scripts/train_detr.py \\")
    print(f"       --config configs/training_soccertrack_phase1.yaml \\")
    print(f"       --train-dir {train_dir} \\")
    print(f"       --val-dir {val_dir}")


if __name__ == "__main__":
    main()
