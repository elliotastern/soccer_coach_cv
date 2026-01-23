#!/usr/bin/env python3
"""
Prepare soccerchallenge2025 dataset for training:
1. Extract frames from video
2. Parse .txt annotations (frame_id,player_id,center_x,center_y,width,height)
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

sys.path.append(str(Path(__file__).parent.parent))


def parse_annotations_file(txt_path: Path) -> Dict[int, List[Dict]]:
    """
    Parse soccerchallenge2025 annotation file
    
    Format: frame_id,player_id,center_x,center_y,width,height,-1,-1,-1,-1
    
    Returns:
        Dict mapping frame_id -> List of annotation dicts
    """
    annotations_by_frame = {}
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            player_id = int(parts[1])
            center_x = float(parts[2])
            center_y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            
            # Convert center format to COCO format (top-left x, y, width, height)
            x = center_x - width / 2.0
            y = center_y - height / 2.0
            
            # Ensure non-negative coordinates
            x = max(0, x)
            y = max(0, y)
            
            if frame_id not in annotations_by_frame:
                annotations_by_frame[frame_id] = []
            
            annotations_by_frame[frame_id].append({
                'player_id': player_id,
                'bbox': [x, y, width, height],  # COCO format: [x, y, w, h]
                'area': width * height
            })
    
    return annotations_by_frame


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[Tuple[int, Path]]:
    """
    Extract frames from video at specified intervals
    
    Returns:
        List of (frame_id, frame_path) tuples
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Extracting every {frame_interval} frames")
    
    frame_data = []
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
            frame_data.append((frame_count, frame_path))
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"  Extracted {extracted_count} frames")
    
    return frame_data


def create_coco_dataset(
    frame_data: List[Tuple[int, Path]],
    annotations_by_frame: Dict[int, List[Dict]],
    output_dir: Path,
    split_ratio: float = 0.8
) -> Tuple[Path, Path]:
    """
    Create COCO format dataset from frames and annotations
    
    Args:
        frame_data: List of (frame_id, frame_path) tuples
        annotations_by_frame: Dict mapping frame_id -> annotations
        output_dir: Output directory for dataset
        split_ratio: Train/val split ratio (train portion)
    
    Returns:
        Tuple of (train_dir, val_dir)
    """
    # Shuffle frames
    random.shuffle(frame_data)
    
    # Split train/val
    split_idx = int(len(frame_data) * split_ratio)
    train_frames = frame_data[:split_idx]
    val_frames = frame_data[split_idx:]
    
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
    
    for frame_id, frame_path in tqdm(train_frames):
        # Copy frame to train images
        new_frame_path = train_images_dir / frame_path.name
        import shutil
        shutil.copy2(frame_path, new_frame_path)
        
        # Get image dimensions
        img = cv2.imread(str(new_frame_path))
        height, width = img.shape[:2]
        
        # Get annotations for this frame
        frame_annos = annotations_by_frame.get(frame_id, [])
        
        # Create image entry
        image_dict = {
            'id': image_id,
            'file_name': frame_path.name,
            'width': width,
            'height': height
        }
        
        # Create annotation entries
        for anno in frame_annos:
            ann = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': 1,  # Player (only class)
                'bbox': anno['bbox'],
                'area': anno['area'],
                'iscrowd': 0
            }
            train_annotations.append(ann)
            ann_id += 1
        
        train_images.append(image_dict)
        image_id += 1
    
    # Process val set
    print(f"\nProcessing val set...")
    val_images = []
    val_annotations = []
    
    for frame_id, frame_path in tqdm(val_frames):
        # Copy frame to val images
        new_frame_path = val_images_dir / frame_path.name
        import shutil
        shutil.copy2(frame_path, new_frame_path)
        
        # Get image dimensions
        img = cv2.imread(str(new_frame_path))
        height, width = img.shape[:2]
        
        # Get annotations for this frame
        frame_annos = annotations_by_frame.get(frame_id, [])
        
        # Create image entry
        image_dict = {
            'id': image_id,
            'file_name': frame_path.name,
            'width': width,
            'height': height
        }
        
        # Create annotation entries
        for anno in frame_annos:
            ann = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': 1,  # Player (only class)
                'bbox': anno['bbox'],
                'area': anno['area'],
                'iscrowd': 0
            }
            val_annotations.append(ann)
            ann_id += 1
        
        val_images.append(image_dict)
        image_id += 1
    
    # Create COCO JSON files
    categories = [
        {'id': 1, 'name': 'player', 'supercategory': 'object'}
    ]
    
    train_coco = {
        'info': {'description': 'SoccerChallenge2025 Training Set'},
        'licenses': [],
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_coco = {
        'info': {'description': 'SoccerChallenge2025 Validation Set'},
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
        description="Prepare SoccerChallenge2025 dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/soccer_coach_cv/data/raw/soccerchallenge2025",
        help="Path to soccerchallenge2025 directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/soccer_coach_cv/datasets/soccerchallenge",
        help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=15,
        help="Extract every Nth frame (default: 15 for 2 fps at 30fps video)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Maximum frames to extract (default: 2000)"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Find video and annotation files
    video_files = list(data_dir.glob("*.mp4"))
    txt_files = list(data_dir.glob("*.txt"))
    
    if not video_files:
        raise ValueError(f"No video files found in {data_dir}")
    if not txt_files:
        raise ValueError(f"No annotation files found in {data_dir}")
    
    # Use first video and matching txt file
    video_path = video_files[0]
    txt_path = txt_files[0]
    
    print("=" * 70)
    print("STEP 1: Parsing annotations")
    print("=" * 70)
    print(f"Annotation file: {txt_path}")
    
    annotations_by_frame = parse_annotations_file(txt_path)
    print(f"✅ Parsed annotations for {len(annotations_by_frame)} frames")
    
    # Count total annotations
    total_annos = sum(len(annos) for annos in annotations_by_frame.values())
    print(f"   Total annotations: {total_annos}")
    
    # Extract frames
    print("\n" + "=" * 70)
    print("STEP 2: Extracting frames from video")
    print("=" * 70)
    
    frames_dir = output_dir / "extracted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_data = extract_frames_from_video(
        video_path,
        frames_dir,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames
    )
    
    print(f"\n✅ Extracted {len(frame_data)} frames")
    
    # Create COCO dataset
    print("\n" + "=" * 70)
    print("STEP 3: Creating COCO format dataset")
    print("=" * 70)
    
    train_dir, val_dir = create_coco_dataset(
        frame_data,
        annotations_by_frame,
        output_dir,
        split_ratio=args.split_ratio
    )
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print(f"\nTrain directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    print(f"\nNext steps:")
    print(f"1. Start training with:")
    print(f"   python scripts/train_detr.py \\")
    print(f"       --config configs/training_soccertrack_phase1.yaml \\")
    print(f"       --train-dir {train_dir} \\")
    print(f"       --val-dir {val_dir}")


if __name__ == "__main__":
    main()
