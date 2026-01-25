#!/usr/bin/env python3
"""
Prepare RF-DETR training dataset from SoccerTrack wide_view CSV annotations
Converts CSV annotations to COCO format for RF-DETR training
"""
import cv2
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
from tqdm import tqdm


def parse_csv_annotations(csv_path: str):
    """
    Parse CSV file and extract ground truth bounding boxes
    Same logic as generate_ground_truth_samples.py
    """
    with open(csv_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse headers
    team_row = lines[0].split(',')
    player_row = lines[1].split(',')
    attr_row = lines[2].split(',')
    
    annotations = {}  # frame_id -> list of detections
    
    # Parse data rows (starting from row 4, index 4)
    for line in lines[4:]:
        parts = line.split(',')
        if not parts[0] or not parts[0].isdigit():
            continue
        
        frame_id = int(parts[0])
        detections = []
        
        # Parse each player/ball (4 columns per entity)
        i = 1
        while i < len(parts) - 3:
            team_id = team_row[i] if i < len(team_row) else None
            player_id = player_row[i] if i < len(player_row) else None
            attr = attr_row[i] if i < len(attr_row) else None
            
            if not team_id or not player_id or not attr:
                break
            
            # Filter out ball - only include players (TeamID 0 and 1)
            if team_id == 'BALL':
                i += 4
                continue
            
            # Only process players (TeamID 0 or 1)
            if team_id not in ['0', '1']:
                i += 4
                continue
            
            # Get the 4 values for this entity
            if i + 3 >= len(parts):
                break
            
            try:
                bb_height = float(parts[i]) if parts[i] else 0
                bb_left = float(parts[i+1]) if parts[i+1] else 0
                bb_top = float(parts[i+2]) if parts[i+2] else 0
                bb_width = float(parts[i+3]) if parts[i+3] else 0
            except (ValueError, IndexError):
                i += 4
                continue
            
            # Skip if all zeros (no detection)
            if bb_left == 0 and bb_top == 0 and bb_width == 0 and bb_height == 0:
                i += 4
                continue
            
            # COCO format: [x_min, y_min, width, height]
            # CSV format: (bb_left, bb_top, bb_width, bb_height) = (x_min, y_min, width, height)
            # They match! No conversion needed
            bbox = [bb_left, bb_top, bb_width, bb_height]
            area = bb_width * bb_height
            
            # All players (both teams) map to single "person" class (category_id=1)
            detections.append({
                'bbox': bbox,  # [x_min, y_min, width, height]
                'area': area,
                'category_id': 1  # "person" - all players combined, no team distinction
            })
            
            i += 4
        
        if detections:
            annotations[frame_id] = detections
    
    return annotations


def extract_frames_from_video(
    video_path: Path,
    annotations: Dict[int, List[Dict]],
    output_dir: Path,
    frame_interval: int = 30,
    video_name: str = None
) -> List[Tuple[str, int, int, int]]:
    """
    Extract frames from video at specified intervals
    Returns list of (image_filename, frame_num, width, height) tuples
    """
    if video_name is None:
        video_name = video_path.stem
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted_frames = []
    frame_numbers = sorted([f for f in annotations.keys() if f <= total_frames])
    
    # Extract frames at specified interval
    for frame_num in frame_numbers[::frame_interval]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-indexed
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Generate filename
        image_filename = f"{video_name}_frame_{frame_num:06d}.jpg"
        output_path = output_dir / image_filename
        
        # Save frame
        cv2.imwrite(str(output_path), frame)
        extracted_frames.append((image_filename, frame_num, width, height))
    
    cap.release()
    return extracted_frames


def create_coco_annotations(
    all_images: List[Dict],
    all_annotations: List[Dict],
    output_path: Path
):
    """Create COCO format annotations.json file"""
    coco_data = {
        "info": {
            "description": "SoccerTrack Wide View - Player Detection Dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": all_images,
        "annotations": all_annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person"
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Created COCO annotations: {output_path}")
    print(f"   Images: {len(all_images)}")
    print(f"   Annotations: {len(all_annotations)}")


def prepare_rf_detr_dataset(
    videos_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    frame_interval: int = 30,
    train_split: float = 0.8
):
    """Prepare RF-DETR training dataset from SoccerTrack wide_view"""
    print("=" * 70)
    print("PREPARING RF-DETR TRAINING DATASET")
    print("=" * 70)
    print()
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_images_dir = train_dir / "images"
    val_images_dir = val_dir / "images"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(annotations_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV annotation files")
    print()
    
    # Split videos into train/val
    random.seed(42)  # For reproducibility
    random.shuffle(csv_files)
    split_idx = int(len(csv_files) * train_split)
    train_csvs = csv_files[:split_idx]
    val_csvs = csv_files[split_idx:]
    
    print(f"Train videos: {len(train_csvs)}")
    print(f"Val videos: {len(val_csvs)}")
    print()
    
    # Process train set
    print("Processing TRAIN set...")
    print("-" * 70)
    train_images = []
    train_annotations = []
    image_id = 1
    annotation_id = 1
    
    for csv_path in tqdm(train_csvs, desc="Train videos"):
        video_name = csv_path.stem
        video_path = videos_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        # Parse CSV annotations (all players combined as "person" class)
        annotations = parse_csv_annotations(str(csv_path))
        
        if not annotations:
            continue
        
        # Extract frames
        extracted = extract_frames_from_video(
            video_path, annotations, train_images_dir,
            frame_interval=frame_interval, video_name=video_name
        )
        
        # Create COCO format entries
        for image_filename, frame_num, width, height in extracted:
            # Add image entry
            train_images.append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            })
            
            # Add annotations for this frame
            if frame_num in annotations:
                for det in annotations[frame_num]:
                    train_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": det['category_id'],
                        "bbox": det['bbox'],  # [x_min, y_min, width, height]
                        "area": det['area'],
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1
    
    # Process val set
    print()
    print("Processing VAL set...")
    print("-" * 70)
    val_images = []
    val_annotations = []
    
    for csv_path in tqdm(val_csvs, desc="Val videos"):
        video_name = csv_path.stem
        video_path = videos_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        # Parse CSV annotations (all players combined as "person" class)
        annotations = parse_csv_annotations(str(csv_path))
        
        if not annotations:
            continue
        
        # Extract frames
        extracted = extract_frames_from_video(
            video_path, annotations, val_images_dir,
            frame_interval=frame_interval, video_name=video_name
        )
        
        # Create COCO format entries
        for image_filename, frame_num, width, height in extracted:
            # Add image entry
            val_images.append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            })
            
            # Add annotations for this frame
            if frame_num in annotations:
                for det in annotations[frame_num]:
                    val_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": det['category_id'],
                        "bbox": det['bbox'],  # [x_min, y_min, width, height]
                        "area": det['area'],
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1
    
    # Create COCO annotation files
    print()
    print("Creating COCO annotation files...")
    print("-" * 70)
    create_coco_annotations(train_images, train_annotations, train_dir / "annotations.json")
    create_coco_annotations(val_images, val_annotations, val_dir / "annotations.json")
    
    print()
    print("=" * 70)
    print("✅ DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Train images: {len(train_images)}")
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Val images: {len(val_images)}")
    print(f"Val annotations: {len(val_annotations)}")
    print()
    print("Dataset is ready for RF-DETR training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RF-DETR training dataset from SoccerTrack wide_view")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="data/raw/soccertrack/wide_view/videos",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/raw/soccertrack/wide_view/annotations",
        help="Directory containing CSV annotation files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/rf_detr_soccertrack",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    prepare_rf_detr_dataset(
        videos_dir=Path(args.videos_dir),
        annotations_dir=Path(args.annotations_dir),
        output_dir=Path(args.output_dir),
        frame_interval=args.frame_interval,
        train_split=args.train_split
    )
