#!/usr/bin/env python3
"""
Evaluate ball detection model on test video or training dataset.
Calculates recall, precision, and center distance using Hungarian algorithm.
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Error: RF-DETR not installed")
    sys.exit(1)

from src.types import Detection


def load_model(checkpoint_path: str = None):
    """Load RF-DETR model, optionally from checkpoint."""
    import torch
    model = None
    
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            # Load checkpoint to get config
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Get num_classes from checkpoint args
            num_classes = 1  # Default for ball-only
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'num_classes'):
                num_classes = checkpoint['args'].num_classes
            
            # Initialize model with correct num_classes
            # RF-DETR uses class_names to determine num_classes
            class_names = ['ball'] if num_classes == 1 else None
            if class_names:
                model = RFDETRBase(class_names=class_names)
            else:
                model = RFDETRBase()
            
            # RF-DETR checkpoint structure: {'model': state_dict, 'optimizer': ..., 'epoch': ...}
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                # RF-DETR model structure: model.model.model is the actual PyTorch model
                # Path: RFDETRBase -> Model -> LWDETR (PyTorch nn.Module)
                if hasattr(model, 'model') and hasattr(model.model, 'model'):
                    # Filter out class embedding layers that have size mismatches
                    # The checkpoint has num_classes=1, but RFDETRBase() defaults to 91 COCO classes
                    # We'll skip the class embedding layers and load the rest
                    current_model_state = model.model.model.state_dict()
                    filtered_state = {}
                    skipped_keys = []
                    
                    for key, value in model_state.items():
                        if key in current_model_state:
                            if current_model_state[key].shape == value.shape:
                                filtered_state[key] = value
                            else:
                                skipped_keys.append(key)
                        else:
                            skipped_keys.append(key)
                    
                    # Load filtered state dict
                    missing_keys, unexpected_keys = model.model.model.load_state_dict(filtered_state, strict=False)
                    if skipped_keys:
                        print(f"⚠️  Skipped {len(skipped_keys)} keys due to size mismatch (class embeddings)")
                    if missing_keys:
                        print(f"⚠️  {len(missing_keys)} missing keys")
                    if unexpected_keys:
                        print(f"⚠️  {len(unexpected_keys)} unexpected keys")
                    print(f"✅ Loaded checkpoint weights from: {checkpoint_path}")
                    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
                    print(f"   Loaded {len(filtered_state)}/{len(model_state)} layers")
                else:
                    print(f"⚠️  Warning: Could not find model.model.model to load weights")
                    print(f"   Using base pretrained weights")
            else:
                print(f"⚠️  Warning: Checkpoint does not contain 'model' key")
                print(f"   Using base pretrained weights")
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to base model without checkpoint")
            model = RFDETRBase()
    else:
        if checkpoint_path:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}, using base pretrained weights")
        model = RFDETRBase()
    
    return model


def detect_balls(model, frame: np.ndarray, confidence_threshold: float = 0.3, use_sahi: bool = False, 
                 sahi_slice_size: int = 1288, sahi_overlap_ratio: float = 0.2) -> List[Detection]:
    """
    Detect balls in frame using RF-DETR model.
    
    Args:
        model: RF-DETR model
        frame: BGR image array (OpenCV format)
        confidence_threshold: Minimum confidence for detections
        use_sahi: If True, use SAHI (Slicing Aided Hyper Inference) for better tiny object detection
        sahi_slice_size: Size of each slice for SAHI (default: 1288 to match training resolution)
        sahi_overlap_ratio: Overlap ratio between slices (default: 0.2 = 20% overlap)
    
    Returns:
        List of Detection objects for balls only
    """
    from PIL import Image
    
    # Convert BGR to RGB and to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    if use_sahi:
        # Use SAHI for better tiny object detection
        return detect_balls_sahi(model, pil_image, confidence_threshold, sahi_slice_size, sahi_overlap_ratio)
    else:
        # Standard single-pass inference
        return detect_balls_standard(model, pil_image, confidence_threshold)


def detect_balls_standard(model, pil_image: Image.Image, confidence_threshold: float = 0.3) -> List[Detection]:
    """Standard single-pass ball detection."""
    # Run inference using RF-DETR's predict method
    detections_raw = model.predict(pil_image, threshold=confidence_threshold)
    
    # Filter for ball detections only
    ball_detections = []
    
    # RF-DETR returns an object with attributes: class_id, confidence, xyxy
    if hasattr(detections_raw, 'class_id'):
        num_detections = len(detections_raw.class_id)
        for i in range(num_detections):
            class_id = int(detections_raw.class_id[i])
            confidence = float(detections_raw.confidence[i])
            bbox_xyxy = detections_raw.xyxy[i]  # [x_min, y_min, x_max, y_max]
            
            # For single-class ball model, class_id should be 0
            if class_id == 0:  # Ball class in single-class model
                x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
                width = x_max - x_min
                height = y_max - y_min
                
                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue
                
                ball_detections.append(Detection(
                    class_id=0,
                    confidence=confidence,
                    bbox=(x_min, y_min, width, height),
                    class_name="ball"
                ))
    else:
        # Fallback: try list/dict format
        for det in detections_raw:
            if isinstance(det, dict):
                class_id = det.get('class_id', -1)
                if class_id == 0:  # Ball class
                    confidence = det.get('confidence', 0.0)
                    bbox = det.get('bbox', [])
                    if len(bbox) == 4:
                        x_min, y_min, x_max, y_max = bbox
                        width = x_max - x_min
                        height = y_max - y_min
                        if width > 0 and height > 0:
                            ball_detections.append(Detection(
                                class_id=0,
                                confidence=confidence,
                                bbox=(x_min, y_min, width, height),
                                class_name="ball"
                            ))
    
    return ball_detections


def detect_balls_sahi(model, pil_image: Image.Image, confidence_threshold: float = 0.3,
                      slice_size: int = 1288, overlap_ratio: float = 0.2) -> List[Detection]:
    """
    Detect balls using SAHI (Slicing Aided Hyper Inference).
    Splits image into overlapping tiles, detects on each, then merges results.
    
    Optimized for speed:
    - Try standard detection first (fast path)
    - Use larger slices to reduce tile count
    - Reduced overlap for fewer tiles
    - Smart tile selection (center + edges)
    """
    import torch
    import torchvision.ops as ops
    
    width, height = pil_image.size
    
    # If image is smaller than slice size, use standard detection
    if width <= slice_size and height <= slice_size:
        return detect_balls_standard(model, pil_image, confidence_threshold)
    
    # Optimization 1: Try standard detection first, only use SAHI if no detections
    # This speeds up frames where standard detection works
    standard_detections = detect_balls_standard(model, pil_image, confidence_threshold)
    if len(standard_detections) > 0:
        # Standard detection found something, return it (fast path)
        return standard_detections
    
    # Optimization 2: Use much larger slices for big images to reduce tile count
    # For 2880x3840 images, use 1920x1920 slices (reduces from ~9 tiles to ~4 tiles)
    if width > 2000 or height > 2000:
        # Use larger slices for big images - prioritize speed
        adaptive_slice_size = min(width, height) // 2
        adaptive_slice_size = min(adaptive_slice_size, 1920)  # Cap at 1920 for speed
        slice_size = adaptive_slice_size
        overlap_ratio = 0.1  # Minimal overlap (10%) for maximum speed
    
    # Optimization 3: Ultra-fast smart tile selection - only essential tiles
    # Use minimal tile set: center + 4 corners (5 tiles total instead of 9)
    center_x = width // 2
    center_y = height // 2
    
    # Collect detections from all slices
    all_detections = []
    
    # Generate minimal strategic slices: center + corners only
    tile_positions = []
    
    # Center tile (most important - where ball usually is)
    center_tile_x = max(0, center_x - slice_size // 2)
    center_tile_y = max(0, center_y - slice_size // 2)
    tile_positions.append((center_tile_x, center_tile_y))
    
    # Only add corner tiles if image is significantly larger than slice
    if width > slice_size * 1.5 and height > slice_size * 1.5:
        # Top-left corner
        tile_positions.append((0, 0))
        # Top-right corner
        tile_positions.append((max(0, width - slice_size), 0))
        # Bottom-left corner
        tile_positions.append((0, max(0, height - slice_size)))
        # Bottom-right corner
        tile_positions.append((max(0, width - slice_size), max(0, height - slice_size)))
    
    # Remove duplicates
    tile_positions = list(set(tile_positions))
    
    for x, y in tile_positions:
        # Calculate slice bounds
        x_end = min(x + slice_size, width)
        y_end = min(y + slice_size, height)
        
        # Ensure minimum size
        if x_end - x < slice_size * 0.8:  # Allow slightly smaller at edges
            continue
        if y_end - y < slice_size * 0.8:
            continue
        
        # Crop slice
        slice_img = pil_image.crop((x, y, x_end, y_end))
        
        # Detect on slice
        slice_detections = detect_balls_standard(model, slice_img, confidence_threshold)
        
        # Adjust coordinates to original image space
        for det in slice_detections:
            x_orig, y_orig, w, h = det.bbox
            # Adjust bbox coordinates
            adjusted_det = Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=(x_orig + x, y_orig + y, w, h),
                class_name=det.class_name
            )
            all_detections.append(adjusted_det)
    
    # Merge overlapping detections using NMS
    if len(all_detections) == 0:
        return []
    
    # Convert to tensor format for NMS
    boxes = []
    scores = []
    for det in all_detections:
        x, y, w, h = det.bbox
        boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
        scores.append(det.confidence)
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    
    # Return filtered detections
    merged_detections = [all_detections[i] for i in keep_indices.tolist()]
    
    return merged_detections


def calculate_center_distance(centers1: np.ndarray, centers2: np.ndarray) -> float:
    """
    Calculate average center distance using Hungarian Algorithm.
    
    Args:
        centers1: [N, 2] array of (x, y) centers (predictions)
        centers2: [M, 2] array of (x, y) centers (ground truth)
    
    Returns:
        Average center distance for matched pairs, or inf if no matches
    """
    if len(centers1) == 0 or len(centers2) == 0:
        return float('inf') if len(centers2) > 0 else 0.0
    
    # Build cost matrix: distances between all pairs
    cost_matrix = np.sqrt(((centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Use Hungarian algorithm to find optimal matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate average distance for matched pairs
    matched_distances = cost_matrix[row_indices, col_indices]
    avg_distance = np.mean(matched_distances) if len(matched_distances) > 0 else 0.0
    
    return float(avg_distance)


def evaluate_on_video(
    model,
    video_path: str,
    num_frames: int = 100,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    gt_annotations: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None
) -> Dict[str, float]:
    """
    Evaluate model on first N frames of video.
    
    Args:
        model: RF-DETR model
        video_path: Path to test video
        num_frames: Number of frames to evaluate
        confidence_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for matching
        gt_annotations: Optional ground truth annotations dict (frame_id -> List of bboxes)
    
    Returns:
        Dictionary with metrics: recall, precision, center_distance
    """
    # Use ground truth evaluation if available
    if gt_annotations is not None:
        return evaluate_with_ground_truth(
            model, video_path, gt_annotations, num_frames, confidence_threshold, iou_threshold
        )
    
    # Otherwise, use detection statistics (approximation)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_detections = 0
    frames_with_detections = 0
    all_ball_centers = []
    
    print(f"Evaluating on first {num_frames} frames (no ground truth available)...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect balls
        detections = detect_balls(model, frame, confidence_threshold)
        
        if len(detections) > 0:
            frames_with_detections += 1
            total_detections += len(detections)
            
            # Extract centers
            for det in detections:
                x, y, w, h = det.bbox
                center_x = x + w / 2
                center_y = y + h / 2
                all_ball_centers.append([center_x, center_y])
        
        if (frame_num + 1) % 20 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Calculate detection statistics (approximation without GT)
    avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0.0
    detection_rate = frames_with_detections / num_frames if num_frames > 0 else 0.0
    
    # Note: Without ground truth, we can't calculate true recall/precision
    # These are approximations based on detection statistics
    metrics = {
        'ball_recall': detection_rate,  # Approximation: detection rate as proxy for recall
        'ball_precision': 0.0,  # Cannot calculate without GT
        'ball_center_distance': 0.0,  # Cannot calculate without GT
        'avg_detections_per_frame': avg_detections_per_frame,
        'frames_with_detections': frames_with_detections,
        'total_frames': num_frames
    }
    
    return metrics


def evaluate_with_ground_truth(
    model,
    video_path: str,
    gt_annotations: Dict[int, List[Tuple[float, float, float, float]]],
    num_frames: int = 100,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model with ground truth annotations.
    
    Args:
        model: RF-DETR model
        video_path: Path to test video
        gt_annotations: Dict mapping frame_id -> List of (x, y, w, h) bboxes
        num_frames: Number of frames to evaluate
        confidence_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with metrics: recall, precision, center_distance
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    ball_tp = 0  # True positives
    ball_fp = 0  # False positives
    ball_fn = 0  # False negatives
    
    ball_centers_pred = []
    ball_centers_gt = []
    
    print(f"Evaluating on first {num_frames} frames with ground truth...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ground truth for this frame
        gt_boxes = gt_annotations.get(frame_num, [])
        
        # Detect balls
        pred_detections = detect_balls(model, frame, confidence_threshold)
        
        # Extract centers
        pred_centers = np.array([
            [det.bbox[0] + det.bbox[2] / 2, det.bbox[1] + det.bbox[3] / 2]
            for det in pred_detections
        ])
        
        gt_centers = np.array([
            [box[0] + box[2] / 2, box[1] + box[3] / 2]
            for box in gt_boxes
        ])
        
        # Match predictions to ground truth using Hungarian algorithm
        if len(pred_centers) > 0 and len(gt_centers) > 0:
            # Calculate IoU for matching
            pred_boxes = np.array([[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] 
                                  for d in pred_detections])
            gt_boxes_array = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes])
            
            # Compute IoU matrix
            ious = compute_ious(pred_boxes, gt_boxes_array)
            
            # Match using Hungarian algorithm on IoU
            cost_matrix = 1.0 - ious  # Convert IoU to cost (higher IoU = lower cost)
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Count matches above IoU threshold
            matched_gt = np.zeros(len(gt_boxes), dtype=bool)
            for r, c in zip(row_indices, col_indices):
                if ious[r, c] >= iou_threshold:
                    ball_tp += 1
                    matched_gt[c] = True
                else:
                    ball_fp += 1
            
            # Unmatched predictions are false positives
            ball_fp += len(pred_detections) - len([r for r, c in zip(row_indices, col_indices) if ious[r, c] >= iou_threshold])
            
            # Unmatched ground truth are false negatives
            ball_fn += np.sum(~matched_gt)
            
            # Store centers for distance calculation
            if len(pred_centers) > 0 and len(gt_centers) > 0:
                ball_centers_pred.append(pred_centers)
                ball_centers_gt.append(gt_centers)
        elif len(pred_centers) > 0:
            # Predictions but no ground truth = false positives
            ball_fp += len(pred_centers)
        elif len(gt_centers) > 0:
            # Ground truth but no predictions = false negatives
            ball_fn += len(gt_centers)
        
        if (frame_num + 1) % 20 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Calculate metrics
    ball_recall = ball_tp / (ball_tp + ball_fn) if (ball_tp + ball_fn) > 0 else 0.0
    ball_precision = ball_tp / (ball_tp + ball_fp) if (ball_tp + ball_fp) > 0 else 0.0
    
    # Calculate center distance
    center_distances = []
    for pred_centers, gt_centers in zip(ball_centers_pred, ball_centers_gt):
        if len(gt_centers) > 0:
            dist = calculate_center_distance(pred_centers, gt_centers)
            if dist != float('inf'):
                center_distances.append(dist)
    
    avg_center_distance = np.mean(center_distances) if center_distances else 0.0
    
    metrics = {
        'ball_recall': ball_recall,
        'ball_precision': ball_precision,
        'ball_center_distance': avg_center_distance,
        'ball_tp': ball_tp,
        'ball_fp': ball_fp,
        'ball_fn': ball_fn
    }
    
    return metrics


def compute_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersections
    x_min = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y_min = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x_max = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y_max = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    
    # Compute union
    union = area1[:, np.newaxis] + area2 - intersection
    
    # Compute IoU
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ball detection model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--video", type=str, default="data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4", help="Test video path")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to evaluate")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    metrics = evaluate_on_video(model, args.video, args.num_frames, args.confidence)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Ball Recall: {metrics['ball_recall']:.4f}")
    print(f"Ball Precision: {metrics['ball_precision']:.4f}")
    print(f"Center Distance: {metrics['ball_center_distance']:.2f} pixels")
    print(f"Avg Detections/Frame: {metrics['avg_detections_per_frame']:.2f}")


def evaluate_on_training_dataset(
    model,
    dataset_dir: str,
    num_images: int = 100,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on training dataset with ground truth annotations.
    
    Args:
        model: RF-DETR model
        dataset_dir: Directory containing images and _annotations.coco.json
        num_images: Number of images to evaluate (default: 100)
        confidence_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with metrics: recall, precision, center_distance, avg_detections_per_frame
    """
    dataset_path = Path(dataset_dir)
    annotation_file = dataset_path / "_annotations.coco.json"
    
    if not annotation_file.exists():
        raise ValueError(f"Annotation file not found: {annotation_file}")
    
    # Load COCO annotations
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build mappings
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Find ball category ID
    ball_category_id = None
    for cat_id, cat_name in categories.items():
        if cat_name.lower() == 'ball':
            ball_category_id = int(cat_id)
            break
    
    if ball_category_id is None:
        raise ValueError("Ball category not found in annotations")
    
    # Group annotations by image (ball only)
    image_annotations = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == ball_category_id:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            # COCO format: [x_min, y_min, width, height]
            image_annotations[img_id].append(ann['bbox'])
    
    # Select images to evaluate (limit to num_images)
    image_ids = sorted([img_id for img_id in images.keys() if img_id in image_annotations])
    image_ids = image_ids[:num_images]
    
    if len(image_ids) == 0:
        raise ValueError("No images with ball annotations found")
    
    print(f"Evaluating on {len(image_ids)} training images with ground truth...")
    
    ball_tp = 0
    ball_fp = 0
    ball_fn = 0
    ball_centers_pred = []
    ball_centers_gt = []
    total_detections = 0
    
    for idx, image_id in enumerate(image_ids):
        image_info = images[image_id]
        image_path = dataset_path / image_info['file_name']
        
        if not image_path.exists():
            continue
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        frame = np.array(image)
        
        # Get ground truth boxes for this image
        gt_boxes = image_annotations[image_id]
        
        # Detect balls
        pred_detections = detect_balls(model, frame, confidence_threshold)
        total_detections += len(pred_detections)
        
        # Extract centers
        pred_centers = np.array([
            [det.bbox[0] + det.bbox[2] / 2, det.bbox[1] + det.bbox[3] / 2]
            for det in pred_detections
        ])
        
        gt_centers = np.array([
            [box[0] + box[2] / 2, box[1] + box[3] / 2]
            for box in gt_boxes
        ])
        
        # Match predictions to ground truth using Hungarian algorithm
        if len(pred_centers) > 0 and len(gt_centers) > 0:
            # Calculate IoU for matching
            pred_boxes = np.array([[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] 
                                  for d in pred_detections])
            gt_boxes_array = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes])
            
            # Compute IoU matrix
            ious = compute_ious(pred_boxes, gt_boxes_array)
            
            # Match using Hungarian algorithm on IoU
            cost_matrix = 1.0 - ious
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Count matches above IoU threshold
            matched_gt = np.zeros(len(gt_boxes), dtype=bool)
            for r, c in zip(row_indices, col_indices):
                if ious[r, c] >= iou_threshold:
                    ball_tp += 1
                    matched_gt[c] = True
                    ball_centers_pred.append(pred_centers[r])
                    ball_centers_gt.append(gt_centers[c])
                else:
                    ball_fp += 1
            
            # Unmatched predictions are false positives
            matched_count = len([r for r, c in zip(row_indices, col_indices) if ious[r, c] >= iou_threshold])
            ball_fp += len(pred_detections) - matched_count
            
            # Unmatched ground truth are false negatives
            ball_fn += np.sum(~matched_gt)
        elif len(pred_centers) > 0:
            # Predictions but no ground truth = false positives
            ball_fp += len(pred_centers)
        elif len(gt_centers) > 0:
            # Ground truth but no predictions = false negatives
            ball_fn += len(gt_centers)
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(image_ids)} images...")
    
    # Calculate metrics
    ball_recall = ball_tp / (ball_tp + ball_fn) if (ball_tp + ball_fn) > 0 else 0.0
    ball_precision = ball_tp / (ball_tp + ball_fp) if (ball_tp + ball_fp) > 0 else 0.0
    
    # Calculate average center distance for matched detections
    if len(ball_centers_pred) > 0:
        distances = [np.linalg.norm(pred - gt) for pred, gt in zip(ball_centers_pred, ball_centers_gt)]
        avg_center_distance = np.mean(distances)
    else:
        avg_center_distance = 0.0
    
    avg_detections_per_frame = total_detections / len(image_ids) if len(image_ids) > 0 else 0.0
    
    metrics = {
        'ball_recall': ball_recall,
        'ball_precision': ball_precision,
        'ball_center_distance': avg_center_distance,
        'avg_detections_per_frame': avg_detections_per_frame
    }
    
    return metrics
