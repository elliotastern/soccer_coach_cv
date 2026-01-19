#!/usr/bin/env python3
"""
Test RF-DETR (Roboflow DETR) on validation dataset
Calculates players mAP@0.5 and soccerball recall, precision, center distance
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.evaluator import Evaluator
from rfdetr import RFDETRBase, RFDETRMedium
from PIL import Image


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotation file"""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def map_category_to_class(category_name: str) -> int:
    """Map category name to class ID: players=0, ball=1"""
    name_lower = category_name.lower()
    if name_lower in ['players', 'player', 'player1', 'player2', 'referee']:
        return 0  # Player class
    elif name_lower == 'ball':
        return 1  # Ball class
    else:
        return 0  # Default to player


def convert_bbox_to_torch_format(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert from (x, y, width, height) to (x_min, y_min, x_max, y_max)"""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get center point from bbox in (x, y, width, height) format"""
    x, y, w, h = bbox
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    return (center_x, center_y)


def calculate_center_distance(centers1: np.ndarray, centers2: np.ndarray) -> float:
    """
    Calculate average center distance using Hungarian Algorithm
    Args:
        centers1: [N, 2] array of (x, y) centers (predictions)
        centers2: [M, 2] array of (x, y) centers (ground truth)
    Returns:
        Average center distance for matched pairs
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


def main():
    """Main test function"""
    # Configuration
    dataset_dir = Path("data/raw/Validation images OFFICIAL/test")
    annotation_file = dataset_dir / "_annotations.coco.json"
    
    # Use RF-DETR Medium (better accuracy) or Base (faster)
    # Both are pre-trained on COCO and Apache 2.0 licensed
    model_size = os.getenv("RF_DETR_SIZE", "medium").lower()
    
    print("=" * 60)
    print("RF-DETR Validation Test")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Model: RF-DETR-{model_size.capitalize()} (pre-trained on COCO)")
    print("License: Apache 2.0 (commercial-friendly)")
    print()
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(str(annotation_file))
    
    # Build mappings
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    print(f"Loaded {len(images)} images")
    print(f"Categories: {list(categories.values())}")
    print()
    
    # Initialize RF-DETR model (pre-trained on COCO)
    print("Initializing RF-DETR model (pre-trained on COCO)...")
    if model_size == "base":
        model = RFDETRBase()
    else:
        model = RFDETRMedium()  # Default to Medium for better accuracy
    print("Model loaded successfully")
    print()
    
    # Initialize evaluator
    evaluator = Evaluator({'iou_thresholds': [0.5], 'max_detections': 100})
    
    # Process images
    print("Processing images...")
    all_predictions = []
    all_targets = []
    
    # For ball metrics
    ball_centers_pred = []
    ball_centers_gt = []
    ball_tp = 0
    ball_fp = 0
    ball_fn = 0
    
    image_ids = sorted(images.keys())
    
    for idx, image_id in enumerate(image_ids):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_ids)} images...")
        
        image_info = images[image_id]
        image_path = dataset_dir / image_info['file_name']
        
        # Load image
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image as PIL Image (RF-DETR expects PIL Image)
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue
        
        # Run RF-DETR inference
        detections = model.predict(pil_image, threshold=0.5)
        
        # Convert detections to torch format
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        # RF-DETR returns detections with: class_id, confidence, bbox (x_min, y_min, x_max, y_max)
        # COCO classes: 0=person, 1=bicycle, 2=car, ... 37=sports ball
        # We need to map: person -> player (0), sports ball (37) -> ball (1)
        coco_person_id = 0
        coco_sports_ball_id = 37
        
        for i in range(len(detections.class_id)):
            coco_class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            bbox = detections.xyxy[i]  # Already in x_min, y_min, x_max, y_max format
            
            # Map COCO classes to our classes
            if coco_class_id == coco_person_id:
                # Person detected -> map to player
                class_id = 0
            elif coco_class_id == coco_sports_ball_id:
                # Sports ball detected -> map to ball
                class_id = 1
            else:
                # Other COCO classes -> skip for now (or map to player if it's a person-like object)
                continue
            
            pred_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            pred_scores.append(float(confidence))
            pred_labels.append(class_id)
        
        # Convert to tensors
        if len(pred_boxes) == 0:
            pred_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores_tensor = torch.zeros((0,), dtype=torch.float32)
            pred_labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            pred_scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
            pred_labels_tensor = torch.tensor(pred_labels, dtype=torch.int64)
        
        # Get ground truth annotations
        annotations = image_annotations.get(image_id, [])
        
        gt_boxes = []
        gt_labels = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            
            # Map category to class
            cat_id = ann['category_id']
            cat_name = categories.get(cat_id, '')
            class_id = map_category_to_class(cat_name)
            
            gt_boxes.append([x_min, y_min, x_max, y_max])
            gt_labels.append(class_id)
        
        # Convert to tensors
        if len(gt_boxes) == 0:
            gt_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.int64)
        
        # Store for evaluation
        all_predictions.append({
            'boxes': pred_boxes_tensor,
            'scores': pred_scores_tensor,
            'labels': pred_labels_tensor
        })
        
        all_targets.append({
            'boxes': gt_boxes_tensor,
            'labels': gt_labels_tensor
        })
        
        # Extract ball centers for center distance calculation
        ball_pred_centers = []
        ball_gt_centers = []
        
        # Get predicted ball centers
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:  # Ball
                bbox = pred_boxes[i]
                # bbox is in x_min, y_min, x_max, y_max format
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                ball_pred_centers.append((center_x, center_y))
        
        # Get ground truth ball centers
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = categories.get(cat_id, '')
            if map_category_to_class(cat_name) == 1:  # Ball
                bbox = ann['bbox']
                center = get_bbox_center(bbox)
                ball_gt_centers.append(center)
        
        # Store centers for this image
        if len(ball_gt_centers) > 0:
            ball_centers_pred.append(np.array(ball_pred_centers) if ball_pred_centers else np.array([]).reshape(0, 2))
            ball_centers_gt.append(np.array(ball_gt_centers))
    
    print(f"Processed {len(image_ids)} images")
    print()
    
    # Calculate players mAP@0.5
    print("Calculating metrics...")
    metrics = evaluator.evaluate(all_predictions, all_targets)
    player_map_05 = metrics['player_map_05']
    
    # Calculate ball recall and precision
    # We need to recalculate with IoU matching for ball specifically
    ball_tp = 0
    ball_fp = 0
    ball_fn = 0
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        # Filter for ball only
        ball_pred_mask = pred_labels == 1
        ball_target_mask = target_labels == 1
        
        if np.sum(ball_target_mask) == 0:
            # No ground truth balls - all predictions are false positives
            ball_fp += np.sum(ball_pred_mask)
            continue
        
        if np.sum(ball_pred_mask) == 0:
            # No predictions - all targets are false negatives
            ball_fn += np.sum(ball_target_mask)
            continue
        
        # Get ball boxes
        ball_pred_boxes = pred_boxes[ball_pred_mask]
        ball_target_boxes = target_boxes[ball_target_mask]
        ball_pred_scores = pred_scores[ball_pred_mask]
        
        # Compute IoU
        ious = evaluator._compute_ious(ball_pred_boxes, ball_target_boxes)
        
        # Match predictions to targets (greedy matching by score)
        matched_targets = np.zeros(len(ball_target_boxes), dtype=bool)
        sorted_indices = np.argsort(ball_pred_scores)[::-1]
        
        for pred_idx in sorted_indices:
            best_iou = 0.0
            best_target_idx = -1
            
            for target_idx in range(len(ball_target_boxes)):
                if not matched_targets[target_idx]:
                    iou = ious[pred_idx, target_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
            
            if best_iou >= 0.5:  # IoU threshold
                matched_targets[best_target_idx] = True
                ball_tp += 1
            else:
                ball_fp += 1
        
        # Count unmatched targets as false negatives
        ball_fn += np.sum(~matched_targets)
    
    # Calculate ball metrics
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
    
    # Print results
    print()
    print("=" * 60)
    print("RF-DETR Validation Results")
    print("=" * 60)
    print()
    print("Players:")
    print(f"  mAP@0.5: {player_map_05:.4f}")
    print()
    print("Soccerball:")
    print(f"  Recall: {ball_recall:.4f}")
    print(f"  Precision: {ball_precision:.4f}")
    print(f"  Center Distance (avg): {avg_center_distance:.2f} pixels")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
