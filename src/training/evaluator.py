"""
Model Evaluation for DETR
Computes mAP (Mean Average Precision) metrics
"""
import torch
import numpy as np
from typing import List, Dict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator:
    """Evaluator for DETR model"""
    
    def __init__(self, eval_config: Dict):
        """
        Initialize evaluator
        
        Args:
            eval_config: Evaluation configuration
        """
        self.iou_thresholds = eval_config.get('iou_thresholds', [0.5, 0.75])
        self.max_detections = eval_config.get('max_detections', 100)
    
    def evaluate(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        Evaluate predictions against targets
        
        Args:
            predictions: List of prediction dictionaries from model
            targets: List of target dictionaries
        
        Returns:
            Dictionary with metrics including per-class metrics at different IoU thresholds
        """
        # Track metrics at different IoU thresholds
        iou_thresholds = [0.5, 0.75]
        
        # Per-class tracking for each IoU threshold
        class_metrics = {}
        for iou_thresh in iou_thresholds:
            class_metrics[iou_thresh] = {
                'tp': {0: 0, 1: 0},
                'predictions': {0: 0, 1: 0},
                'targets': {0: 0, 1: 0}
            }
        
        # Track overall metrics (using IoU 0.5)
        total_tp = 0
        total_predictions = 0
        total_targets = 0
        
        # Track ball predictions per image (for ball count metric)
        ball_predictions_per_image = []
        images_with_balls = 0
        
        num_images = 0
        
        for pred, target in zip(predictions, targets):
            if len(target['boxes']) == 0:
                continue
            
            # Get predicted boxes and scores
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            
            # Get target boxes and labels
            target_boxes = target['boxes'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()
            
            # Check if image has balls
            has_balls = np.any(target_labels == 1)
            if has_balls:
                images_with_balls += 1
            
            # Compute IoU for each predicted box
            if len(pred_boxes) > 0:
                ious = self._compute_ious(pred_boxes, target_boxes)
                
                # Count ball predictions for this image
                ball_preds_this_image = np.sum(pred_labels == 1)
                if has_balls:
                    ball_predictions_per_image.append(ball_preds_this_image)
                
                # Evaluate at each IoU threshold
                for iou_thresh in iou_thresholds:
                    matched = np.zeros(len(target_boxes), dtype=bool)
                    tp = 0
                    class_tp = {0: 0, 1: 0}
                    
                    # Sort predictions by score
                    sorted_indices = np.argsort(pred_scores)[::-1]
                    
                    for pred_idx in sorted_indices[:self.max_detections]:
                        # Background is already filtered in model inference
                        if pred_labels[pred_idx] < 0 or pred_labels[pred_idx] > 1:
                            continue
                        
                        pred_class = int(pred_labels[pred_idx])
                        class_metrics[iou_thresh]['predictions'][pred_class] += 1
                        
                        # Find best matching target
                        best_iou = 0.0
                        best_target_idx = -1
                        
                        for target_idx, target_label in enumerate(target_labels):
                            if not matched[target_idx] and target_label == pred_labels[pred_idx]:
                                iou = ious[pred_idx, target_idx]
                                if iou > best_iou:
                                    best_iou = iou
                                    best_target_idx = target_idx
                        
                        # Check if match is good enough for this IoU threshold
                        if best_iou >= iou_thresh:
                            matched[best_target_idx] = True
                            tp += 1
                            class_tp[pred_class] += 1
                    
                    # Accumulate per-class metrics for this IoU threshold
                    for cls in [0, 1]:
                        class_metrics[iou_thresh]['tp'][cls] += class_tp[cls]
                    
                    # Track targets for all IoU thresholds (targets are the same, only matches differ)
                    for target_label in target_labels:
                        if 0 <= target_label <= 1:
                            class_metrics[iou_thresh]['targets'][int(target_label)] += 1
                    
                    # For IoU 0.5, also track overall metrics
                    if iou_thresh == 0.5:
                        num_targets = len(target_boxes)
                        num_predictions = min(len([p for p in pred_labels if 0 <= p <= 1]), self.max_detections)
                        
                        total_tp += tp
                        total_predictions += num_predictions
                        total_targets += num_targets
                
                num_images += 1
        
        # Compute overall metrics
        if num_images == 0:
            return self._empty_metrics()
        
        # Metrics at IoU 0.5 (primary metrics)
        metrics_05 = class_metrics[0.5]
        precision = total_tp / total_predictions if total_predictions > 0 else 0.0
        recall = total_tp / total_targets if total_targets > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Player metrics at IoU 0.5
        player_precision_05 = metrics_05['tp'][0] / metrics_05['predictions'][0] if metrics_05['predictions'][0] > 0 else 0.0
        player_recall_05 = metrics_05['tp'][0] / metrics_05['targets'][0] if metrics_05['targets'][0] > 0 else 0.0
        player_map_05 = self._calculate_map(metrics_05['tp'][0], metrics_05['predictions'][0], metrics_05['targets'][0])
        
        # Ball metrics at IoU 0.5
        ball_precision_05 = metrics_05['tp'][1] / metrics_05['predictions'][1] if metrics_05['predictions'][1] > 0 else 0.0
        ball_recall_05 = metrics_05['tp'][1] / metrics_05['targets'][1] if metrics_05['targets'][1] > 0 else 0.0
        ball_map_05 = self._calculate_map(metrics_05['tp'][1], metrics_05['predictions'][1], metrics_05['targets'][1])
        
        # Metrics at IoU 0.75
        metrics_75 = class_metrics[0.75]
        player_map_75 = self._calculate_map(metrics_75['tp'][0], metrics_75['predictions'][0], metrics_75['targets'][0])
        ball_map_75 = self._calculate_map(metrics_75['tp'][1], metrics_75['predictions'][1], metrics_75['targets'][1])
        
        # Ball count metric (average predictions per image with balls)
        avg_ball_predictions = np.mean(ball_predictions_per_image) if ball_predictions_per_image else 0.0
        
        # Build metrics dictionary
        metrics = {
            'map': f1,  # Overall mAP (using F1 as approximation)
            'precision': precision,
            'recall': recall,
            'f1': f1,
            
            # Player metrics at IoU 0.5
            'player_map_05': player_map_05,
            'player_precision_05': player_precision_05,
            'player_recall_05': player_recall_05,
            'player_f1': 2 * (player_precision_05 * player_recall_05) / (player_precision_05 + player_recall_05) if (player_precision_05 + player_recall_05) > 0 else 0.0,
            
            # Player metrics at IoU 0.75
            'player_map_75': player_map_75,
            
            # Ball metrics at IoU 0.5
            'ball_map_05': ball_map_05,
            'ball_precision_05': ball_precision_05,
            'ball_recall_05': ball_recall_05,
            'ball_f1': 2 * (ball_precision_05 * ball_recall_05) / (ball_precision_05 + ball_recall_05) if (ball_precision_05 + ball_recall_05) > 0 else 0.0,
            
            # Ball metrics at IoU 0.75
            'ball_map_75': ball_map_75,
            
            # Ball count metric
            'ball_avg_predictions_per_image': avg_ball_predictions,
            'images_with_balls': images_with_balls,
            
            # Legacy metrics (for backward compatibility)
            'player_map': player_map_05,
            'player_precision': player_precision_05,
            'player_recall': player_recall_05,
            'ball_map': ball_map_05,
            'ball_precision': ball_precision_05,
            'ball_recall': ball_recall_05,
        }
        
        return metrics
    
    def _calculate_map(self, tp: int, predictions: int, targets: int) -> float:
        """
        Calculate mAP (Mean Average Precision) for a class
        
        Args:
            tp: True positives
            predictions: Total predictions
            targets: Total targets
        
        Returns:
            mAP score (precision if targets > 0, else 0)
        """
        if targets == 0:
            return 0.0
        if predictions == 0:
            return 0.0
        
        precision = tp / predictions if predictions > 0 else 0.0
        recall = tp / targets if targets > 0 else 0.0
        
        # mAP is precision when we have matches
        # For proper mAP, we'd need to compute precision at different recall levels
        # This is a simplified version
        return precision * recall if (precision + recall) > 0 else 0.0
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'map': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'player_map_05': 0.0, 'player_precision_05': 0.0, 'player_recall_05': 0.0, 'player_f1': 0.0,
            'player_map_75': 0.0,
            'ball_map_05': 0.0, 'ball_precision_05': 0.0, 'ball_recall_05': 0.0, 'ball_f1': 0.0,
            'ball_map_75': 0.0,
            'ball_avg_predictions_per_image': 0.0, 'images_with_balls': 0,
            'player_map': 0.0, 'player_precision': 0.0, 'player_recall': 0.0,
            'ball_map': 0.0, 'ball_precision': 0.0, 'ball_recall': 0.0
        }
    
    def _compute_ious(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of boxes
        
        Args:
            boxes1: [N, 4] array of boxes (x_min, y_min, x_max, y_max)
            boxes2: [M, 4] array of boxes
        
        Returns:
            [N, M] array of IoU values
        """
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
