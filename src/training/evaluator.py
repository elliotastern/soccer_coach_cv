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
            Dictionary with metrics including per-class metrics
        """
        # Track overall metrics
        total_tp = 0
        total_predictions = 0
        total_targets = 0
        
        # Track per-class metrics (0=player, 1=ball)
        class_tp = {0: 0, 1: 0}  # True positives per class
        class_predictions = {0: 0, 1: 0}  # Total predictions per class
        class_targets = {0: 0, 1: 0}  # Total targets per class
        
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
            
            # Compute IoU for each predicted box
            if len(pred_boxes) > 0:
                ious = self._compute_ious(pred_boxes, target_boxes)
                
                # Match predictions to targets
                matched = np.zeros(len(target_boxes), dtype=bool)
                tp = 0
                
                # Sort predictions by score
                sorted_indices = np.argsort(pred_scores)[::-1]
                
                for pred_idx in sorted_indices[:self.max_detections]:
                    # Background is already filtered in model inference
                    if pred_labels[pred_idx] < 0 or pred_labels[pred_idx] > 1:
                        continue
                    
                    pred_class = int(pred_labels[pred_idx])
                    class_predictions[pred_class] += 1
                    
                    # Find best matching target
                    best_iou = 0.0
                    best_target_idx = -1
                    
                    for target_idx, target_label in enumerate(target_labels):
                        if not matched[target_idx] and target_label == pred_labels[pred_idx]:
                            iou = ious[pred_idx, target_idx]
                            if iou > best_iou:
                                best_iou = iou
                                best_target_idx = target_idx
                    
                    # Check if match is good enough
                    if best_iou >= 0.5:  # IoU threshold
                        matched[best_target_idx] = True
                        tp += 1
                        class_tp[pred_class] += 1
                
                # Accumulate overall metrics
                num_targets = len(target_boxes)
                num_predictions = min(len([p for p in pred_labels if 0 <= p <= 1]), self.max_detections)
                
                total_tp += tp
                total_predictions += num_predictions
                total_targets += num_targets
                
                # Accumulate per-class targets
                for target_label in target_labels:
                    if 0 <= target_label <= 1:
                        class_targets[int(target_label)] += 1
                
                num_images += 1
        
        # Compute overall metrics
        if num_images == 0:
            return {
                'map': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'player_map': 0.0, 'player_precision': 0.0, 'player_recall': 0.0, 'player_f1': 0.0,
                'ball_map': 0.0, 'ball_precision': 0.0, 'ball_recall': 0.0, 'ball_f1': 0.0
            }
        
        precision = total_tp / total_predictions if total_predictions > 0 else 0.0
        recall = total_tp / total_targets if total_targets > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        map_score = f1
        
        # Compute per-class metrics
        metrics = {
            'map': map_score,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Player metrics (class 0)
        player_precision = class_tp[0] / class_predictions[0] if class_predictions[0] > 0 else 0.0
        player_recall = class_tp[0] / class_targets[0] if class_targets[0] > 0 else 0.0
        player_f1 = 2 * (player_precision * player_recall) / (player_precision + player_recall) if (player_precision + player_recall) > 0 else 0.0
        
        metrics.update({
            'player_map': player_f1,
            'player_precision': player_precision,
            'player_recall': player_recall,
            'player_f1': player_f1
        })
        
        # Ball metrics (class 1)
        ball_precision = class_tp[1] / class_predictions[1] if class_predictions[1] > 0 else 0.0
        ball_recall = class_tp[1] / class_targets[1] if class_targets[1] > 0 else 0.0
        ball_f1 = 2 * (ball_precision * ball_recall) / (ball_precision + ball_recall) if (ball_precision + ball_recall) > 0 else 0.0
        
        metrics.update({
            'ball_map': ball_f1,
            'ball_precision': ball_precision,
            'ball_recall': ball_recall,
            'ball_f1': ball_f1
        })
        
        return metrics
    
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
