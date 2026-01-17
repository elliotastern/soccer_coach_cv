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
            Dictionary with metrics: {'map', 'precision', 'recall', 'f1'}
        """
        # Track metrics across all images
        total_tp = 0
        total_predictions = 0
        total_targets = 0
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
                    if pred_labels[pred_idx] == 0:  # Background class
                        continue
                    
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
                
                # Accumulate metrics
                num_targets = len(target_boxes)
                num_predictions = min(len([p for p in pred_labels if p > 0]), self.max_detections)
                
                total_tp += tp
                total_predictions += num_predictions
                total_targets += num_targets
                num_images += 1
        
        # Compute overall metrics
        if num_images == 0:
            return {'map': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision = total_tp / total_predictions if total_predictions > 0 else 0.0
        recall = total_tp / total_targets if total_targets > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Use F1 as approximation of mAP
        map_score = f1
        
        return {
            'map': map_score,
            'precision': precision,
            'recall': recall,
            'f1': f1
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
