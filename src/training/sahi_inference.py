"""
SAHI (Slicing Aided Hyper Inference) for small object detection
Improves detection of small objects like balls by processing image slices
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
import torchvision.ops as ops


def slice_image(image: Image.Image, slice_height: int = 640, slice_width: int = 640, 
                overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2) -> List[Tuple[Image.Image, Tuple[int, int]]]:
    """
    Slice image into overlapping patches
    
    Args:
        image: PIL Image to slice
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Overlap ratio for height (0.2 = 20% overlap)
        overlap_width_ratio: Overlap ratio for width (0.2 = 20% overlap)
    
    Returns:
        List of (slice_image, (x_offset, y_offset)) tuples
    """
    width, height = image.size
    slices = []
    
    # Calculate step sizes
    step_height = int(slice_height * (1 - overlap_height_ratio))
    step_width = int(slice_width * (1 - overlap_width_ratio))
    
    # Generate slices
    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculate slice bounds
            x_end = min(x + slice_width, width)
            y_end = min(y + slice_height, height)
            
            # Adjust if slice would be smaller than expected
            if x_end - x < slice_width:
                x = max(0, x_end - slice_width)
            if y_end - y < slice_height:
                y = max(0, y_end - slice_height)
            
            # Crop slice
            slice_img = image.crop((x, y, x_end, y_end))
            slices.append((slice_img, (x, y)))
            
            x += step_width
            if x >= width:
                break
        y += step_height
        if y >= height:
            break
    
    return slices


def merge_predictions(predictions: List[Dict], image_size: Tuple[int, int], 
                     iou_threshold: float = 0.5) -> Dict:
    """
    Merge predictions from multiple slices using NMS
    
    Args:
        predictions: List of prediction dicts from slices, each with 'boxes', 'scores', 'labels'
        image_size: (width, height) of original image
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Merged prediction dictionary
    """
    if len(predictions) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64)
        }
    
    # Collect all boxes, scores, labels
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for pred in predictions:
        if len(pred['boxes']) > 0:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_labels.append(pred['labels'])
    
    if len(all_boxes) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64)
        }
    
    # Concatenate all predictions
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Clip boxes to image bounds
    width, height = image_size
    all_boxes[:, 0] = torch.clamp(all_boxes[:, 0], 0, width)
    all_boxes[:, 1] = torch.clamp(all_boxes[:, 1], 0, height)
    all_boxes[:, 2] = torch.clamp(all_boxes[:, 2], 0, width)
    all_boxes[:, 3] = torch.clamp(all_boxes[:, 3], 0, height)
    
    # Apply NMS per class
    unique_labels = torch.unique(all_labels)
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() == 0:
            continue
        
        boxes_class = all_boxes[mask]
        scores_class = all_scores[mask]
        
        # Apply NMS
        keep = ops.nms(boxes_class, scores_class, iou_threshold)
        
        merged_boxes.append(boxes_class[keep])
        merged_scores.append(scores_class[keep])
        merged_labels.append(all_labels[mask][keep])
    
    if len(merged_boxes) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64)
        }
    
    # Concatenate results
    return {
        'boxes': torch.cat(merged_boxes, dim=0),
        'scores': torch.cat(merged_scores, dim=0),
        'labels': torch.cat(merged_labels, dim=0)
    }


def sahi_predict(model, image: Image.Image, slice_size: int = 640, 
                overlap_ratio: float = 0.2, iou_threshold: float = 0.5,
                device: torch.device = None) -> Dict:
    """
    Run SAHI inference on an image
    
    Args:
        model: DETR model (should accept list of images and return predictions)
        image: PIL Image to process
        slice_size: Size of each slice (square)
        overlap_ratio: Overlap ratio between slices
        iou_threshold: IoU threshold for NMS when merging
        device: Device to run inference on
    
    Returns:
        Merged prediction dictionary
    """
    # Slice image
    slices = slice_image(image, slice_size, slice_size, overlap_ratio, overlap_ratio)
    
    if len(slices) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64)
        }
    
    # Run inference on each slice
    predictions = []
    for slice_img, (x_offset, y_offset) in slices:
        # Convert slice to tensor (model expects list of tensors)
        # This is a simplified version - full implementation would use model's preprocessing
        slice_tensor = torch.from_numpy(np.array(slice_img)).permute(2, 0, 1).float() / 255.0
        
        if device is not None:
            slice_tensor = slice_tensor.to(device)
        
        # Run inference (model expects list of images)
        with torch.no_grad():
            slice_preds = model([slice_tensor])
        
        if len(slice_preds) > 0:
            pred = slice_preds[0]
            # Adjust box coordinates to original image coordinates
            if len(pred['boxes']) > 0:
                pred['boxes'][:, 0] += x_offset
                pred['boxes'][:, 1] += y_offset
                pred['boxes'][:, 2] += x_offset
                pred['boxes'][:, 3] += y_offset
            predictions.append(pred)
    
    # Merge predictions
    image_size = image.size
    merged = merge_predictions(predictions, image_size, iou_threshold)
    
    return merged
