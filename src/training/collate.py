"""
Custom collate function for DETR training
Handles variable-sized images and formats targets for DETR
"""
import torch
from typing import List, Tuple, Dict


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for DETR
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        images: List of image tensors (DETR expects list, not batched tensor)
        targets: List of target dictionaries
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # DETR expects list of images, not batched tensor
    # Images are already tensors from transforms
    
    return images, targets
