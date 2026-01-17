"""
DETR Model Setup for Training
"""
import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from typing import Dict


def get_detr_model(config: Dict) -> nn.Module:
    """
    Get DETR model with specified configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        DETR model
    """
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    
    # Load pre-trained DETR model
    # DETR uses num_classes + 1 (background class)
    model = detr_resnet50(pretrained=pretrained, num_classes=num_classes + 1)
    
    # Modify classifier head if needed
    # The model already has the correct number of classes from initialization
    
    return model
