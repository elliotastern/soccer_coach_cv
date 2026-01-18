"""
Data Augmentation for DETR Training
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, Tuple
from PIL import Image
import random


class Compose:
    """Compose transforms that handle both image and target"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            if 'boxes' in target and len(target['boxes']) > 0:
                width = image.width
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coordinates
                target['boxes'] = boxes
        return image, target


class ColorJitter:
    """Color jitter augmentation"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, target):
        image = self.transform(image)
        return image, target


class ToTensor:
    """Convert PIL Image to tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image and adjust boxes"""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, image, target):
        orig_width, orig_height = image.size
        image = F.resize(image, self.size)
        new_width, new_height = self.size
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_width / orig_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_height / orig_height)
            target['boxes'] = boxes
        
        return image, target


def get_train_transforms(aug_config: dict) -> Compose:
    """
    Get training transforms
    
    Args:
        aug_config: Augmentation configuration dictionary
    
    Returns:
        Compose transform pipeline
    """
    transforms = []
    
    # Resize (DETR uses variable size during training)
    if 'resize_range' in aug_config:
        # For training, we'll use a fixed size for simplicity
        # In practice, DETR uses multi-scale training
        transforms.append(Resize(1333))
    else:
        transforms.append(Resize(1333))
    
    # Random horizontal flip
    if aug_config.get('horizontal_flip', 0) > 0:
        transforms.append(RandomHorizontalFlip(prob=aug_config['horizontal_flip']))
    
    # Color jitter
    if 'color_jitter' in aug_config:
        cj_config = aug_config['color_jitter']
        transforms.append(ColorJitter(
            brightness=cj_config.get('brightness', 0.2),
            contrast=cj_config.get('contrast', 0.2),
            saturation=cj_config.get('saturation', 0.2),
            hue=cj_config.get('hue', 0.1)
        ))
    
    # To tensor and normalize
    transforms.append(ToTensor())
    transforms.append(Normalize())
    
    return Compose(transforms)


def get_val_transforms(aug_config: dict) -> Compose:
    """
    Get validation transforms (no augmentation)
    
    Args:
        aug_config: Augmentation configuration dictionary
    
    Returns:
        Compose transform pipeline
    """
    transforms = []
    
    # Resize
    resize_size = aug_config.get('resize', 1333)
    transforms.append(Resize(resize_size))
    
    # To tensor and normalize
    transforms.append(ToTensor())
    transforms.append(Normalize())
    
    return Compose(transforms)
