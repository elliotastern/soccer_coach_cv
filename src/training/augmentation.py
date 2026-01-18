"""
Data Augmentation for DETR Training
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, Tuple, List, Optional
from PIL import Image
import random
import numpy as np
import cv2


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


class MotionBlurAugmentation:
    """
    Motion blur augmentation for realistic ball detection
    Helps model learn to detect blurred/fast-moving balls
    """
    def __init__(self, prob: float = 0.3, max_kernel_size: int = 15):
        """
        Args:
            prob: Probability of applying motion blur
            max_kernel_size: Maximum motion blur kernel size
        """
        self.prob = prob
        self.max_kernel_size = max_kernel_size
    
    def __call__(self, image, target):
        """
        Apply motion blur to image
        
        Args:
            image: PIL Image
            target: Target dictionary (unchanged)
        
        Returns:
            Blurred image and target
        """
        if random.random() > self.prob:
            return image, target
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Generate random motion blur kernel
        kernel_size = random.randint(3, self.max_kernel_size)
        angle = random.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        # Apply blur
        if len(img_np.shape) == 3:
            img_np = cv2.filter2D(img_np, -1, kernel)
        else:
            img_np = cv2.filter2D(img_np, -1, kernel)
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
        return image, target


class CLAHEAugmentation:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
    Helps with synthetic image fog and improves far-field visibility
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Args:
            clip_limit: Contrast limiting threshold
            tile_grid_size: Size of grid for adaptive equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, image, target):
        """
        Apply CLAHE to image
        
        Args:
            image: PIL Image
            target: Target dictionary (unchanged)
        
        Returns:
            Enhanced image and target
        """
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Convert RGB to LAB color space
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            # Apply CLAHE to L channel only
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            # Convert back to RGB
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            img_np = self.clahe.apply(img_np)
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
        return image, target


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for ball class balancing.
    Extracts ball patches from images and pastes them at random locations.
    """
    def __init__(self, prob=0.5, ball_class_id=2, max_pastes=3):
        """
        Args:
            prob: Probability of applying copy-paste
            ball_class_id: Class ID for ball (1-based: 2=ball)
            max_pastes: Maximum number of balls to paste per image
        """
        self.prob = prob
        self.ball_class_id = ball_class_id
        self.max_pastes = max_pastes
        self.ball_patches = []  # Will be populated by dataset
    
    def set_ball_patches(self, patches: List[Tuple[Image.Image, Dict]]):
        """
        Set ball patches to use for pasting
        
        Args:
            patches: List of (ball_image, bbox_info) tuples
        """
        self.ball_patches = patches
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def __call__(self, image, target):
        """
        Apply copy-paste augmentation
        
        Args:
            image: PIL Image
            target: Target dictionary with 'boxes' and 'labels'
        
        Returns:
            Augmented image and target
        """
        if random.random() > self.prob or len(self.ball_patches) == 0:
            return image, target
        
        if 'boxes' not in target or len(target['boxes']) == 0:
            return image, target
        
        image_np = np.array(image)
        width, height = image.size
        boxes = target['boxes'].clone()
        labels = target['labels'].clone()
        
        # Try to paste balls
        num_pastes = random.randint(1, self.max_pastes)
        new_boxes = []
        new_labels = []
        
        for _ in range(num_pastes):
            # Select random ball patch
            ball_patch, bbox_info = random.choice(self.ball_patches)
            ball_img = np.array(ball_patch)
            ball_h, ball_w = ball_img.shape[:2]
            
            # Find random location that doesn't overlap with existing boxes
            max_attempts = 50
            for attempt in range(max_attempts):
                # Random position
                x_min = random.randint(0, max(1, width - ball_w))
                y_min = random.randint(0, max(1, height - ball_h))
                x_max = x_min + ball_w
                y_max = y_min + ball_h
                
                # Check overlap with existing boxes
                new_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=boxes.dtype)
                overlaps = False
                for existing_box in boxes:
                    iou = self._compute_iou(new_box, existing_box)
                    if iou > 0.1:  # Threshold for overlap
                        overlaps = True
                        break
                
                if not overlaps:
                    # Paste ball
                    if x_max <= width and y_max <= height:
                        image_np[y_min:y_max, x_min:x_max] = ball_img
                        new_boxes.append(new_box)
                        new_labels.append(torch.tensor(self.ball_class_id, dtype=labels.dtype))
                    break
        
        # Update image and target
        image = Image.fromarray(image_np)
        if len(new_boxes) > 0:
            new_boxes_tensor = torch.stack(new_boxes)
            new_labels_tensor = torch.stack(new_labels)
            target['boxes'] = torch.cat([boxes, new_boxes_tensor])
            target['labels'] = torch.cat([labels, new_labels_tensor])
        
        return image, target


def get_train_transforms(aug_config: dict, copy_paste_aug: Optional[CopyPasteAugmentation] = None) -> Compose:
    """
    Get training transforms
    
    Args:
        aug_config: Augmentation configuration dictionary
        copy_paste_aug: Optional Copy-Paste augmentation instance
    
    Returns:
        Compose transform pipeline
    """
    transforms = []
    
    # CLAHE contrast enhancement (applied first, before other transforms)
    if aug_config.get('clahe', {}).get('enabled', False):
        clahe_config = aug_config['clahe']
        transforms.append(CLAHEAugmentation(
            clip_limit=clahe_config.get('clip_limit', 2.0),
            tile_grid_size=tuple(clahe_config.get('tile_grid_size', [8, 8]))
        ))
    
    # Copy-Paste augmentation (applied early, before other transforms)
    if copy_paste_aug is not None:
        transforms.append(copy_paste_aug)
    
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
    
    # Motion blur augmentation
    if aug_config.get('motion_blur', {}).get('enabled', False):
        motion_blur_config = aug_config['motion_blur']
        transforms.append(MotionBlurAugmentation(
            prob=motion_blur_config.get('prob', 0.3),
            max_kernel_size=motion_blur_config.get('max_kernel_size', 15)
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
