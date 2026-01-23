"""
Data Augmentation for DETR Training
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, Tuple, List, Optional
from torch.utils.data import Dataset
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


class GaussianBlurAugmentation:
    """
    Gaussian blur augmentation for sensor noise simulation
    Helps model learn to detect objects in slightly blurred images
    """
    def __init__(self, prob: float = 0.3, kernel_size_range: Tuple[int, int] = (3, 7), 
                 sigma_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Args:
            prob: Probability of applying Gaussian blur
            kernel_size_range: Range of kernel sizes (must be odd)
            sigma_range: Range of sigma values for blur
        """
        self.prob = prob
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Generate random kernel size (must be odd)
        kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate random sigma
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # Apply Gaussian blur
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
        return image, target


class ISONoiseAugmentation:
    """
    ISO noise injection for camera sensor noise simulation
    Simulates high ISO noise in low-light conditions
    """
    def __init__(self, prob: float = 0.3, noise_level: Tuple[int, int] = (5, 25)):
        """
        Args:
            prob: Probability of applying ISO noise
            noise_level: Range of noise levels (higher = more noise)
        """
        self.prob = prob
        self.noise_level = noise_level
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # Convert PIL to numpy
        img_np = np.array(image, dtype=np.float32)
        
        # Generate random noise level
        noise_level = random.randint(self.noise_level[0], self.noise_level[1])
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.float32)
        
        # Add noise
        img_np = img_np + noise
        
        # Clip to valid range
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        image = Image.fromarray(img_np)
        
        return image, target


class JPEGCompressionAugmentation:
    """
    JPEG compression artifacts simulation
    Simulates broadcast video compression artifacts
    """
    def __init__(self, prob: float = 0.2, quality_range: Tuple[int, int] = (60, 95)):
        """
        Args:
            prob: Probability of applying JPEG compression
            quality_range: Range of JPEG quality (lower = more compression)
        """
        self.prob = prob
        self.quality_range = quality_range
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Generate random quality
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        
        # Encode and decode with JPEG compression
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        image = Image.open(buffer)
        image = image.convert('RGB')
        
        return image, target


class MixUpAugmentation:
    """
    MixUp augmentation for occlusion handling
    Blends two images together to simulate partial occlusion
    """
    def __init__(self, prob: float = 0.3, alpha: float = 0.2, dataset=None):
        """
        Args:
            prob: Probability of applying MixUp
            alpha: MixUp mixing parameter (beta distribution parameter)
            dataset: Optional dataset to sample from for mixing
        """
        self.prob = prob
        self.alpha = alpha
        self.dataset = dataset
    
    def __call__(self, image, target):
        if random.random() > self.prob or self.dataset is None:
            return image, target
        
        # Sample another image from dataset
        try:
            idx = random.randint(0, len(self.dataset) - 1)
            mix_image, mix_target = self.dataset[idx]
            
            # Convert to numpy if needed
            if isinstance(image, Image.Image):
                img1 = np.array(image, dtype=np.float32)
            else:
                img1 = np.array(image, dtype=np.float32)
            
            if isinstance(mix_image, Image.Image):
                img2 = np.array(mix_image, dtype=np.float32)
            else:
                img2 = np.array(mix_image, dtype=np.float32)
            
            # Resize mix_image to match image size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Sample lambda from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Mix images
            mixed_img = lam * img1 + (1 - lam) * img2
            mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
            
            # Mix targets (keep both sets of boxes)
            if 'boxes' in target and 'boxes' in mix_target and len(mix_target['boxes']) > 0:
                # Combine boxes and labels
                boxes1 = target['boxes']
                boxes2 = mix_target['boxes']
                labels1 = target['labels']
                labels2 = mix_target['labels']
                
                # Scale boxes2 to match image size if needed
                if img1.shape != img2.shape:
                    scale_x = img1.shape[1] / img2.shape[1]
                    scale_y = img1.shape[0] / img2.shape[0]
                    boxes2 = boxes2.clone()
                    boxes2[:, [0, 2]] *= scale_x
                    boxes2[:, [1, 3]] *= scale_y
                
                # Combine boxes and labels
                combined_boxes = torch.cat([boxes1, boxes2])
                combined_labels = torch.cat([labels1, labels2])
                
                target['boxes'] = combined_boxes
                target['labels'] = combined_labels
            
            image = Image.fromarray(mixed_img)
            
        except Exception as e:
            # If MixUp fails, return original
            pass
        
        return image, target


class MosaicAugmentation:
    """
    Mosaic augmentation for multi-scale learning
    Combines 4 images into a grid to learn multi-scale features
    """
    def __init__(self, prob: float = 0.5, min_scale: float = 0.4, max_scale: float = 1.0, dataset=None):
        """
        Args:
            prob: Probability of applying Mosaic
            min_scale: Minimum scale for cropped images
            max_scale: Maximum scale for cropped images
            dataset: Optional dataset to sample from
        """
        self.prob = prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.dataset = dataset
    
    def __call__(self, image, target):
        if random.random() > self.prob or self.dataset is None:
            return image, target
        
        try:
            # Sample 3 additional images
            indices = random.sample(range(len(self.dataset)), 3)
            images = [image]
            targets = [target]
            
            for idx in indices:
                img, tgt = self.dataset[idx]
                images.append(img)
                targets.append(tgt)
            
            # Get base image size
            if isinstance(images[0], Image.Image):
                base_w, base_h = images[0].size
            else:
                base_h, base_w = images[0].shape[:2]
            
            # Create mosaic canvas
            canvas = np.zeros((base_h * 2, base_w * 2, 3), dtype=np.uint8)
            canvas_targets = {'boxes': [], 'labels': []}
            
            # Place images in 2x2 grid
            positions = [
                (0, 0),  # Top-left
                (base_w, 0),  # Top-right
                (0, base_h),  # Bottom-left
                (base_w, base_h)  # Bottom-right
            ]
            
            for i, (img, tgt) in enumerate(zip(images, targets)):
                x_offset, y_offset = positions[i]
                
                # Convert to numpy
                if isinstance(img, Image.Image):
                    img_np = np.array(img)
                else:
                    img_np = np.array(img)
                
                # Resize image
                h, w = img_np.shape[:2]
                scale = random.uniform(self.min_scale, self.max_scale)
                new_h, new_w = int(h * scale), int(w * scale)
                img_np = cv2.resize(img_np, (new_w, new_h))
                
                # Place on canvas
                y_end = min(y_offset + new_h, base_h * 2)
                x_end = min(x_offset + new_w, base_w * 2)
                canvas[y_offset:y_end, x_offset:x_end] = img_np[:y_end-y_offset, :x_end-x_offset]
                
                # Adjust target boxes
                if 'boxes' in tgt and len(tgt['boxes']) > 0:
                    boxes = tgt['boxes'].clone()
                    labels = tgt['labels'].clone()
                    
                    # Scale boxes
                    boxes[:, [0, 2]] *= (new_w / w)
                    boxes[:, [1, 3]] *= (new_h / h)
                    
                    # Offset boxes
                    boxes[:, [0, 2]] += x_offset
                    boxes[:, [1, 3]] += y_offset
                    
                    # Clip boxes to canvas
                    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, base_w * 2)
                    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, base_h * 2)
                    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, base_w * 2)
                    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, base_h * 2)
                    
                    # Filter out invalid boxes
                    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    if valid.any():
                        canvas_targets['boxes'].append(boxes[valid])
                        canvas_targets['labels'].append(labels[valid])
            
            # Combine all boxes
            if canvas_targets['boxes']:
                target['boxes'] = torch.cat(canvas_targets['boxes'])
                target['labels'] = torch.cat(canvas_targets['labels'])
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
            # Resize canvas to original size
            canvas = cv2.resize(canvas, (base_w, base_h))
            image = Image.fromarray(canvas)
            
        except Exception as e:
            # If Mosaic fails, return original
            pass
        
        return image, target


def get_train_transforms(aug_config: dict, copy_paste_aug: Optional[CopyPasteAugmentation] = None, 
                         dataset: Optional[Dataset] = None) -> Compose:
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
    
    # Gaussian blur augmentation
    if aug_config.get('gaussian_blur', {}).get('enabled', False):
        gaussian_blur_config = aug_config['gaussian_blur']
        transforms.append(GaussianBlurAugmentation(
            prob=gaussian_blur_config.get('prob', 0.3),
            kernel_size_range=tuple(gaussian_blur_config.get('kernel_size_range', [3, 7])),
            sigma_range=tuple(gaussian_blur_config.get('sigma_range', [0.5, 2.0]))
        ))
    
    # ISO noise augmentation
    if aug_config.get('iso_noise', {}).get('enabled', False):
        iso_noise_config = aug_config['iso_noise']
        transforms.append(ISONoiseAugmentation(
            prob=iso_noise_config.get('prob', 0.3),
            noise_level=tuple(iso_noise_config.get('noise_level', [5, 25]))
        ))
    
    # JPEG compression augmentation
    if aug_config.get('jpeg_compression', {}).get('enabled', False):
        jpeg_config = aug_config['jpeg_compression']
        transforms.append(JPEGCompressionAugmentation(
            prob=jpeg_config.get('prob', 0.2),
            quality_range=tuple(jpeg_config.get('quality_range', [60, 95]))
        ))
    
    # MixUp augmentation (requires dataset)
    if aug_config.get('mixup', {}).get('enabled', False) and dataset is not None:
        mixup_config = aug_config['mixup']
        transforms.append(MixUpAugmentation(
            prob=mixup_config.get('prob', 0.3),
            alpha=mixup_config.get('alpha', 0.2),
            dataset=dataset
        ))
    
    # Mosaic augmentation (requires dataset)
    if aug_config.get('mosaic', {}).get('enabled', False) and dataset is not None:
        mosaic_config = aug_config['mosaic']
        transforms.append(MosaicAugmentation(
            prob=mosaic_config.get('prob', 0.5),
            min_scale=mosaic_config.get('min_scale', 0.4),
            max_scale=mosaic_config.get('max_scale', 1.0),
            dataset=dataset
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
