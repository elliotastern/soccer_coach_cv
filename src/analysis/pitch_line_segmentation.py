"""
Semantic segmentation for pitch line detection.
Uses DeepLabV3 or custom U-Net to segment pitch lines from images.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision import transforms


class PitchLineSegmenter:
    """
    Semantic segmentation model for detecting pitch lines.
    
    Uses torchvision DeepLabV3 (BSD licensed, free for commercial use)
    to output binary masks of pitch lines, which are then used for
    Hough line detection instead of raw image processing.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_type: str = "deeplabv3",
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 use_pretrained: bool = True):
        """
        Initialize pitch line segmenter.
        
        Args:
            model_path: Path to trained model checkpoint (None = use color fallback)
            model_type: "deeplabv3" or "unet"
            device: "cuda", "cpu", or None (auto-detect)
            threshold: Binary threshold for mask (0-1)
            use_pretrained: Use pre-trained weights as starting point
        """
        self.model_path = model_path
        self.model_type = model_type
        self.threshold = threshold
        self.use_pretrained = use_pretrained
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self._model_loaded = False
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self._load_model()
        elif model_path:
            print(f"Warning: Model path {model_path} does not exist. Using color-based fallback.")
        elif model_type == "unet":
            # Create U-Net from scratch (e.g. for training)
            self.model = self._create_unet_model()
            self.model.to(self.device)
            self._model_loaded = True
    
    def _load_model(self):
        """Load segmentation model from checkpoint."""
        try:
            if self.model_type == "deeplabv3":
                self.model = self._create_deeplabv3_model()
            elif self.model_type == "unet":
                self.model = self._create_unet_model()
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            
            # Load checkpoint if available
            if self.model_path:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded segmentation model from {self.model_path}")
            
            self.model.eval()
            self.model.to(self.device)
            self._model_loaded = True
            
        except Exception as e:
            print(f"Failed to load segmentation model: {e}")
            print("Falling back to color-based line detection")
            self.model = None
            self._model_loaded = False
    
    def _create_deeplabv3_model(self) -> nn.Module:
        """Create DeepLabV3 model for binary segmentation."""
        try:
            import torchvision.models.segmentation as segmentation
            
            # Load pre-trained DeepLabV3 with ResNet-50 backbone
            model = segmentation.deeplabv3_resnet50(pretrained=self.use_pretrained)
            
            # Modify final classifier for binary segmentation (1 class: lines)
            # Original: 21 classes (COCO), we need 1 class (lines)
            model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
            
            return model
        except ImportError:
            raise ImportError("torchvision not available. Install with: pip install torchvision")
    
    def _create_unet_model(self) -> nn.Module:
        """Create lightweight U-Net model for binary segmentation with dilated bridge."""
        # U-Net with dilated convolutions in bridge for larger receptive field (hallucination)
        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._conv_block(3, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)
                # Bridge: dilated convolutions (receptive field without losing resolution)
                self.bridge = nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=4, dilation=4),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=8, dilation=8),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                # Decoder
                self.dec4 = self._conv_block(512 + 256, 256)
                self.dec3 = self._conv_block(256 + 128, 128)
                self.dec2 = self._conv_block(128 + 64, 64)
                self.dec1 = nn.Conv2d(64, 1, kernel_size=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                e4 = self.bridge(e4)  # Dilated bridge for global context
                d4 = self.upsample(e4)
                d4 = torch.cat([d4, e3], dim=1)
                d4 = self.dec4(d4)
                d3 = self.upsample(d4)
                d3 = torch.cat([d3, e2], dim=1)
                d3 = self.dec3(d3)
                d2 = self.upsample(d3)
                d2 = torch.cat([d2, e1], dim=1)
                d2 = self.dec2(d2)
                d1 = self.upsample(d2)
                d1 = self.dec1(d1)
                return d1  # Logits (sigmoid applied at inference / BCEWithLogits in training)
        
        return UNet()
    
    def segment_pitch_lines_averaged(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Segment pitch lines by averaging masks across multiple frames.
        Since camera is static, averaging improves stability and reduces noise.
        This is especially helpful for detecting faint off-white elements like center circle.
        
        Args:
            images: List of frames from the same camera position (static camera)
        
        Returns:
            Averaged binary mask (uint8, 0-255) where white pixels are pitch lines
        """
        if not images:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Generate masks for all frames
        masks = []
        for image in images:
            mask = self.segment_pitch_lines(image)
            masks.append(mask.astype(np.float32))
        
        # Average masks (simple mean)
        averaged_mask = np.mean(masks, axis=0)
        
        # Threshold to binary (0 or 255)
        # Use lower threshold for averaged mask since averaging reduces noise
        threshold = 0.3  # Lower than single frame (0.5) because averaging is more stable
        binary_mask = (averaged_mask > threshold * 255).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary_mask
    
    def segment_pitch_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Segment pitch lines from image using semantic segmentation.
        
        Args:
            image: Input image (BGR format, numpy array)
        
        Returns:
            Binary mask (uint8, 0 = background, 255 = line pixel)
        """
        if not self._model_loaded or self.model is None:
            # Fallback to color-based detection
            return self._color_based_fallback(image)
        
        try:
            # Preprocess image
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (maintain aspect ratio, pad if needed)
            target_size = 512  # Common size for segmentation models
            scale = min(target_size / w, target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image_rgb, (new_w, new_h))
            
            # Pad to target_size x target_size
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
            # Convert to tensor and normalize
            tensor = F.to_tensor(padded).unsqueeze(0).to(self.device)
            tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            # Inference
            with torch.no_grad():
                output = self.model(tensor)
                
                # Handle different output formats
                if isinstance(output, dict):
                    output = output['out']
                
                # Get binary mask
                mask = torch.sigmoid(output[0, 0]).cpu().numpy()
                mask = (mask > self.threshold).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Post-processing: morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
            mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
            
            return mask_resized
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return self._color_based_fallback(image)
    
    def _color_based_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Simple approach: mask any shade of white vs everything else.
        Uses multiple color spaces to catch all white/off-white/cream variations.
        """
        # Method 1: Grayscale - simplest approach
        # Any pixel that's bright enough is considered white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Very permissive threshold - catch all shades of white/off-white
        gray_mask = (gray > 100).astype(np.uint8) * 255  # Low threshold to catch all whites
        
        # Method 2: LAB lightness channel - perceptual lightness
        # This catches off-white better than RGB/HSV
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Lightness channel (0-255)
        lab_mask = (l_channel > 120).astype(np.uint8) * 255  # Catch all light colors
        
        # Method 3: HSV - catch white with any hue (low saturation, high value)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Very broad range: any hue, low saturation, high value = white/off-white
        lower_white = np.array([0, 0, 100])  # Very low threshold for value
        upper_white = np.array([180, 100, 255])  # High saturation tolerance
        hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all methods (OR operation) - if ANY method says it's white, it's white
        line_mask = cv2.bitwise_or(gray_mask, lab_mask)
        line_mask = cv2.bitwise_or(line_mask, hsv_mask)
        
        # Apply morphological operations to clean up and connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)  # Connect gaps
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        
        return line_mask
