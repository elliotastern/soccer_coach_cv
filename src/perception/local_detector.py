"""
Local DETR Detector for Inference
Uses trained PyTorch model instead of Roboflow API
"""
import torch
import torchvision.transforms as T
from typing import List
import numpy as np
from PIL import Image
import cv2
from src.types import Detection


class LocalDetector:
    """Local DETR detector using trained PyTorch model"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize local detector
        
        Args:
            model_path: Path to trained model checkpoint
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cuda' or 'cpu'), auto-detect if None
        """
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Import model creation function
        from src.training.model import get_detr_model
        
        # Get model config from checkpoint or use default
        model_config = checkpoint.get('config', {}).get('model', {
            'num_classes': 2,
            'pretrained': False
        })
        
        self.model = get_detr_model(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        # Setup transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame
        
        Args:
            frame: BGR image array (OpenCV format)
        
        Returns:
            List of Detection objects
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform image
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model([image_tensor[0]])
        
        # Process outputs
        detections = []
        output = outputs[0]
        
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        # Filter by confidence and remove background class (class 0 in DETR is background)
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.confidence_threshold and label > 0:
                # Convert from [x_min, y_min, x_max, y_max] to [x, y, width, height]
                x_min, y_min, x_max, y_max = box
                x = float(x_min)
                y = float(y_min)
                width = float(x_max - x_min)
                height = float(y_max - y_min)
                
                # Map label: DETR uses 1-indexed classes, we want 0-indexed (player=0, ball=1)
                class_id = int(label) - 1
                class_name = "player" if class_id == 0 else "ball" if class_id == 1 else "unknown"
                
                detections.append(Detection(
                    class_id=class_id,
                    confidence=float(score),
                    bbox=(x, y, width, height),
                    class_name=class_name
                ))
        
        return detections
