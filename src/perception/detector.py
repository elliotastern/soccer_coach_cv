# RF-DETR wrapper
import os
from typing import List
import numpy as np
from roboflow import Roboflow
from src.types import Detection


class Detector:
    """RF-DETR detector wrapper using Roboflow SDK"""
    
    def __init__(self, model_id: str, api_key: str, confidence_threshold: float = 0.5):
        """
        Initialize detector
        
        Args:
            model_id: Roboflow model ID
            api_key: Roboflow API key
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(model_id)
        self.model = self.project.version(1).model
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame
        
        Args:
            frame: BGR image array
        
        Returns:
            List of Detection objects in COCO JSON format
        """
        predictions = self.model.predict(frame, confidence=self.confidence_threshold)
        
        detections = []
        if hasattr(predictions, 'json'):
            # Roboflow prediction object
            pred_json = predictions.json()
            for pred in pred_json.get('predictions', []):
                bbox = pred.get('bbox', {})
                class_id = pred.get('class_id', 0)
                class_name = pred.get('class', '')
                confidence = pred.get('confidence', 0.0)
                
                detections.append(Detection(
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(bbox.get('x', 0), bbox.get('y', 0), 
                          bbox.get('width', 0), bbox.get('height', 0)),
                    class_name=class_name
                ))
        elif isinstance(predictions, list):
            # List of predictions
            for pred in predictions:
                if isinstance(pred, dict):
                    bbox = pred.get('bbox', {})
                    detections.append(Detection(
                        class_id=pred.get('class_id', 0),
                        confidence=pred.get('confidence', 0.0),
                        bbox=(bbox.get('x', 0), bbox.get('y', 0),
                              bbox.get('width', 0), bbox.get('height', 0)),
                        class_name=pred.get('class', '')
                    ))
        
        return detections
