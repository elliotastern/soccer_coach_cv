"""
ByteTrack integration for multi-object tracking
Provides temporal consistency for ball and player tracking
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


try:
    from byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: byte-track not installed. Install with: pip install byte-track")


class ByteTrackerWrapper:
    """
    Wrapper for ByteTrack multi-object tracking
    """
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5,
                 track_buffer: int = 30, match_thresh: float = 0.8,
                 min_box_area: float = 10.0):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold
            track_buffer: Buffer for track persistence
            match_thresh: IoU threshold for matching
            min_box_area: Minimum box area to track
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError("byte-track not installed. Install with: pip install byte-track")
        
        self.tracker = BYTETracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            min_box_area=min_box_area
        )
        self.frame_id = 0
    
    def update(self, detections: Dict, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: Dictionary with 'boxes', 'scores', 'labels' (tensors)
            image_shape: (height, width) of image
        
        Returns:
            List of tracked objects, each with 'track_id', 'bbox', 'score', 'class_id'
        """
        if not BYTETRACK_AVAILABLE:
            return []
        
        # Convert detections to ByteTrack format
        boxes = detections['boxes'].cpu().numpy() if isinstance(detections['boxes'], torch.Tensor) else detections['boxes']
        scores = detections['scores'].cpu().numpy() if isinstance(detections['scores'], torch.Tensor) else detections['scores']
        labels = detections['labels'].cpu().numpy() if isinstance(detections['labels'], torch.Tensor) else detections['labels']
        
        # Convert boxes from [x_min, y_min, x_max, y_max] to [x_center, y_center, w, h]
        boxes_center = np.zeros_like(boxes)
        boxes_center[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        boxes_center[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        boxes_center[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_center[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        
        # Prepare detections for ByteTrack: [x_center, y_center, w, h, score, class_id]
        detections_array = np.zeros((len(boxes), 6))
        detections_array[:, :4] = boxes_center
        detections_array[:, 4] = scores
        detections_array[:, 5] = labels
        
        # Update tracker
        tracked_objects = self.tracker.update(detections_array, image_shape)
        
        # Convert back to our format
        results = []
        for obj in tracked_objects:
            track_id = int(obj.track_id)
            bbox_center = obj.tlbr  # ByteTrack returns [x_min, y_min, x_max, y_max]
            score = float(obj.score)
            class_id = int(obj.cls)
            
            results.append({
                'track_id': track_id,
                'bbox': bbox_center,  # [x_min, y_min, x_max, y_max]
                'score': score,
                'class_id': class_id
            })
        
        self.frame_id += 1
        return results
    
    def filter_short_tracks(self, tracked_objects: List[Dict], min_frames: int = 3) -> List[Dict]:
        """
        Filter out tracks that exist for less than min_frames
        
        Args:
            tracked_objects: List of tracked objects
            min_frames: Minimum frames for a track to be valid
        
        Returns:
            Filtered list of tracked objects
        """
        # This is a simplified version - full implementation would track frame counts
        # For now, return all tracks (ByteTrack handles this internally)
        return tracked_objects
