# ByteTrack logic
from typing import List, Dict
import numpy as np
try:
    from supervision.tracker.byte_tracker.core import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    try:
        from supervision.tracker.byte_tracker import ByteTrack
        BYTETRACK_AVAILABLE = True
    except ImportError:
        BYTETRACK_AVAILABLE = False
        ByteTrack = None

from supervision.detection.core import Detections
from src.types import Detection, TrackedObject


class Tracker:
    """ByteTrack-based multi-object tracker"""
    
    def __init__(self, track_thresh: float = 0.5, high_thresh: float = 0.6,
                 track_buffer: int = 30, match_thresh: float = 0.8, 
                 frame_rate: int = 30):
        """
        Initialize ByteTrack tracker
        
        Args:
            track_thresh: Detection confidence threshold
            high_thresh: High confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for tracks
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTrack not available. Install supervision with: pip install supervision")
        
        self.byte_tracker = ByteTrack(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections
            frame: Current frame (for shape info)
        
        Returns:
            List of tracked objects with IDs
        """
        if not detections:
            return []
        
        # Convert to supervision Detections format
        xyxy = []
        confidence = []
        class_id = []
        
        for det in detections:
            x, y, w, h = det.bbox
            xyxy.append([x, y, x + w, y + h])
            confidence.append(det.confidence)
            class_id.append(det.class_id)
        
        supervision_detections = Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=np.int32)
        )
        
        # Update tracker
        tracks = self.byte_tracker.update_with_detections(supervision_detections)
        
        # Convert back to TrackedObject format
        tracked_objects = []
        for i, track in enumerate(tracks):
            # Match track to original detection by position
            track_xyxy = track.xyxy[0]
            det_idx = None
            min_dist = float('inf')
            
            for j, det in enumerate(detections):
                x, y, w, h = det.bbox
                det_center_x = x + w / 2
                det_center_y = y + h / 2
                track_center_x = (track_xyxy[0] + track_xyxy[2]) / 2
                track_center_y = (track_xyxy[1] + track_xyxy[3]) / 2
                
                dist = np.sqrt((det_center_x - track_center_x)**2 + (det_center_y - track_center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    det_idx = j
            
            if det_idx is not None and min_dist < 50:  # Within 50 pixels
                detection = detections[det_idx]
                tracked_objects.append(TrackedObject(
                    object_id=int(track.tracker_id),
                    detection=detection
                ))
        
        return tracked_objects
    
    def reset(self):
        """Reset tracker state (call on scene cut)"""
        if BYTETRACK_AVAILABLE:
            self.byte_tracker = ByteTrack(
                track_thresh=self.track_thresh,
                high_thresh=self.high_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate
            )
