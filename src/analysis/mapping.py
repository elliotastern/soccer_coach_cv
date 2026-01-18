# Homography / View Transformer
import numpy as np
import cv2
from typing import Tuple, Optional, List
from src.types import Location


class PitchMapper:
    """Maps pixel coordinates to pitch coordinates using homography"""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 homography_matrix: Optional[np.ndarray] = None):
        """
        Initialize pitch mapper
        
        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            homography_matrix: Pre-computed homography matrix (3x3) or None for manual
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography = homography_matrix
    
    def set_homography_from_points(self, src_points: List[Tuple[float, float]],
                                   dst_points: List[Tuple[float, float]]):
        """
        Compute homography from point correspondences
        
        Args:
            src_points: List of (x, y) pixel coordinates
            dst_points: List of (x, y) pitch coordinates (meters)
        """
        if len(src_points) != len(dst_points) or len(src_points) < 4:
            raise ValueError("Need at least 4 point correspondences")
        
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        self.homography, _ = cv2.findHomography(src_pts, dst_pts)
    
    def pixel_to_pitch(self, x_pixel: float, y_pixel: float) -> Location:
        """
        Convert pixel coordinates to pitch coordinates
        
        Args:
            x_pixel: X coordinate in pixels
            y_pixel: Y coordinate in pixels
        
        Returns:
            Location in pitch coordinates (meters)
        """
        if self.homography is None:
            # Fallback: simple scaling (assumes top-down view)
            # This is a placeholder - should use proper homography
            x_pitch = (x_pixel / 1920.0) * self.pitch_length - self.pitch_length / 2
            y_pitch = (y_pixel / 1080.0) * self.pitch_width - self.pitch_width / 2
            return Location(x=x_pitch, y=y_pitch)
        
        # Apply homography
        point = np.array([[x_pixel, y_pixel]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.homography)
        x_pitch, y_pitch = transformed[0][0]
        
        return Location(x=x_pitch, y=y_pitch)
    
    def bbox_center_to_pitch(self, bbox: Tuple[float, float, float, float]) -> Location:
        """
        Convert bounding box center to pitch coordinates
        
        Args:
            bbox: (x, y, width, height) in pixels
        
        Returns:
            Location in pitch coordinates
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        return self.pixel_to_pitch(center_x, center_y)
