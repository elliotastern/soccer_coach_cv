# Homography / View Transformer
import numpy as np
import cv2
from typing import Tuple, Optional, List
from src.types import Location


class PitchMapper:
    """Maps pixel coordinates to pitch coordinates using homography"""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 homography_matrix: Optional[np.ndarray] = None,
                 y_axis_scale: float = 1.0):
        """
        Initialize pitch mapper
        
        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            homography_matrix: Pre-computed homography matrix (3x3) or None for manual
            y_axis_scale: Y-axis scale factor for correction (1.0 = no correction)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography = homography_matrix
        self.y_axis_scale = y_axis_scale
    
    def set_homography(self, homography_matrix: np.ndarray, y_axis_scale: float = 1.0):
        """
        Set homography matrix directly
        
        Args:
            homography_matrix: 3x3 homography matrix
            y_axis_scale: Y-axis scale factor for correction (default: 1.0)
        """
        self.homography = homography_matrix
        self.y_axis_scale = y_axis_scale
    
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
            # Apply y-axis scale correction even in fallback mode
            y_pitch *= self.y_axis_scale
            return Location(x=x_pitch, y=y_pitch)
        
        # Apply homography
        if self.homography is None:
            raise ValueError("Homography matrix not set")
        
        point = np.array([[x_pixel, y_pixel]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.homography)
        x_pitch, y_pitch = transformed[0][0]
        
        # Apply y-axis scale correction as post-transform
        # This corrects for y-axis compression/expansion detected from center circle
        y_pitch *= self.y_axis_scale
        
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

    def distance_to_center(self, x_pitch: float, y_pitch: float) -> float:
        """
        Calculate distance to center circle accounting for y-axis compression.
        
        This method calculates the Euclidean distance to the center circle (0, 0)
        using corrected y-coordinates that account for y-axis compression.
        
        Args:
            x_pitch: X coordinate in pitch space
            y_pitch: Y coordinate in pitch space (may be compressed)
        
        Returns:
            Distance in meters to center circle (0, 0)
        """
        # Apply y-axis correction if needed
        y_corrected = y_pitch * self.y_axis_scale
        
        # Calculate Euclidean distance to center (0, 0)
        distance = np.sqrt(x_pitch**2 + y_corrected**2)
        
        return distance
