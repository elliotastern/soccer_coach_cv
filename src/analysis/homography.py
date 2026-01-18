"""
Homography estimation for Game State Reconstruction
Transforms pixel coordinates to pitch coordinates
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import torch


def detect_pitch_keypoints(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect pitch keypoints (corners, penalty box, center circle)
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Array of keypoint coordinates or None if detection fails
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Detect lines using HoughLinesP
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return None
    
    # Extract line endpoints
    keypoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        keypoints.append([x1, y1])
        keypoints.append([x2, y2])
    
    if len(keypoints) < 4:
        return None
    
    return np.array(keypoints, dtype=np.float32)


def estimate_homography_manual(image_points: np.ndarray, pitch_points: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate homography matrix from manual point correspondences
    
    Args:
        image_points: Points in image coordinates [N, 2]
        pitch_points: Corresponding points in pitch coordinates [N, 2]
    
    Returns:
        Homography matrix [3, 3] or None if estimation fails
    """
    if len(image_points) < 4 or len(pitch_points) < 4:
        return None
    
    if len(image_points) != len(pitch_points):
        return None
    
    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(image_points, pitch_points, 
                                 method=cv2.RANSAC, 
                                 ransacReprojThreshold=5.0)
    
    return H


def estimate_homography_auto(image: np.ndarray, pitch_width: float = 105.0, 
                            pitch_height: float = 68.0) -> Optional[np.ndarray]:
    """
    Automatically estimate homography from image
    
    Args:
        image: Input image
        pitch_width: Standard pitch width in meters
        pitch_height: Standard pitch height in meters
    
    Returns:
        Homography matrix [3, 3] or None if estimation fails
    """
    # Detect keypoints
    keypoints = detect_pitch_keypoints(image)
    if keypoints is None or len(keypoints) < 4:
        return None
    
    # Define standard pitch coordinates (normalized to [0, 1])
    # Pitch corners: top-left, top-right, bottom-right, bottom-left
    pitch_corners = np.array([
        [0, 0],  # Top-left
        [1, 0],  # Top-right
        [1, 1],  # Bottom-right
        [0, 1]   # Bottom-left
    ], dtype=np.float32)
    
    # Try to match keypoints to pitch corners
    # This is simplified - full implementation would use more sophisticated matching
    if len(keypoints) >= 4:
        # Use first 4 keypoints (would need better matching in production)
        image_corners = keypoints[:4]
        
        # Estimate homography
        H = estimate_homography_manual(image_corners, pitch_corners)
        return H
    
    return None


def transform_point(homography: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    """
    Transform a point from image coordinates to pitch coordinates
    
    Args:
        homography: Homography matrix [3, 3]
        point: Point in image coordinates (x, y)
    
    Returns:
        Point in pitch coordinates (x, y)
    """
    x, y = point
    point_homogeneous = np.array([x, y, 1.0])
    transformed = homography @ point_homogeneous
    x_pitch = transformed[0] / transformed[2]
    y_pitch = transformed[1] / transformed[2]
    return (x_pitch, y_pitch)


def transform_boxes(homography: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
    """
    Transform bounding boxes from image coordinates to pitch coordinates
    
    Args:
        homography: Homography matrix [3, 3]
        boxes: Boxes in image coordinates [N, 4] (x_min, y_min, x_max, y_max)
    
    Returns:
        Boxes in pitch coordinates [N, 4]
    """
    if len(boxes) == 0:
        return boxes
    
    boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
    pitch_boxes = np.zeros_like(boxes_np)
    
    for i, box in enumerate(boxes_np):
        x_min, y_min, x_max, y_max = box
        
        # Transform corners
        corners_image = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.float32)
        
        corners_pitch = []
        for corner in corners_image:
            x_p, y_p = transform_point(homography, (corner[0], corner[1]))
            corners_pitch.append([x_p, y_p])
        
        corners_pitch = np.array(corners_pitch)
        
        # Get bounding box in pitch coordinates
        pitch_boxes[i, 0] = corners_pitch[:, 0].min()
        pitch_boxes[i, 1] = corners_pitch[:, 1].min()
        pitch_boxes[i, 2] = corners_pitch[:, 0].max()
        pitch_boxes[i, 3] = corners_pitch[:, 1].max()
    
    return torch.from_numpy(pitch_boxes).to(boxes.device) if isinstance(boxes, torch.Tensor) else pitch_boxes


class HomographyEstimator:
    """
    Homography estimator for pitch coordinate transformation
    """
    def __init__(self, pitch_width: float = 105.0, pitch_height: float = 68.0):
        """
        Initialize homography estimator
        
        Args:
            pitch_width: Standard pitch width in meters
            pitch_height: Standard pitch height in meters
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.homography = None
    
    def estimate(self, image: np.ndarray, manual_points: Optional[Dict] = None) -> bool:
        """
        Estimate homography from image
        
        Args:
            image: Input image
            manual_points: Optional manual point correspondences
                          {'image_points': [[x, y], ...], 'pitch_points': [[x, y], ...]}
        
        Returns:
            True if estimation successful, False otherwise
        """
        if manual_points is not None:
            image_points = np.array(manual_points['image_points'], dtype=np.float32)
            pitch_points = np.array(manual_points['pitch_points'], dtype=np.float32)
            self.homography = estimate_homography_manual(image_points, pitch_points)
        else:
            self.homography = estimate_homography_auto(image, self.pitch_width, self.pitch_height)
        
        return self.homography is not None
    
    def transform(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform point to pitch coordinates
        
        Args:
            point: Point in image coordinates (x, y)
        
        Returns:
            Point in pitch coordinates (x, y) or None if homography not estimated
        """
        if self.homography is None:
            return None
        
        return transform_point(self.homography, point)
    
    def transform_boxes(self, boxes: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Transform boxes to pitch coordinates
        
        Args:
            boxes: Boxes in image coordinates [N, 4]
        
        Returns:
            Boxes in pitch coordinates [N, 4] or None if homography not estimated
        """
        if self.homography is None:
            return None
        
        return transform_boxes(self.homography, boxes)
