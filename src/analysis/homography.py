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
    # With more points, use tighter threshold for better accuracy
    # Since camera is static, we can use many stable reference points
    if len(image_points) > 20:
        reproj_threshold = 2.0  # Tighter threshold for many points
    elif len(image_points) > 10:
        reproj_threshold = 3.0
    else:
        reproj_threshold = 5.0
    
    # RANSAC with high confidence for robust estimation
    # More points = better accuracy, especially for y-axis
    H, mask = cv2.findHomography(image_points, pitch_points, 
                                 method=cv2.RANSAC, 
                                 ransacReprojThreshold=reproj_threshold,
                                 maxIters=3000,  # More iterations for better results with many points
                                 confidence=0.995)  # High confidence for robust estimation
    
    # Log how many inliers were used (for debugging)
    if mask is not None and len(image_points) > 15:
        inlier_count = np.sum(mask)
        print(f"   📍 Using {inlier_count}/{len(image_points)} reference points for homography (RANSAC inliers)")
    
    return H


def estimate_homography_auto(image: np.ndarray, pitch_length: float = 105.0, 
                            pitch_width: float = 68.0, 
                            correct_distortion: bool = True) -> Optional[np.ndarray]:
    """
    Automatically estimate homography from image using enhanced keypoint detection.
    Optionally corrects lens distortion (fisheye) before estimation for accurate y-axis mapping.
    
    Args:
        image: Input image
        pitch_width: Standard pitch width in meters
        pitch_height: Standard pitch height in meters
        correct_distortion: If True, attempt to correct lens distortion before homography
    
    Returns:
        Homography matrix [3, 3] or None if estimation fails
    """
    try:
        # Try enhanced keypoint detection first
        from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto
        from src.analysis.undistortion import (
            estimate_camera_from_landmarks, 
            undistort_image,
            detect_fisheye_distortion,
            DistortionParams
        )
        
        # Detect keypoints on original (potentially distorted) image
        keypoint_data = detect_pitch_keypoints_auto(
            image, 
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            min_points=4,
            max_points=40  # Increased to include more touchline points for y-axis accuracy
        )
        
        if keypoint_data is not None:
            image_points = np.array(keypoint_data['image_points'], dtype=np.float32)
            pitch_points = np.array(keypoint_data['pitch_points'], dtype=np.float32)
            
            # Attempt distortion correction if enabled
            distortion_params = None
            if correct_distortion and len(image_points) >= 6:
                h, w = image.shape[:2]
                
                # Detect if fisheye distortion is present
                is_fisheye = detect_fisheye_distortion(image_points, pitch_points, (w, h))
                
                # Estimate camera parameters from landmarks
                distortion_params = estimate_camera_from_landmarks(
                    image_points,
                    pitch_points,
                    (w, h),
                    is_fisheye=is_fisheye
                )
                
                if distortion_params and distortion_params.confidence > 0.3:
                    # Undistort image and re-detect keypoints for better accuracy
                    undistorted_image = undistort_image(image, distortion_params)
                    
                    # Re-detect keypoints on undistorted image
                    keypoint_data_undistorted = detect_pitch_keypoints_auto(
                        undistorted_image,
                        pitch_length=pitch_length,
                        pitch_width=pitch_width,
                        min_points=4,
                        max_points=50  # Increased to use all available stable reference points
                    )
                    
                    if keypoint_data_undistorted is not None:
                        # Use undistorted keypoints
                        image_points = np.array(keypoint_data_undistorted['image_points'], dtype=np.float32)
                        pitch_points = np.array(keypoint_data_undistorted['pitch_points'], dtype=np.float32)
                        image = undistorted_image  # Use undistorted image for final homography
            
            # Estimate homography with detected keypoints (now potentially undistorted)
            H = estimate_homography_manual(image_points, pitch_points)
            
            # Note: Center circle calibration is now handled in HomographyEstimator.estimate()
            # to properly store the y_axis_scale factor
            
            return H
    except ImportError:
        # Fallback to basic detection if enhanced detector not available
        pass
    
    # Fallback to basic keypoint detection
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


def apply_homography_vectorized(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography to multiple points (vectorized for speed).
    
    Args:
        points: (N, 2) array of (u, v) pixel coordinates
        H: 3x3 Homography Matrix
    
    Returns:
        (N, 2) array of (x, y) pitch coordinates
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to homogeneous coords
    points_h = np.hstack([points, np.ones((len(points), 1))])
    
    # Matrix multiplication
    transformed = (H @ points_h.T).T
    
    # Normalize by the scaling factor w
    w = transformed[:, 2]
    # Avoid division by zero
    w[np.abs(w) < 1e-10] = 1e-10
    
    x = transformed[:, 0] / w
    y = transformed[:, 1] / w
    
    return np.column_stack([x, y])


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
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize homography estimator
        
        Args:
            pitch_length: Standard pitch length in meters (105.0)
            pitch_width: Standard pitch width in meters (68.0)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography = None
        self.y_axis_distortion_detected = False
        self.y_axis_error_ratio = 1.0
        self.center_circle_detected = False
        self.center_circle_radius_px = 0.0
        self.y_axis_scale = 1.0  # Scale factor for y-axis correction (1.0 = no correction)
        self.center_circle_detected = False
        self.center_circle_radius_px = 0.0
    
    def estimate(self, image: np.ndarray, manual_points: Optional[Dict] = None, 
                 use_auto_detection: bool = True, correct_distortion: bool = True) -> bool:
        """
        Estimate homography from image
        
        Args:
            image: Input image
            manual_points: Optional manual point correspondences
                          {'image_points': [[x, y], ...], 'pitch_points': [[x, y], ...]}
            use_auto_detection: If True and manual_points not provided, use automatic keypoint detection
            correct_distortion: If True, attempt to correct lens distortion (fisheye) for y-axis accuracy
        
        Returns:
            True if estimation successful, False otherwise
        """
        if manual_points is not None:
            image_points = np.array(manual_points['image_points'], dtype=np.float32)
            pitch_points = np.array(manual_points['pitch_points'], dtype=np.float32)
            self.homography = estimate_homography_manual(image_points, pitch_points)
        elif use_auto_detection:
            # Use enhanced automatic keypoint detection with optional distortion correction
            self.homography = estimate_homography_auto(
                image, 
                self.pitch_length, 
                self.pitch_width,
                correct_distortion=correct_distortion
            )
        else:
            # Fallback to basic detection
            self.homography = estimate_homography_auto(
                image, 
                self.pitch_length, 
                self.pitch_width,
                correct_distortion=correct_distortion
            )
        
        # Validate homography and check for y-axis distortion issues
        if self.homography is not None:
            self._validate_y_axis_accuracy(image, manual_points)
            
            # Refine y-axis using center circle calibration (for stationary camera)
            if correct_distortion and use_auto_detection:
                try:
                    from src.analysis.y_axis_calibration import (
                        refine_homography_with_center_circle,
                        calibrate_y_axis_from_field_width
                    )
                    from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto
                    
                    # First, do center circle calibration
                    result = refine_homography_with_center_circle(self.homography, image)
                    
                    # Handle both old (2-tuple) and new (3-tuple) return signatures
                    if len(result) == 3:
                        refined_homography, center_circle, y_axis_scale = result
                        
                        # Use refined homography (directly refined using center circle constraint)
                        self.homography = refined_homography
                    else:
                        # Old signature - unpack 2 values
                        self.homography, center_circle = result
                        y_axis_scale = None
                    
                    if center_circle is not None:
                        self.center_circle_detected = True
                        self.center_circle_radius_px = center_circle.radius
                    else:
                        self.center_circle_detected = False
                    
                    # Now try to get field width calibration from touchline points
                    # Re-detect keypoints to get touchline y-coordinates
                    keypoint_data = detect_pitch_keypoints_auto(
                        image,
                        pitch_length=self.pitch_length,
                        pitch_width=self.pitch_width,
                        min_points=4,
                        max_points=50
                    )
                    
                    player_y_coords = None
                    if keypoint_data is not None and 'keypoints' in keypoint_data:
                        # Extract touchline points - they already have pitch coordinates
                        touchline_y_coords = []
                        for kp in keypoint_data['keypoints']:
                            if kp.landmark_type == 'touchline':
                                # Use the pitch y-coordinate directly (already in pitch space)
                                touchline_y_coords.append(kp.pitch_point[1])
                        
                        if len(touchline_y_coords) >= 4:
                            player_y_coords = touchline_y_coords
                    
                    # Enhance center circle calibration with field width if available
                    if player_y_coords is not None and center_circle is not None:
                        from src.analysis.y_axis_calibration import calibrate_y_axis_from_center_circle
                        # Re-calibrate with field width constraints
                        enhanced_scale = calibrate_y_axis_from_center_circle(
                            self.homography,
                            center_circle,
                            known_radius_m=9.15,
                            player_y_coords=player_y_coords,
                            expected_width=self.pitch_width
                        )
                        if enhanced_scale is not None:
                            y_axis_scale = enhanced_scale
                            print(f"   ✅ Enhanced calibration with field width: scale={y_axis_scale:.3f}")
                    
                    # Store y-axis scale factor for post-transform application
                    # This provides additional correction if direct refinement wasn't perfect
                    if y_axis_scale is not None:
                        self.y_axis_scale = y_axis_scale
                    else:
                        self.y_axis_scale = 1.0  # No correction needed
                except ImportError:
                    self.center_circle_detected = False
                    self.y_axis_scale = 1.0
                except Exception as e:
                    # Fallback if calibration fails
                    print(f"   ⚠️  Center circle refinement failed: {e}")
                    self.center_circle_detected = False
                    self.y_axis_scale = 1.0
        
        return self.homography is not None
    
    def _validate_y_axis_accuracy(self, image: np.ndarray, manual_points: Optional[Dict] = None):
        """
        Validate homography accuracy, especially on y-axis.
        Detects potential fisheye distortion issues.
        
        Args:
            image: Input image
            manual_points: Manual points if available for validation
        """
        if self.homography is None:
            return
        
        # If we have manual points, calculate separate x/y errors
        if manual_points is not None:
            image_points = np.array(manual_points['image_points'], dtype=np.float32)
            pitch_points = np.array(manual_points['pitch_points'], dtype=np.float32)
            
            # Transform points
            points_h = np.hstack([image_points, np.ones((len(image_points), 1))])
            transformed = (self.homography @ points_h.T).T
            w = transformed[:, 2]
            w[np.abs(w) < 1e-10] = 1e-10
            predicted = transformed[:, :2] / w[:, np.newaxis]
            
            # Calculate separate x and y errors
            errors = predicted - pitch_points
            x_errors = np.abs(errors[:, 0])
            y_errors = np.abs(errors[:, 1])
            
            mean_x_error = np.mean(x_errors)
            mean_y_error = np.mean(y_errors)
            max_x_error = np.max(x_errors)
            max_y_error = np.max(y_errors)
            
            # Check if y-axis errors are significantly larger
            y_x_ratio = mean_y_error / (mean_x_error + 1e-6)
            
            if y_x_ratio > 1.5:  # Y-axis error is 50%+ larger than x-axis
                # This suggests y-axis distortion issues
                # Store this info for potential correction
                self.y_axis_distortion_detected = True
                self.y_axis_error_ratio = y_x_ratio
            else:
                self.y_axis_distortion_detected = False
                self.y_axis_error_ratio = y_x_ratio
    
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
