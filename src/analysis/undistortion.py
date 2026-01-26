"""
Camera undistortion module for correcting lens distortion (fisheye/barrel).
Estimates distortion coefficients from detected landmarks and corrects images
before homography estimation for accurate y-axis mapping.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DistortionParams:
    """Camera distortion parameters"""
    camera_matrix: np.ndarray  # 3x3 camera matrix
    dist_coeffs: np.ndarray   # Distortion coefficients
    is_fisheye: bool          # True if fisheye model, False for standard
    confidence: float         # Confidence in estimated parameters (0-1)


def estimate_camera_from_landmarks(
    image_points: np.ndarray,
    world_points: np.ndarray,
    image_size: Tuple[int, int],
    is_fisheye: bool = False
) -> Optional[DistortionParams]:
    """
    Estimate camera parameters and distortion coefficients from landmark correspondences.
    
    This uses the detected pitch landmarks to estimate camera intrinsics and distortion,
    which can then be used to undistort images before homography estimation.
    
    Args:
        image_points: Detected landmark positions in image [N, 2]
        world_points: Corresponding world coordinates [N, 2] (pitch coordinates in meters)
        image_size: (width, height) of image
        is_fisheye: If True, use fisheye model; if False, use standard model
    
    Returns:
        DistortionParams if estimation successful, None otherwise
    """
    if len(image_points) < 4:
        return None
    
    width, height = image_size
    
    # Initialize camera matrix with reasonable defaults
    # Focal length estimated from image size (typical for wide-angle sports cameras)
    fx = fy = max(width, height) * 0.8  # Conservative estimate
    cx = width / 2.0
    cy = height / 2.0
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize distortion coefficients
    if is_fisheye:
        # Fisheye model: 4 coefficients (k1, k2, k3, k4)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    else:
        # Standard model: 5 coefficients (k1, k2, p1, p2, k3)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # Convert 2D world points to 3D (z=0 for pitch plane)
    world_points_3d = np.zeros((len(world_points), 3), dtype=np.float32)
    world_points_3d[:, :2] = world_points
    
    # Convert to object points format for calibration
    object_points = [world_points_3d]
    image_points_list = [image_points.reshape(-1, 1, 2)]
    
    try:
        if is_fisheye:
            # Use fisheye calibration
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points_list,
                (width, height),
                camera_matrix,
                dist_coeffs,
                flags=flags,
                criteria=criteria
            )
            
            if ret:
                return DistortionParams(
                    camera_matrix=K,
                    dist_coeffs=D,
                    is_fisheye=True,
                    confidence=0.7  # Fisheye calibration is less reliable from landmarks alone
                )
        else:
            # Use standard calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points_list,
                (width, height),
                camera_matrix,
                dist_coeffs,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
            
            if ret:
                # Calculate reprojection error as confidence measure
                total_error = 0
                for i in range(len(object_points)):
                    imgpoints2, _ = cv2.projectPoints(
                        object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(image_points_list[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                
                mean_error = total_error / len(object_points)
                # Confidence inversely related to error (lower error = higher confidence)
                confidence = max(0.0, min(1.0, 1.0 - (mean_error / 10.0)))
                
                return DistortionParams(
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    is_fisheye=False,
                    confidence=confidence
                )
    except Exception as e:
        # Calibration failed
        return None
    
    return None


def undistort_image(
    image: np.ndarray,
    distortion_params: DistortionParams
) -> np.ndarray:
    """
    Undistort an image using estimated camera parameters.
    
    Args:
        image: Input image (BGR format)
        distortion_params: Estimated distortion parameters
    
    Returns:
        Undistorted image
    """
    h, w = image.shape[:2]
    
    if distortion_params.is_fisheye:
        # Fisheye undistortion
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            distortion_params.camera_matrix,
            distortion_params.dist_coeffs,
            (w, h),
            np.eye(3),
            balance=0.0  # No cropping, preserve full FOV
        )
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            distortion_params.camera_matrix,
            distortion_params.dist_coeffs,
            np.eye(3),
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2
        )
        
        undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        # Standard undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            distortion_params.camera_matrix,
            distortion_params.dist_coeffs,
            (w, h),
            alpha=1.0  # No cropping, preserve full FOV
        )
        
        undistorted = cv2.undistort(image, distortion_params.camera_matrix, distortion_params.dist_coeffs, None, new_camera_matrix)
    
    return undistorted


def undistort_points(
    points: np.ndarray,
    distortion_params: DistortionParams
) -> np.ndarray:
    """
    Undistort point coordinates.
    
    Args:
        points: Points in distorted image coordinates [N, 2]
        distortion_params: Estimated distortion parameters
    
    Returns:
        Undistorted points [N, 2]
    """
    if distortion_params.is_fisheye:
        # Fisheye point undistortion
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.fisheye.undistortPoints(
            points_reshaped,
            distortion_params.camera_matrix,
            distortion_params.dist_coeffs,
            P=distortion_params.camera_matrix
        )
        return undistorted.reshape(-1, 2)
    else:
        # Standard point undistortion
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(
            points_reshaped,
            distortion_params.camera_matrix,
            distortion_params.dist_coeffs,
            P=distortion_params.camera_matrix
        )
        return undistorted.reshape(-1, 2)


def detect_fisheye_distortion(
    image_points: np.ndarray,
    world_points: np.ndarray,
    image_size: Tuple[int, int]
) -> bool:
    """
    Detect if image has fisheye distortion by analyzing landmark distribution.
    
    Fisheye distortion causes:
    - Curved lines near edges
    - Compression of features near image boundaries
    - Expansion near center
    
    Args:
        image_points: Detected landmark positions [N, 2]
        world_points: Corresponding world coordinates [N, 2]
        image_size: (width, height) of image
    
    Returns:
        True if fisheye distortion detected, False otherwise
    """
    if len(image_points) < 6:
        return False
    
    width, height = image_size
    
    # Calculate distances from image center
    center = np.array([width / 2, height / 2])
    distances_from_center = np.linalg.norm(image_points - center, axis=1)
    max_distance = np.sqrt(width**2 + height**2) / 2
    normalized_distances = distances_from_center / max_distance
    
    # Calculate expected vs actual spacing for landmarks
    # In fisheye, landmarks near edges appear closer together than they should
    # Check if landmarks near edges have compressed spacing
    edge_threshold = 0.7  # Consider points >70% from center as "edge"
    edge_mask = normalized_distances > edge_threshold
    
    if np.sum(edge_mask) < 3:
        return False
    
    # Calculate spacing between adjacent edge landmarks
    edge_points = image_points[edge_mask]
    edge_world = world_points[edge_mask]
    
    if len(edge_points) < 3:
        return False
    
    # Calculate image spacing vs world spacing ratio
    # In fisheye, image spacing should be compressed relative to world spacing
    image_distances = []
    world_distances = []
    
    for i in range(len(edge_points) - 1):
        img_dist = np.linalg.norm(edge_points[i+1] - edge_points[i])
        world_dist = np.linalg.norm(edge_world[i+1] - edge_world[i])
        if world_dist > 0.1:  # Only consider significant distances
            image_distances.append(img_dist)
            world_distances.append(world_dist)
    
    if len(image_distances) < 2:
        return False
    
    # Calculate compression ratio (lower = more compression = more fisheye)
    ratios = np.array(image_distances) / (np.array(world_distances) + 1e-6)
    mean_ratio = np.mean(ratios)
    
    # If edge landmarks are significantly compressed, likely fisheye
    # Typical fisheye compression: 0.3-0.7 for edge features
    return mean_ratio < 0.8


def apply_polynomial_y_correction(
    y_coords: np.ndarray,
    image_height: int,
    correction_factor: float = 1.2
) -> np.ndarray:
    """
    Apply polynomial correction to y-axis coordinates to reduce fisheye effect.
    
    This is a fallback method when full camera calibration isn't available.
    Uses a polynomial model to correct y-axis compression near edges.
    
    Args:
        y_coords: Y coordinates to correct (normalized 0-1 or pixel coordinates)
        image_height: Image height for normalization
        correction_factor: Strength of correction (1.0 = no correction, >1.0 = more correction)
    
    Returns:
        Corrected y coordinates
    """
    # Normalize to 0-1 range
    if np.max(y_coords) > 1.0:
        y_norm = y_coords / image_height
    else:
        y_norm = y_coords
    
    # Center around 0.5 (middle of image)
    y_centered = y_norm - 0.5
    
    # Apply polynomial correction: y_corrected = y + k * y^3
    # This expands coordinates near edges (where |y| is large)
    y_corrected = y_centered + (correction_factor - 1.0) * y_centered**3
    
    # Shift back and denormalize
    y_final = y_corrected + 0.5
    
    if np.max(y_coords) > 1.0:
        return y_final * image_height
    else:
        return y_final
