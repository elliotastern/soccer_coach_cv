# Gatekeeper Module - Filter gameplay frames and detect scene cuts
import cv2
import numpy as np
from typing import Optional

_LOWER_GREEN = np.array([40, 50, 50])
_UPPER_GREEN = np.array([80, 255, 255])


def compute_green_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels in the green pitch HSV band."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LOWER_GREEN, _UPPER_GREEN)
    return float(np.sum(mask > 0) / (frame.shape[0] * frame.shape[1]))


def is_gameplay_view(frame: np.ndarray, green_threshold: float = 0.5) -> bool:
    """
    Check if frame shows gameplay view (pitch visible)
    
    Args:
        frame: BGR image
        green_threshold: Minimum ratio of green pixels (default 0.5 = 50%)
    
    Returns:
        True if gameplay view, False otherwise
    """
    return compute_green_ratio(frame) >= green_threshold


def detect_scene_cut(frame: np.ndarray, prev_frame: Optional[np.ndarray], 
                     threshold: float = 0.7) -> bool:
    """
    Detect scene cut between consecutive frames
    
    Args:
        frame: Current frame (BGR)
        prev_frame: Previous frame (BGR) or None
        threshold: Histogram difference threshold (default 0.7)
    
    Returns:
        True if scene cut detected, False otherwise
    """
    if prev_frame is None:
        return False
    
    if frame.shape != prev_frame.shape:
        return True
    
    # Calculate histogram for each channel
    hist_diff = 0.0
    for i in range(3):
        hist1 = cv2.calcHist([frame], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([prev_frame], [i], None, [256], [0, 256])
        
        # Normalized correlation
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_diff += (1.0 - corr) / 3.0
    
    return hist_diff > threshold
