# Gatekeeper Module - Filter gameplay frames and detect scene cuts
import cv2
import numpy as np
from typing import Optional


def is_gameplay_view(frame: np.ndarray, green_threshold: float = 0.5) -> bool:
    """
    Check if frame shows gameplay view (pitch visible)
    
    Args:
        frame: BGR image
        green_threshold: Minimum ratio of green pixels (default 0.5 = 50%)
    
    Returns:
        True if gameplay view, False otherwise
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green pitch color range in HSV
    # H: 40-80 (green), S: >50, V: >50
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    
    return green_ratio >= green_threshold


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
