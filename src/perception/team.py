# K-Means clustering for team assignment
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
import cv2
from src.types import TrackedObject, Player


def extract_jersey_color(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract dominant color from jersey region
    
    Args:
        frame: BGR image
        bbox: (x, y, width, height)
    
    Returns:
        RGB color vector
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Extract region (with bounds checking)
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame))
    y = max(0, min(y, h_frame))
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    
    if w <= 0 or h <= 0:
        return np.array([128, 128, 128])  # Default gray
    
    # Extract upper third of bbox (jersey area)
    jersey_region = frame[y:y+h//3, x:x+w]
    
    if jersey_region.size == 0:
        return np.array([128, 128, 128])
    
    # Reshape to list of pixels
    pixels = jersey_region.reshape(-1, 3)
    
    # Get dominant color (mean of top 10% brightest pixels)
    brightness = np.sum(pixels, axis=1)
    top_indices = np.argsort(brightness)[-len(pixels)//10:]
    dominant_color = np.mean(pixels[top_indices], axis=0)
    
    # Convert BGR to RGB
    return dominant_color[::-1]


def assign_teams(tracked_objects: List[TrackedObject], frame: np.ndarray, 
                 n_clusters: int = 2) -> List[TrackedObject]:
    """
    Assign team IDs using K-Means clustering on jersey colors
    
    Args:
        tracked_objects: List of tracked objects
        frame: Current frame
        n_clusters: Number of teams (default 2)
    
    Returns:
        List of tracked objects with team_id assigned
    """
    if len(tracked_objects) < n_clusters:
        return tracked_objects
    
    # Extract colors
    colors = []
    valid_indices = []
    for i, obj in enumerate(tracked_objects):
        # Only process players (class_id 0), skip ball (class_id 1)
        if obj.detection.class_id == 0:
            color = extract_jersey_color(frame, obj.detection.bbox)
            colors.append(color)
            valid_indices.append(i)
    
    if len(colors) < n_clusters:
        return tracked_objects
    
    # K-Means clustering
    colors_array = np.array(colors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors_array)
    
    # Assign team IDs
    for idx, label in zip(valid_indices, labels):
        tracked_objects[idx].team_id = int(label)
    
    return tracked_objects
