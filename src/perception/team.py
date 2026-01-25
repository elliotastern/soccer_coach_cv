"""
Team identification utilities and crop extraction
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from src.types import Detection, TrackedObject
from src.logic.team_id import TeamClusterer, TeamAssignment


def extract_player_crop(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Extract player crop from frame using bounding box.
    
    Args:
        frame: BGR image (numpy array from OpenCV)
        bbox: Bounding box (x, y, width, height) in pixels
    
    Returns:
        Cropped image or None if invalid
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Bounds checking
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame))
    y = max(0, min(y, h_frame))
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    
    if w <= 0 or h <= 0:
        return None
    
    crop = frame[y:y+h, x:x+w]
    return crop


def extract_player_crops(frame: np.ndarray, detections: List[Detection], 
                         class_filter: str = 'player') -> List[Tuple[np.ndarray, Detection]]:
    """
    Extract crops for all player detections in a frame.
    
    Args:
        frame: BGR image
        detections: List of Detection objects
        class_filter: Filter by class name ('player', 'ball', or None for all)
    
    Returns:
        List of (crop, detection) tuples
    """
    crops = []
    
    for detection in detections:
        # Filter by class if specified
        if class_filter and detection.class_name != class_filter:
            continue
        
        crop = extract_player_crop(frame, detection.bbox)
        if crop is not None:
            crops.append((crop, detection))
    
    return crops


def extract_crops_for_team_assignment(frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
    """
    Extract crops specifically for team assignment (players only).
    
    Args:
        frame: BGR image
        detections: List of Detection objects
    
    Returns:
        List of player crop images
    """
    crops = []
    
    for detection in detections:
        # Only process players (class_id 0 or class_name 'player')
        if detection.class_id == 0 or detection.class_name == 'player':
            crop = extract_player_crop(frame, detection.bbox)
            if crop is not None:
                crops.append(crop)
    
    return crops


def assign_teams_to_tracked_objects(
    tracked_objects: List[TrackedObject],
    frame: np.ndarray,
    team_clusterer: TeamClusterer,
    pitch_positions: Optional[List[Tuple[float, float]]] = None
) -> List[TrackedObject]:
    """
    Assign team IDs to tracked objects using TeamClusterer.
    
    Args:
        tracked_objects: List of TrackedObject instances
        frame: Current frame (BGR)
        team_clusterer: Trained TeamClusterer instance
        pitch_positions: Optional list of (x, y) pitch coordinates for each object
    
    Returns:
        List of TrackedObject instances with team_id and role assigned
    """
    if not team_clusterer.is_trained():
        return tracked_objects
    
    # Extract crops for players only
    player_crops = []
    player_indices = []
    
    for i, obj in enumerate(tracked_objects):
        if obj.detection.class_id == 0 or obj.detection.class_name == 'player':
            crop = extract_player_crop(frame, obj.detection.bbox)
            if crop is not None:
                player_crops.append(crop)
                player_indices.append(i)
    
    if not player_crops:
        return tracked_objects
    
    # Get pitch positions for players (if available)
    player_positions = None
    if pitch_positions:
        player_positions = [pitch_positions[i] for i in player_indices if i < len(pitch_positions)]
    
    # Predict team assignments
    assignments = team_clusterer.predict_batch(player_crops, player_positions)
    
    # Assign to tracked objects
    for idx, assignment in zip(player_indices, assignments):
        tracked_objects[idx].team_id = assignment.team_id
        # Note: role would be stored if we add it to TrackedObject
    
    return tracked_objects


# Legacy function for backward compatibility
def extract_jersey_color(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract dominant color from jersey region (legacy method).
    
    This is kept for backward compatibility but TeamClusterer should be used instead.
    
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


# Legacy function for backward compatibility
def assign_teams(tracked_objects: List[TrackedObject], frame: np.ndarray, 
                 n_clusters: int = 2) -> List[TrackedObject]:
    """
    Assign team IDs using K-Means clustering on jersey colors (legacy method).
    
    This is kept for backward compatibility but TeamClusterer should be used instead.
    
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
    from sklearn.cluster import KMeans
    colors_array = np.array(colors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors_array)
    
    # Assign team IDs
    for idx, label in zip(valid_indices, labels):
        tracked_objects[idx].team_id = int(label)
    
    return tracked_objects
