"""
Team ID Assignment (R-002)
HSV-based color clustering with green suppression and GK detection
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TeamAssignment:
    """Result of team assignment"""
    team_id: Optional[int]  # 0 or 1 for teams, None for unassigned
    role: str  # "PLAYER", "GK", "REF"
    confidence: float  # Distance-based confidence
    is_outlier: bool


class TeamClusterer:
    """
    Team identification using HSV color space clustering.
    
    Implements the "Global Appearance Model" strategy:
    1. Golden Batch: Initialize with high-confidence crops from first N frames
    2. Frame-by-frame: Fast vectorized assignment using pre-computed centroids
    3. Outlier Detection: Identify GK and referees using distance + spatial heuristics
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize team clusterer
        
        Args:
            pitch_length: Standard pitch length in meters (for penalty box detection)
            pitch_width: Standard pitch width in meters
        """
        self.kmeans = None
        self.team_centroids = None
        self.outlier_threshold = 0.0
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.penalty_box_length = 16.5  # Standard penalty box length
        
        # Accumulated crops for Golden Batch
        self.accumulated_crops = []
        self.accumulated_positions = []  # For spatial verification
        
    def fit(self, player_crops: List[np.ndarray], positions: Optional[List[Tuple[float, float]]] = None,
            confidence_threshold: float = 0.8, min_crops: int = 20):
        """
        Initialize the model using a 'Golden Batch' of crops.
        
        Args:
            player_crops: List of player crop images (numpy arrays)
            positions: Optional list of (x, y) pitch coordinates for each crop
            confidence_threshold: Minimum confidence for crops to include
            min_crops: Minimum number of crops required for training
        
        Returns:
            True if training successful, False otherwise
        """
        if len(player_crops) < min_crops:
            return False
        
        # Extract HSV features from all crops
        features = []
        valid_positions = []
        
        for i, crop in enumerate(player_crops):
            if crop is None or crop.size == 0:
                continue
                
            feature = self._extract_hsv_feature(crop)
            if feature is not None:
                features.append(feature)
                if positions and i < len(positions):
                    valid_positions.append(positions[i])
        
        if len(features) < min_crops:
            return False
        
        # Train K-Means (K=2 for two teams)
        features_array = np.array(features)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.kmeans.fit(features_array)
        self.team_centroids = self.kmeans.cluster_centers_
        
        # Calculate outlier threshold (95th percentile distance)
        distances = self.kmeans.transform(features_array)
        min_dists = np.min(distances, axis=1)
        self.outlier_threshold = np.percentile(min_dists, 95)
        
        # Store accumulated data for potential re-training
        self.accumulated_crops = player_crops
        self.accumulated_positions = valid_positions
        
        return True
    
    def predict(self, crop: np.ndarray, position_xy: Optional[Tuple[float, float]] = None) -> TeamAssignment:
        """
        Assigns Team ID or role based on feature distance and position.
        
        Args:
            crop: Player crop image (numpy array)
            position_xy: Optional (x, y) pitch coordinates for spatial verification
        
        Returns:
            TeamAssignment with team_id, role, confidence, and outlier flag
        """
        if self.team_centroids is None:
            return TeamAssignment(
                team_id=None,
                role="PLAYER",
                confidence=0.0,
                is_outlier=True
            )
        
        # Extract feature
        feature = self._extract_hsv_feature(crop)
        if feature is None:
            return TeamAssignment(
                team_id=None,
                role="PLAYER",
                confidence=0.0,
                is_outlier=True
            )
        
        # Vectorized distance calculation
        dists = np.linalg.norm(self.team_centroids - feature, axis=1)
        min_dist = np.min(dists)
        label = np.argmin(dists)
        
        # Outlier Logic
        if min_dist > self.outlier_threshold * 1.5:
            # Check if inside penalty box (spatial verification for GK)
            if position_xy and self._is_in_penalty_box(position_xy):
                return TeamAssignment(
                    team_id=None,
                    role="GK",
                    confidence=1.0 - (min_dist / (self.outlier_threshold * 2.0)),
                    is_outlier=True
                )
            else:
                # Likely referee (outlier but not in penalty box)
                return TeamAssignment(
                    team_id=None,
                    role="REF",
                    confidence=1.0 - (min_dist / (self.outlier_threshold * 2.0)),
                    is_outlier=True
                )
        
        # Normal team assignment
        confidence = 1.0 - (min_dist / (self.outlier_threshold * 2.0))
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return TeamAssignment(
            team_id=int(label),
            role="PLAYER",
            confidence=float(confidence),
            is_outlier=False
        )
    
    def predict_batch(self, crops: List[np.ndarray], 
                     positions: Optional[List[Tuple[float, float]]] = None) -> List[TeamAssignment]:
        """
        Batch prediction for multiple crops (vectorized for speed).
        
        Args:
            crops: List of player crop images
            positions: Optional list of (x, y) pitch coordinates
        
        Returns:
            List of TeamAssignment objects
        """
        if self.team_centroids is None:
            return [TeamAssignment(None, "PLAYER", 0.0, True) for _ in crops]
        
        # Extract features for all crops
        features = []
        valid_indices = []
        
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            feature = self._extract_hsv_feature(crop)
            if feature is not None:
                features.append(feature)
                valid_indices.append(i)
        
        if not features:
            return [TeamAssignment(None, "PLAYER", 0.0, True) for _ in crops]
        
        # Vectorized distance calculation
        features_array = np.array(features)
        dists = np.linalg.norm(
            self.team_centroids[np.newaxis, :, :] - features_array[:, np.newaxis, :],
            axis=2
        )
        min_dists = np.min(dists, axis=1)
        labels = np.argmin(dists, axis=1)
        
        # Create assignments
        assignments = [TeamAssignment(None, "PLAYER", 0.0, True) for _ in crops]
        
        for idx, valid_idx in enumerate(valid_indices):
            min_dist = min_dists[idx]
            label = labels[idx]
            position = positions[valid_idx] if positions and valid_idx < len(positions) else None
            
            # Outlier check
            if min_dist > self.outlier_threshold * 1.5:
                if position and self._is_in_penalty_box(position):
                    role = "GK"
                else:
                    role = "REF"
                confidence = 1.0 - (min_dist / (self.outlier_threshold * 2.0))
                confidence = np.clip(confidence, 0.0, 1.0)
                
                assignments[valid_idx] = TeamAssignment(
                    team_id=None,
                    role=role,
                    confidence=float(confidence),
                    is_outlier=True
                )
            else:
                confidence = 1.0 - (min_dist / (self.outlier_threshold * 2.0))
                confidence = np.clip(confidence, 0.0, 1.0)
                
                assignments[valid_idx] = TeamAssignment(
                    team_id=int(label),
                    role="PLAYER",
                    confidence=float(confidence),
                    is_outlier=False
                )
        
        return assignments
    
    def _extract_hsv_feature(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract HSV feature vector from player crop.
        
        Strategy:
        1. Central crop (30% of bounding box) to focus on jersey
        2. Convert RGB → HSV
        3. Green masking (Hue 35-85) to exclude pitch background
        4. Compute mean HSV of non-green pixels
        
        Args:
            crop: Player crop image (BGR format from OpenCV)
        
        Returns:
            Mean HSV vector [H, S, V] or None if insufficient data
        """
        if crop is None or crop.size == 0:
            return None
        
        h, w = crop.shape[:2]
        
        # 1. Central Crop (30% of bounding box)
        cy, cx = h // 2, w // 2
        dy, dx = int(h * 0.15), int(w * 0.15)
        
        # Ensure valid bounds
        y_start = max(0, cy - dy)
        y_end = min(h, cy + dy)
        x_start = max(0, cx - dx)
        x_end = min(w, cx + dx)
        
        if y_end <= y_start or x_end <= x_start:
            return None
        
        center = crop[y_start:y_end, x_start:x_end]
        
        if center.size == 0:
            return None
        
        # 2. HSV Conversion
        # OpenCV uses BGR, so convert BGR to HSV
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        
        # 3. Green Masking (Hue 35-85 in OpenCV HSV: H is 0-179)
        # Green typically falls in this range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        
        # 4. Compute Mean of Non-Green Pixels
        non_green_count = cv2.countNonZero(mask_inv)
        if non_green_count == 0:
            # Fallback: use all pixels if no non-green pixels
            mean_val = cv2.mean(hsv)[:3]
        else:
            mean_val = cv2.mean(hsv, mask=mask_inv)[:3]
        
        return np.array(mean_val, dtype=np.float32)
    
    def _is_in_penalty_box(self, position_xy: Tuple[float, float]) -> bool:
        """
        Check if position is within penalty box.
        
        Standard pitch: 105m x 68m
        Penalty box: 16.5m from goal line (each end)
        
        Args:
            position_xy: (x, y) pitch coordinates in meters
        
        Returns:
            True if in penalty box, False otherwise
        """
        if position_xy is None:
            return False
        
        x, y = position_xy
        
        # Check if x is within penalty box (either end)
        # Assuming x=0 is one goal line, x=105 is the other
        in_left_box = x < self.penalty_box_length
        in_right_box = x > (self.pitch_length - self.penalty_box_length)
        
        return in_left_box or in_right_box
    
    def get_team_colors(self) -> Optional[Dict[int, np.ndarray]]:
        """
        Get the learned team color centroids (for visualization).
        
        Returns:
            Dictionary mapping team_id to HSV color vector, or None if not trained
        """
        if self.team_centroids is None:
            return None
        
        return {
            0: self.team_centroids[0],
            1: self.team_centroids[1]
        }
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.team_centroids is not None
