"""
Comprehensive pitch landmark database with FIFA-standard coordinates.
Supports adaptive scaling for non-standard fields.
"""
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Landmark:
    """Pitch landmark with coordinates and metadata"""
    name: str
    pitch_coords: Tuple[float, float]  # (x, y) in meters, origin at center
    landmark_type: str  # "goal", "corner", "penalty_box", "center_circle", etc.
    priority: int  # 1=primary, 2=secondary, 3=tertiary
    description: str = ""


class LandmarkDatabase:
    """
    Comprehensive database of FIFA-standard pitch landmarks.
    Supports adaptive scaling for non-standard field dimensions.
    """
    
    # FIFA standard dimensions
    FIFA_LENGTH = 105.0  # meters
    FIFA_WIDTH = 68.0    # meters
    
    # Standard pitch measurements
    GOAL_WIDTH = 7.32  # meters
    PENALTY_BOX_DEPTH = 16.5  # meters (from goal line)
    PENALTY_BOX_WIDTH = 40.32  # meters
    GOAL_AREA_DEPTH = 5.5  # meters (6-yard box)
    GOAL_AREA_WIDTH = 18.32  # meters
    PENALTY_SPOT_DISTANCE = 11.0  # meters from goal line
    CENTER_CIRCLE_RADIUS = 9.15  # meters
    CORNER_ARC_RADIUS = 1.0  # meters
    
    def __init__(self, pitch_length: float = FIFA_LENGTH, pitch_width: float = FIFA_WIDTH):
        """
        Initialize landmark database
        
        Args:
            pitch_length: Pitch length in meters (default: 105.0)
            pitch_width: Pitch width in meters (default: 68.0)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.scale_factor_x = pitch_length / self.FIFA_LENGTH
        self.scale_factor_y = pitch_width / self.FIFA_WIDTH
        
        # Generate all landmarks
        self._landmarks = self._generate_landmarks()
    
    def _generate_landmarks(self) -> List[Landmark]:
        """Generate all pitch landmarks based on current dimensions"""
        landmarks = []
        
        # Scale factors for non-standard fields
        sx = self.scale_factor_x
        sy = self.scale_factor_y
        
        # Primary Landmarks (Priority 1)
        
        # Goals (2)
        landmarks.append(Landmark(
            name="goal_left_center",
            pitch_coords=(-self.pitch_length / 2, 0.0),
            landmark_type="goal",
            priority=1,
            description="Left goal center"
        ))
        landmarks.append(Landmark(
            name="goal_right_center",
            pitch_coords=(self.pitch_length / 2, 0.0),
            landmark_type="goal",
            priority=1,
            description="Right goal center"
        ))
        
        # Center point (1)
        landmarks.append(Landmark(
            name="center_point",
            pitch_coords=(0.0, 0.0),
            landmark_type="center_line",
            priority=1,
            description="Center of pitch (halfway point)"
        ))
        
        # Field corners (4)
        landmarks.append(Landmark(
            name="corner_top_left",
            pitch_coords=(-self.pitch_length / 2, -self.pitch_width / 2),
            landmark_type="corner",
            priority=1,
            description="Top-left corner"
        ))
        landmarks.append(Landmark(
            name="corner_top_right",
            pitch_coords=(self.pitch_length / 2, -self.pitch_width / 2),
            landmark_type="corner",
            priority=1,
            description="Top-right corner"
        ))
        landmarks.append(Landmark(
            name="corner_bottom_right",
            pitch_coords=(self.pitch_length / 2, self.pitch_width / 2),
            landmark_type="corner",
            priority=1,
            description="Bottom-right corner"
        ))
        landmarks.append(Landmark(
            name="corner_bottom_left",
            pitch_coords=(-self.pitch_length / 2, self.pitch_width / 2),
            landmark_type="corner",
            priority=1,
            description="Bottom-left corner"
        ))
        
        # Center circle center (1)
        landmarks.append(Landmark(
            name="center_circle_center",
            pitch_coords=(0.0, 0.0),
            landmark_type="center_circle",
            priority=1,
            description="Center circle center"
        ))
        
        # Secondary Landmarks (Priority 2)
        
        # Penalty box corners - Left box (4 corners)
        penalty_left_x = -self.pitch_length / 2 + self.PENALTY_BOX_DEPTH * sx
        penalty_width_scaled = self.PENALTY_BOX_WIDTH * sy
        
        landmarks.append(Landmark(
            name="penalty_box_left_top",
            pitch_coords=(penalty_left_x, -penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Left penalty box top corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_left_bottom",
            pitch_coords=(penalty_left_x, penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Left penalty box bottom corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_left_goal_top",
            pitch_coords=(-self.pitch_length / 2, -penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Left penalty box goal line top corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_left_goal_bottom",
            pitch_coords=(-self.pitch_length / 2, penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Left penalty box goal line bottom corner"
        ))
        
        # Penalty box corners - Right box (4 corners)
        penalty_right_x = self.pitch_length / 2 - self.PENALTY_BOX_DEPTH * sx
        
        landmarks.append(Landmark(
            name="penalty_box_right_top",
            pitch_coords=(penalty_right_x, -penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Right penalty box top corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_right_bottom",
            pitch_coords=(penalty_right_x, penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Right penalty box bottom corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_right_goal_top",
            pitch_coords=(self.pitch_length / 2, -penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Right penalty box goal line top corner"
        ))
        landmarks.append(Landmark(
            name="penalty_box_right_goal_bottom",
            pitch_coords=(self.pitch_length / 2, penalty_width_scaled / 2),
            landmark_type="penalty_box",
            priority=2,
            description="Right penalty box goal line bottom corner"
        ))
        
        # Penalty spots (2)
        penalty_spot_dist = self.PENALTY_SPOT_DISTANCE * sx
        landmarks.append(Landmark(
            name="penalty_spot_left",
            pitch_coords=(-self.pitch_length / 2 + penalty_spot_dist, 0.0),
            landmark_type="penalty_spot",
            priority=2,
            description="Left penalty spot"
        ))
        landmarks.append(Landmark(
            name="penalty_spot_right",
            pitch_coords=(self.pitch_length / 2 - penalty_spot_dist, 0.0),
            landmark_type="penalty_spot",
            priority=2,
            description="Right penalty spot"
        ))
        
        # Goal area (6-yard box) corners - Left (2 corners)
        goal_area_left_x = -self.pitch_length / 2 + self.GOAL_AREA_DEPTH * sx
        goal_area_width_scaled = self.GOAL_AREA_WIDTH * sy
        
        landmarks.append(Landmark(
            name="goal_area_left_top",
            pitch_coords=(goal_area_left_x, -goal_area_width_scaled / 2),
            landmark_type="goal_area",
            priority=2,
            description="Left goal area (6-yard box) top corner"
        ))
        landmarks.append(Landmark(
            name="goal_area_left_bottom",
            pitch_coords=(goal_area_left_x, goal_area_width_scaled / 2),
            landmark_type="goal_area",
            priority=2,
            description="Left goal area (6-yard box) bottom corner"
        ))
        
        # Goal area corners - Right (2 corners)
        goal_area_right_x = self.pitch_length / 2 - self.GOAL_AREA_DEPTH * sx
        
        landmarks.append(Landmark(
            name="goal_area_right_top",
            pitch_coords=(goal_area_right_x, -goal_area_width_scaled / 2),
            landmark_type="goal_area",
            priority=2,
            description="Right goal area (6-yard box) top corner"
        ))
        landmarks.append(Landmark(
            name="goal_area_right_bottom",
            pitch_coords=(goal_area_right_x, goal_area_width_scaled / 2),
            landmark_type="goal_area",
            priority=2,
            description="Right goal area (6-yard box) bottom corner"
        ))
        
        # Center circle points (4 cardinal directions)
        circle_radius_scaled = self.CENTER_CIRCLE_RADIUS * min(sx, sy)
        landmarks.append(Landmark(
            name="center_circle_top",
            pitch_coords=(0.0, -circle_radius_scaled),
            landmark_type="center_circle",
            priority=2,
            description="Center circle top point"
        ))
        landmarks.append(Landmark(
            name="center_circle_bottom",
            pitch_coords=(0.0, circle_radius_scaled),
            landmark_type="center_circle",
            priority=2,
            description="Center circle bottom point"
        ))
        landmarks.append(Landmark(
            name="center_circle_left",
            pitch_coords=(-circle_radius_scaled, 0.0),
            landmark_type="center_circle",
            priority=2,
            description="Center circle left point"
        ))
        landmarks.append(Landmark(
            name="center_circle_right",
            pitch_coords=(circle_radius_scaled, 0.0),
            landmark_type="center_circle",
            priority=2,
            description="Center circle right point"
        ))
        
        # Center line endpoints (2)
        landmarks.append(Landmark(
            name="center_line_left",
            pitch_coords=(-self.pitch_length / 2, 0.0),
            landmark_type="center_line",
            priority=2,
            description="Center line left endpoint"
        ))
        landmarks.append(Landmark(
            name="center_line_right",
            pitch_coords=(self.pitch_length / 2, 0.0),
            landmark_type="center_line",
            priority=2,
            description="Center line right endpoint"
        ))
        
        # Tertiary Landmarks (Priority 3)
        
        # Corner arcs (4) - points on corner arcs
        corner_arc_radius_scaled = self.CORNER_ARC_RADIUS * min(sx, sy)
        landmarks.append(Landmark(
            name="corner_arc_top_left_x",
            pitch_coords=(-self.pitch_length / 2 + corner_arc_radius_scaled, -self.pitch_width / 2),
            landmark_type="corner_arc",
            priority=3,
            description="Top-left corner arc (x-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_top_left_y",
            pitch_coords=(-self.pitch_length / 2, -self.pitch_width / 2 + corner_arc_radius_scaled),
            landmark_type="corner_arc",
            priority=3,
            description="Top-left corner arc (y-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_top_right_x",
            pitch_coords=(self.pitch_length / 2 - corner_arc_radius_scaled, -self.pitch_width / 2),
            landmark_type="corner_arc",
            priority=3,
            description="Top-right corner arc (x-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_top_right_y",
            pitch_coords=(self.pitch_length / 2, -self.pitch_width / 2 + corner_arc_radius_scaled),
            landmark_type="corner_arc",
            priority=3,
            description="Top-right corner arc (y-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_bottom_right_x",
            pitch_coords=(self.pitch_length / 2 - corner_arc_radius_scaled, self.pitch_width / 2),
            landmark_type="corner_arc",
            priority=3,
            description="Bottom-right corner arc (x-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_bottom_right_y",
            pitch_coords=(self.pitch_length / 2, self.pitch_width / 2 - corner_arc_radius_scaled),
            landmark_type="corner_arc",
            priority=3,
            description="Bottom-right corner arc (y-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_bottom_left_x",
            pitch_coords=(-self.pitch_length / 2 + corner_arc_radius_scaled, self.pitch_width / 2),
            landmark_type="corner_arc",
            priority=3,
            description="Bottom-left corner arc (x-axis)"
        ))
        landmarks.append(Landmark(
            name="corner_arc_bottom_left_y",
            pitch_coords=(-self.pitch_length / 2, self.pitch_width / 2 - corner_arc_radius_scaled),
            landmark_type="corner_arc",
            priority=3,
            description="Bottom-left corner arc (y-axis)"
        ))
        
        # Touchline midpoints (4)
        landmarks.append(Landmark(
            name="touchline_left_midpoint",
            pitch_coords=(-self.pitch_length / 2, 0.0),
            landmark_type="touchline",
            priority=3,
            description="Left touchline midpoint"
        ))
        landmarks.append(Landmark(
            name="touchline_right_midpoint",
            pitch_coords=(self.pitch_length / 2, 0.0),
            landmark_type="touchline",
            priority=3,
            description="Right touchline midpoint"
        ))
        landmarks.append(Landmark(
            name="touchline_top_midpoint",
            pitch_coords=(0.0, -self.pitch_width / 2),
            landmark_type="touchline",
            priority=3,
            description="Top touchline midpoint"
        ))
        landmarks.append(Landmark(
            name="touchline_bottom_midpoint",
            pitch_coords=(0.0, self.pitch_width / 2),
            landmark_type="touchline",
            priority=3,
            description="Bottom touchline midpoint"
        ))
        
        # Goal area centers (2)
        landmarks.append(Landmark(
            name="goal_area_left_center",
            pitch_coords=(-self.pitch_length / 2 + self.GOAL_AREA_DEPTH * sx / 2, 0.0),
            landmark_type="goal_area",
            priority=3,
            description="Left goal area center"
        ))
        landmarks.append(Landmark(
            name="goal_area_right_center",
            pitch_coords=(self.pitch_length / 2 - self.GOAL_AREA_DEPTH * sx / 2, 0.0),
            landmark_type="goal_area",
            priority=3,
            description="Right goal area center"
        ))
        
        return landmarks
    
    def get_all_landmarks(self) -> List[Landmark]:
        """Get all landmarks"""
        return self._landmarks.copy()
    
    def get_landmarks_by_priority(self, priority: int) -> List[Landmark]:
        """Get landmarks by priority level (1=primary, 2=secondary, 3=tertiary)"""
        return [lm for lm in self._landmarks if lm.priority == priority]
    
    def get_landmarks_by_type(self, landmark_type: str) -> List[Landmark]:
        """Get landmarks by type"""
        return [lm for lm in self._landmarks if lm.landmark_type == landmark_type]
    
    def get_landmark_by_name(self, name: str) -> Optional[Landmark]:
        """Get a specific landmark by name"""
        for lm in self._landmarks:
            if lm.name == name:
                return lm
        return None
    
    def scale_landmarks(self, new_length: float, new_width: float) -> 'LandmarkDatabase':
        """
        Create a new database with scaled landmarks for non-standard fields
        
        Args:
            new_length: New pitch length in meters
            new_width: New pitch width in meters
        
        Returns:
            New LandmarkDatabase instance with scaled landmarks
        """
        return LandmarkDatabase(new_length, new_width)
    
    def get_landmark_coordinates_dict(self) -> Dict[str, Tuple[float, float]]:
        """
        Get dictionary mapping landmark names to coordinates
        
        Returns:
            Dictionary {landmark_name: (x, y)}
        """
        return {lm.name: lm.pitch_coords for lm in self._landmarks}
    
    def get_primary_landmarks(self) -> List[Landmark]:
        """Get primary landmarks (priority 1) - most critical for homography"""
        return self.get_landmarks_by_priority(1)
    
    def get_secondary_landmarks(self) -> List[Landmark]:
        """Get secondary landmarks (priority 2) - important for accuracy"""
        return self.get_landmarks_by_priority(2)
    
    def get_tertiary_landmarks(self) -> List[Landmark]:
        """Get tertiary landmarks (priority 3) - validation and refinement"""
        return self.get_landmarks_by_priority(3)
    
    def count_landmarks(self) -> Dict[str, int]:
        """Get count of landmarks by type and priority"""
        counts = {
            'total': len(self._landmarks),
            'by_priority': {1: 0, 2: 0, 3: 0},
            'by_type': {}
        }
        
        for lm in self._landmarks:
            counts['by_priority'][lm.priority] += 1
            counts['by_type'][lm.landmark_type] = counts['by_type'].get(lm.landmark_type, 0) + 1
        
        return counts
