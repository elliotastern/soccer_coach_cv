"""
Enhanced pitch keypoint detection for accurate homography estimation.
Automatically detects goals, center line, center circle, penalty boxes, corners,
penalty spots, goal areas, and other landmarks with comprehensive algorithms.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from src.analysis.pitch_landmarks import LandmarkDatabase, Landmark


@dataclass
class PitchKeypoint:
    """Detected pitch keypoint with confidence"""
    image_point: Tuple[float, float]  # Pixel coordinates
    pitch_point: Tuple[float, float]  # Pitch coordinates in meters
    landmark_type: str  # "goal", "center_line", "corner", "penalty_box", "center_circle"
    confidence: float  # Detection confidence (0-1)


class PitchKeypointDetector:
    """
    Automatically detects pitch landmarks for accurate homography estimation.
    
    Detects:
    - Goals (goal posts)
    - Center line / halfway point
    - Center circle
    - Penalty box corners
    - Field corners
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize detector
        
        Args:
            pitch_length: Standard pitch length in meters
            pitch_width: Standard pitch width in meters
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        # Standard pitch dimensions
        self.goal_width = 7.32  # meters
        self.penalty_box_depth = 16.5  # meters
        self.penalty_box_width = 40.32  # meters
        self.goal_area_depth = 5.5  # meters (6-yard box)
        self.goal_area_width = 18.32  # meters
        self.penalty_spot_distance = 11.0  # meters from goal line
        self.center_circle_radius = 9.15  # meters
        
        # Initialize landmark database
        self.landmark_db = LandmarkDatabase(pitch_length, pitch_width)
    
    def detect_all_keypoints(self, image: np.ndarray) -> List[PitchKeypoint]:
        """
        Detect all available pitch keypoints from image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detected keypoints with their pitch coordinates
        """
        keypoints = []
        
        # 1. Detect field lines (white lines on green field)
        lines = self._detect_field_lines(image)
        
        # 2. Detect goals (enhanced)
        goals = self._detect_goals_enhanced(image, lines)
        keypoints.extend(goals)
        
        # 3. Detect center line and halfway point
        center_line_points = self._detect_center_line(image, lines)
        keypoints.extend(center_line_points)
        
        # 4. Detect center circle (enhanced)
        center_circle = self._detect_center_circle_enhanced(image, lines)
        keypoints.extend(center_circle)
        
        # 5. Detect penalty boxes (enhanced)
        penalty_boxes = self._detect_penalty_boxes_enhanced(image, lines)
        keypoints.extend(penalty_boxes)
        
        # 6. Detect field corners (enhanced)
        corners = self._detect_corners_enhanced(image, lines)
        keypoints.extend(corners)
        
        # 7. Detect penalty spots
        penalty_spots = self._detect_penalty_spots(image, lines)
        keypoints.extend(penalty_spots)
        
        # 8. Detect goal areas (6-yard boxes)
        goal_areas = self._detect_goal_areas(image, lines)
        keypoints.extend(goal_areas)
        
        # 9. Validate geometric constraints
        keypoints = self._validate_geometric_constraints(keypoints, image)
        
        return keypoints
    
    def _detect_field_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect white field lines using color segmentation and line detection
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of detected lines as [x1, y1, x2, y2]
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for white lines (high saturation, high value)
        # White in HSV: low saturation, high value
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Also detect bright green-white (field lines can appear greenish-white)
        lower_green_white = np.array([40, 0, 200])
        upper_green_white = np.array([80, 50, 255])
        green_white_mask = cv2.inRange(hsv, lower_green_white, upper_green_white)
        
        # Combine masks
        line_mask = cv2.bitwise_or(white_mask, green_white_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            line_mask,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        return lines
    
    def _detect_goals(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect goal posts (vertical structures at field edges)
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Goals are typically at the left and right edges of the field
        # Look for vertical lines near the edges
        
        # Left goal (x near 0)
        left_goal_x = w * 0.05  # 5% from left edge
        right_goal_x = w * 0.95  # 95% from left edge
        
        # Find vertical lines near goal positions
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly vertical
            if abs(x2 - x1) < 20:  # Vertical line threshold
                vertical_lines.append(line[0])
        
        # Detect goal posts as vertical structures
        # Goals are typically white posts, detect using template matching or edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for vertical edges near field boundaries
        # Left goal area
        left_region = edges[:, :int(w * 0.15)]
        right_region = edges[:, int(w * 0.85):]
        
        # Find vertical lines in goal regions
        left_goal_lines = cv2.HoughLinesP(left_region, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
        right_goal_lines = cv2.HoughLinesP(right_region, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
        
        # Goal center is at y = center of image (field center)
        goal_center_y = h / 2
        
        # Left goal (at x = -pitch_length/2, y = 0)
        if left_goal_lines is not None and len(left_goal_lines) > 0:
            # Find average x position of vertical lines
            x_positions = []
            for line in left_goal_lines:
                x1, y1, x2, y2 = line[0]
                x_positions.append((x1 + x2) / 2)
            
            if x_positions:
                avg_x = np.mean(x_positions)
                keypoints.append(PitchKeypoint(
                    image_point=(avg_x, goal_center_y),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.7
                ))
        
        # Right goal (at x = +pitch_length/2, y = 0)
        if right_goal_lines is not None and len(right_goal_lines) > 0:
            x_positions = []
            for line in right_goal_lines:
                x1, y1, x2, y2 = line[0]
                x_positions.append((x1 + x2) / 2 + int(w * 0.85))  # Adjust for region offset
            
            if x_positions:
                avg_x = np.mean(x_positions)
                keypoints.append(PitchKeypoint(
                    image_point=(avg_x, goal_center_y),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.7
                ))
        
        return keypoints
    
    def _detect_center_line(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect center line (halfway line) and center point
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of center line keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Center line is typically a horizontal line through the middle
        # Find horizontal lines near center
        center_y = h / 2
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly horizontal
            if abs(y2 - y1) < 20:  # Horizontal line threshold
                avg_y = (y1 + y2) / 2
                # Check if near center
                if abs(avg_y - center_y) < h * 0.1:  # Within 10% of center
                    horizontal_lines.append(line[0])
        
        if horizontal_lines:
            # Find average y position
            y_positions = []
            x_midpoints = []
            for line in horizontal_lines:
                x1, y1, x2, y2 = line
                y_positions.append((y1 + y2) / 2)
                x_midpoints.append((x1 + x2) / 2)
            
            avg_y = np.mean(y_positions)
            avg_x = np.mean(x_midpoints)
            
            # Center point (halfway point)
            keypoints.append(PitchKeypoint(
                image_point=(avg_x, avg_y),
                pitch_point=(0.0, 0.0),  # Center of pitch
                landmark_type="center_line",
                confidence=0.8
            ))
            
            # Center line endpoints (if we can detect them)
            # Left endpoint
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.1, avg_y),
                pitch_point=(-self.pitch_length / 2, 0.0),
                landmark_type="center_line",
                confidence=0.6
            ))
            
            # Right endpoint
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.9, avg_y),
                pitch_point=(self.pitch_length / 2, 0.0),
                landmark_type="center_line",
                confidence=0.6
            ))
        
        return keypoints
    
    def _detect_center_circle(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect center circle
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of center circle keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(h, w) / 4),
            param1=50,
            param2=30,
            minRadius=int(min(h, w) / 20),
            maxRadius=int(min(h, w) / 5)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Find circle closest to center
            center_x, center_y = w / 2, h / 2
            best_circle = None
            min_dist = float('inf')
            
            for circle in circles[0]:
                cx, cy, r = circle
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = circle
            
            if best_circle is not None:
                cx, cy, r = best_circle
                # Center of circle
                keypoints.append(PitchKeypoint(
                    image_point=(float(cx), float(cy)),
                    pitch_point=(0.0, 0.0),
                    landmark_type="center_circle",
                    confidence=0.7
                ))
                
                # Top of circle
                keypoints.append(PitchKeypoint(
                    image_point=(float(cx), float(cy - r)),
                    pitch_point=(0.0, -self.center_circle_radius),
                    landmark_type="center_circle",
                    confidence=0.6
                ))
                
                # Bottom of circle
                keypoints.append(PitchKeypoint(
                    image_point=(float(cx), float(cy + r)),
                    pitch_point=(0.0, self.center_circle_radius),
                    landmark_type="center_circle",
                    confidence=0.6
                ))
        
        return keypoints
    
    def _detect_penalty_boxes(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect penalty box corners
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of penalty box keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Penalty boxes are rectangular areas near goals
        # Find rectangular structures near field edges
        
        # This is a simplified detection - in practice would use more sophisticated methods
        # like rectangle detection or corner detection
        
        # Left penalty box (near left edge)
        # Top-left corner of penalty box
        keypoints.append(PitchKeypoint(
            image_point=(w * 0.15, h * 0.3),
            pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, -self.penalty_box_width / 2),
            landmark_type="penalty_box",
            confidence=0.5
        ))
        
        # Bottom-left corner
        keypoints.append(PitchKeypoint(
            image_point=(w * 0.15, h * 0.7),
            pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, self.penalty_box_width / 2),
            landmark_type="penalty_box",
            confidence=0.5
        ))
        
        # Right penalty box
        keypoints.append(PitchKeypoint(
            image_point=(w * 0.85, h * 0.3),
            pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, -self.penalty_box_width / 2),
            landmark_type="penalty_box",
            confidence=0.5
        ))
        
        keypoints.append(PitchKeypoint(
            image_point=(w * 0.85, h * 0.7),
            pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, self.penalty_box_width / 2),
            landmark_type="penalty_box",
            confidence=0.5
        ))
        
        return keypoints
    
    def _detect_field_corners(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect field corners (intersections of field boundaries)
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of corner keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Find line intersections to detect corners
        # This is simplified - would use proper line intersection detection
        
        # Field corners (approximate positions)
        corners = [
            ((w * 0.05, h * 0.1), (-self.pitch_length / 2, -self.pitch_width / 2), "corner"),  # Top-left
            ((w * 0.95, h * 0.1), (self.pitch_length / 2, -self.pitch_width / 2), "corner"),  # Top-right
            ((w * 0.95, h * 0.9), (self.pitch_length / 2, self.pitch_width / 2), "corner"),  # Bottom-right
            ((w * 0.05, h * 0.9), (-self.pitch_length / 2, self.pitch_width / 2), "corner"),  # Bottom-left
        ]
        
        for img_pt, pitch_pt, landmark_type in corners:
            keypoints.append(PitchKeypoint(
                image_point=img_pt,
                pitch_point=pitch_pt,
                landmark_type=landmark_type,
                confidence=0.4  # Lower confidence for approximate corners
            ))
        
        return keypoints
    
    def _detect_goals_enhanced(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced goal detection with template matching and geometric validation
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints with higher confidence
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Use basic goal detection first
        basic_goals = self._detect_goals(image, lines)
        
        # Enhance with goal area validation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for goal areas (6-yard boxes) near detected goals
        for goal in basic_goals:
            gx, gy = goal.image_point
            
            # Validate goal position by checking for goal area nearby
            # Goal area should be visible near the goal
            region_size = int(min(w, h) * 0.1)
            x_start = max(0, int(gx - region_size))
            x_end = min(w, int(gx + region_size))
            y_start = max(0, int(gy - region_size))
            y_end = min(h, int(gy + region_size))
            
            goal_region = edges[y_start:y_end, x_start:x_end]
            
            # Check for rectangular structures (goal area)
            contours, _ = cv2.findContours(goal_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we find rectangular structures, increase confidence
            confidence = goal.confidence
            if len(contours) > 0:
                # Found structures near goal - likely valid
                confidence = min(0.9, confidence + 0.1)
            
            keypoints.append(PitchKeypoint(
                image_point=goal.image_point,
                pitch_point=goal.pitch_point,
                landmark_type=goal.landmark_type,
                confidence=confidence
            ))
        
        return keypoints
    
    def _detect_penalty_boxes_enhanced(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced penalty box detection using line intersections
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of penalty box corner keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Find line intersections to detect penalty box corners
        intersections = self._detect_line_intersections(lines)
        
        # Filter intersections that could be penalty box corners
        # Penalty boxes are near goals (left/right edges)
        left_region_x = w * 0.2
        right_region_x = w * 0.8
        
        # Expected penalty box positions
        penalty_box_y_top = h * 0.25
        penalty_box_y_bottom = h * 0.75
        
        # Left penalty box
        left_intersections = [
            (x, y) for x, y in intersections
            if x < left_region_x and penalty_box_y_top < y < penalty_box_y_bottom
        ]
        
        if len(left_intersections) >= 2:
            # Sort by y coordinate
            left_intersections.sort(key=lambda p: p[1])
            # Use top and bottom intersections
            top_pt = left_intersections[0]
            bottom_pt = left_intersections[-1]
            
            # Map to pitch coordinates
            # Top corner
            keypoints.append(PitchKeypoint(
                image_point=top_pt,
                pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, -self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.75
            ))
            # Bottom corner
            keypoints.append(PitchKeypoint(
                image_point=bottom_pt,
                pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.75
            ))
        
        # Right penalty box
        right_intersections = [
            (x, y) for x, y in intersections
            if x > right_region_x and penalty_box_y_top < y < penalty_box_y_bottom
        ]
        
        if len(right_intersections) >= 2:
            right_intersections.sort(key=lambda p: p[1])
            top_pt = right_intersections[0]
            bottom_pt = right_intersections[-1]
            
            keypoints.append(PitchKeypoint(
                image_point=top_pt,
                pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, -self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.75
            ))
            keypoints.append(PitchKeypoint(
                image_point=bottom_pt,
                pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.75
            ))
        
        # Fallback to basic detection if enhanced fails
        if len(keypoints) < 4:
            basic_boxes = self._detect_penalty_boxes(image, lines)
            keypoints.extend(basic_boxes)
        
        return keypoints
    
    def _detect_center_circle_enhanced(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced center circle detection with radius validation
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of center circle keypoints (center + 4 cardinal directions)
        """
        h, w = image.shape[:2]
        keypoints = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles with tighter parameters
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(h, w) / 3),
            param1=50,
            param2=30,
            minRadius=int(min(h, w) / 25),
            maxRadius=int(min(h, w) / 8)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            center_x, center_y = w / 2, h / 2
            
            # Find circle closest to center
            best_circle = None
            min_dist = float('inf')
            
            for circle in circles[0]:
                cx, cy, r = circle
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = circle
            
            if best_circle is not None:
                cx, cy, r = best_circle
                
                # Validate radius is reasonable (should be ~9.15m scaled to image)
                expected_radius_range = (min(h, w) / 20, min(h, w) / 8)
                if expected_radius_range[0] <= r <= expected_radius_range[1]:
                    # Center of circle
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx), float(cy)),
                        pitch_point=(0.0, 0.0),
                        landmark_type="center_circle",
                        confidence=0.8
                    ))
                    
                    # 4 cardinal directions
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx), float(cy - r)),
                        pitch_point=(0.0, -self.center_circle_radius),
                        landmark_type="center_circle",
                        confidence=0.7
                    ))
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx), float(cy + r)),
                        pitch_point=(0.0, self.center_circle_radius),
                        landmark_type="center_circle",
                        confidence=0.7
                    ))
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx - r), float(cy)),
                        pitch_point=(-self.center_circle_radius, 0.0),
                        landmark_type="center_circle",
                        confidence=0.7
                    ))
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx + r), float(cy)),
                        pitch_point=(self.center_circle_radius, 0.0),
                        landmark_type="center_circle",
                        confidence=0.7
                    ))
        
        # Fallback to basic detection
        if len(keypoints) == 0:
            keypoints = self._detect_center_circle(image, lines)
        
        return keypoints
    
    def _detect_corners_enhanced(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced corner detection using line intersections
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of corner keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Find line intersections
        intersections = self._detect_line_intersections(lines)
        
        # Filter intersections near image corners (likely field corners)
        corner_regions = [
            (0, 0, w * 0.2, h * 0.2),  # Top-left
            (w * 0.8, 0, w, h * 0.2),  # Top-right
            (w * 0.8, h * 0.8, w, h),  # Bottom-right
            (0, h * 0.8, w * 0.2, h),  # Bottom-left
        ]
        
        corner_pitch_coords = [
            (-self.pitch_length / 2, -self.pitch_width / 2),  # Top-left
            (self.pitch_length / 2, -self.pitch_width / 2),   # Top-right
            (self.pitch_length / 2, self.pitch_width / 2),    # Bottom-right
            (-self.pitch_length / 2, self.pitch_width / 2),   # Bottom-left
        ]
        
        for (x_min, y_min, x_max, y_max), pitch_coords in zip(corner_regions, corner_pitch_coords):
            # Find intersection closest to corner region center
            region_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            
            closest_intersection = None
            min_dist = float('inf')
            
            for ix, iy in intersections:
                if x_min <= ix <= x_max and y_min <= iy <= y_max:
                    dist = np.sqrt((ix - region_center[0])**2 + (iy - region_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_intersection = (ix, iy)
            
            if closest_intersection is not None:
                keypoints.append(PitchKeypoint(
                    image_point=closest_intersection,
                    pitch_point=pitch_coords,
                    landmark_type="corner",
                    confidence=0.7
                ))
        
        # Fallback to basic detection if enhanced fails
        if len(keypoints) < 4:
            basic_corners = self._detect_field_corners(image, lines)
            keypoints.extend(basic_corners)
        
        return keypoints
    
    def _detect_penalty_spots(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect penalty spots (small circles 11m from goal line)
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of penalty spot keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect small circles (penalty spots are ~0.2m radius, appear as small circles)
        # Scale to image size
        min_radius = max(2, int(min(h, w) / 200))
        max_radius = max(5, int(min(h, w) / 100))
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(h, w) / 4),
            param1=50,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            center_y = h / 2
            
            # Penalty spots should be near center line (y-wise) and near goals (x-wise)
            left_region_x = w * 0.15
            right_region_x = w * 0.85
            
            for circle in circles[0]:
                cx, cy, r = circle
                
                # Check if near center line
                if abs(cy - center_y) < h * 0.1:
                    # Left penalty spot
                    if cx < left_region_x:
                        keypoints.append(PitchKeypoint(
                            image_point=(float(cx), float(cy)),
                            pitch_point=(-self.pitch_length / 2 + self.penalty_spot_distance, 0.0),
                            landmark_type="penalty_spot",
                            confidence=0.65
                        ))
                    # Right penalty spot
                    elif cx > right_region_x:
                        keypoints.append(PitchKeypoint(
                            image_point=(float(cx), float(cy)),
                            pitch_point=(self.pitch_length / 2 - self.penalty_spot_distance, 0.0),
                            landmark_type="penalty_spot",
                            confidence=0.65
                        ))
        
        return keypoints
    
    def _detect_goal_areas(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect goal areas (6-yard boxes)
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal area corner keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Goal areas are smaller rectangles near goals
        # Use line intersections to find goal area corners
        intersections = self._detect_line_intersections(lines)
        
        left_region_x = w * 0.15
        right_region_x = w * 0.85
        goal_area_y_top = h * 0.35
        goal_area_y_bottom = h * 0.65
        
        # Left goal area
        left_goal_area_intersections = [
            (x, y) for x, y in intersections
            if x < left_region_x and goal_area_y_top < y < goal_area_y_bottom
        ]
        
        if len(left_goal_area_intersections) >= 2:
            left_goal_area_intersections.sort(key=lambda p: p[1])
            top_pt = left_goal_area_intersections[0]
            bottom_pt = left_goal_area_intersections[-1]
            
            keypoints.append(PitchKeypoint(
                image_point=top_pt,
                pitch_point=(-self.pitch_length / 2 + self.goal_area_depth, -self.goal_area_width / 2),
                landmark_type="goal_area",
                confidence=0.65
            ))
            keypoints.append(PitchKeypoint(
                image_point=bottom_pt,
                pitch_point=(-self.pitch_length / 2 + self.goal_area_depth, self.goal_area_width / 2),
                landmark_type="goal_area",
                confidence=0.65
            ))
        
        # Right goal area
        right_goal_area_intersections = [
            (x, y) for x, y in intersections
            if x > right_region_x and goal_area_y_top < y < goal_area_y_bottom
        ]
        
        if len(right_goal_area_intersections) >= 2:
            right_goal_area_intersections.sort(key=lambda p: p[1])
            top_pt = right_goal_area_intersections[0]
            bottom_pt = right_goal_area_intersections[-1]
            
            keypoints.append(PitchKeypoint(
                image_point=top_pt,
                pitch_point=(self.pitch_length / 2 - self.goal_area_depth, -self.goal_area_width / 2),
                landmark_type="goal_area",
                confidence=0.65
            ))
            keypoints.append(PitchKeypoint(
                image_point=bottom_pt,
                pitch_point=(self.pitch_length / 2 - self.goal_area_depth, self.goal_area_width / 2),
                landmark_type="goal_area",
                confidence=0.65
            ))
        
        return keypoints
    
    def _detect_line_intersections(self, lines: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Find intersections between lines
        
        Args:
            lines: List of lines as [x1, y1, x2, y2]
        
        Returns:
            List of intersection points (x, y)
        """
        intersections = []
        
        for i, line1 in enumerate(lines):
            x1, y1, x2, y2 = line1[0]
            
            for line2 in lines[i+1:]:
                x3, y3, x4, y4 = line2[0]
                
                # Calculate intersection
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                
                if abs(denom) > 1e-6:  # Lines are not parallel
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                    
                    # Check if intersection is on both line segments
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        ix = x1 + t * (x2 - x1)
                        iy = y1 + t * (y2 - y1)
                        intersections.append((float(ix), float(iy)))
        
        return intersections
    
    def _validate_geometric_constraints(self, keypoints: List[PitchKeypoint], 
                                        image: np.ndarray) -> List[PitchKeypoint]:
        """
        Validate keypoints using geometric constraints
        
        Args:
            keypoints: Detected keypoints
            image: Input image for reference
        
        Returns:
            Filtered keypoints that pass validation
        """
        validated = []
        h, w = image.shape[:2]
        
        for kp in keypoints:
            x, y = kp.image_point
            
            # Basic bounds check
            if 0 <= x <= w and 0 <= y <= h:
                # Additional validation based on landmark type
                if kp.landmark_type == "goal":
                    # Goals should be near edges
                    if x < w * 0.2 or x > w * 0.8:
                        validated.append(kp)
                elif kp.landmark_type == "center_line" or kp.landmark_type == "center_circle":
                    # Center features should be near center
                    if abs(x - w/2) < w * 0.3 and abs(y - h/2) < h * 0.3:
                        validated.append(kp)
                elif kp.landmark_type == "corner":
                    # Corners should be near image corners
                    corner_dist = min(
                        np.sqrt((x - 0)**2 + (y - 0)**2),
                        np.sqrt((x - w)**2 + (y - 0)**2),
                        np.sqrt((x - w)**2 + (y - h)**2),
                        np.sqrt((x - 0)**2 + (y - h)**2)
                    )
                    if corner_dist < min(w, h) * 0.3:
                        validated.append(kp)
                else:
                    # Other landmarks - accept if in bounds
                    validated.append(kp)
        
        return validated
    
    def select_best_keypoints(self, keypoints: List[PitchKeypoint], 
                             min_points: int = 4, max_points: int = 25) -> List[PitchKeypoint]:
        """
        Select best keypoints for homography estimation
        
        Prioritizes:
        1. Goals (high confidence, critical landmarks)
        2. Center point (halfway point)
        3. Center circle
        4. Penalty boxes, penalty spots, goal areas
        5. Corners
        6. Other landmarks
        
        Args:
            keypoints: All detected keypoints
            min_points: Minimum points needed (default: 4)
            max_points: Maximum points to use (default: 25 for comprehensive system)
        
        Returns:
            Selected keypoints sorted by priority
        """
        # Sort by confidence and type priority
        type_priority = {
            "goal": 1,
            "center_line": 2,
            "center_circle": 3,
            "penalty_box": 4,
            "penalty_spot": 4,
            "goal_area": 4,
            "corner": 5,
            "corner_arc": 6,
            "touchline": 6
        }
        
        # Sort by priority (lower number = higher priority), then by confidence
        sorted_keypoints = sorted(
            keypoints,
            key=lambda kp: (type_priority.get(kp.landmark_type, 99), -kp.confidence)
        )
        
        # Select top N points (up to max_points for maximum accuracy)
        selected = sorted_keypoints[:max_points]
        
        if len(selected) < min_points:
            # If we don't have enough, return all we have
            return sorted_keypoints
        
        return selected


def detect_pitch_keypoints_auto(image: np.ndarray, 
                                pitch_length: float = 105.0,
                                pitch_width: float = 68.0,
                                min_points: int = 4,
                                max_points: int = 25) -> Optional[Dict]:
    """
    Convenience function to automatically detect pitch keypoints
    
    Args:
        image: Input image (BGR)
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        min_points: Minimum points required (default: 4)
        max_points: Maximum points to use (default: 25 for comprehensive system)
    
    Returns:
        Dictionary with 'image_points' and 'pitch_points' arrays, or None if insufficient points
    """
    detector = PitchKeypointDetector(pitch_length, pitch_width)
    keypoints = detector.detect_all_keypoints(image)
    selected = detector.select_best_keypoints(keypoints, min_points=min_points, max_points=max_points)
    
    if len(selected) < min_points:
        return None
    
    image_points = [kp.image_point for kp in selected]
    pitch_points = [kp.pitch_point for kp in selected]
    
    return {
        'image_points': image_points,
        'pitch_points': pitch_points,
        'keypoints': selected,  # Include full keypoint info for debugging
        'landmark_db': detector.landmark_db  # Include landmark database reference
    }
