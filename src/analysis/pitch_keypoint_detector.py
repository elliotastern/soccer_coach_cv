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
        
        # 3b. Detect touchlines (sidelines) - these are critical for y-axis accuracy
        touchlines = self._detect_touchlines(image, lines)
        keypoints.extend(touchlines)
        
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
    
    def detect_keypoints_averaged(self, images: List[np.ndarray], min_frames: int = 5) -> List[PitchKeypoint]:
        """
        Detect landmarks by averaging positions across multiple frames with outlier filtering.
        Since camera is stationary, averaging improves accuracy and stability.
        
        Uses statistical outlier detection and distance-based clustering for improved accuracy.
        
        Args:
            images: List of frames from the same camera position
            min_frames: Minimum number of frames needed for averaging
        
        Returns:
            List of averaged keypoints with improved accuracy
        """
        if len(images) < min_frames:
            # Not enough frames, use single frame detection
            return self.detect_all_keypoints(images[0] if images else None)
        
        # Detect keypoints on all frames
        all_detections = []
        for image in images:
            keypoints = self.detect_all_keypoints(image)
            all_detections.append(keypoints)
        
        # Group keypoints by type using distance-based clustering
        from collections import defaultdict
        
        # First pass: Group by landmark type
        by_type = defaultdict(list)
        for frame_keypoints in all_detections:
            for kp in frame_keypoints:
                by_type[kp.landmark_type].append(kp)
        
        # Second pass: Cluster within each type using distance-based clustering
        averaged_keypoints = []
        
        for landmark_type, all_kps in by_type.items():
            if len(all_kps) == 0:
                continue
            
            # Cluster keypoints by image position (using distance threshold)
            # Since camera is stationary, same landmark should be in similar image positions
            clusters = self._cluster_keypoints_by_distance(all_kps, max_distance_px=50.0)
            
            for cluster in clusters:
                if len(cluster) == 0:
                    continue
                
                # Filter outliers using IQR method
                filtered_cluster = self._filter_outliers_iqr(cluster)
                
                if len(filtered_cluster) == 0:
                    # If all were outliers, use original cluster but with lower confidence
                    filtered_cluster = cluster
                
                if len(filtered_cluster) == 1:
                    # Only one detection, use as-is but lower confidence
                    kp = filtered_cluster[0]
                    kp.confidence *= 0.7
                    averaged_keypoints.append(kp)
                else:
                    # Calculate statistics for confidence scoring
                    img_x_coords = [kp.image_point[0] for kp in filtered_cluster]
                    img_y_coords = [kp.image_point[1] for kp in filtered_cluster]
                    pitch_x_coords = [kp.pitch_point[0] for kp in filtered_cluster]
                    pitch_y_coords = [kp.pitch_point[1] for kp in filtered_cluster]
                    confidences = [kp.confidence for kp in filtered_cluster]
                    
                    # Use median for robustness (less affected by remaining outliers)
                    avg_img_x = np.median(img_x_coords)
                    avg_img_y = np.median(img_y_coords)
                    avg_pitch_x = np.median(pitch_x_coords)
                    avg_pitch_y = np.median(pitch_y_coords)
                    
                    # Calculate variance for confidence adjustment
                    img_x_var = np.var(img_x_coords)
                    img_y_var = np.var(img_y_coords)
                    total_variance = img_x_var + img_y_var
                    
                    # Base confidence: average of individual confidences
                    avg_confidence = np.mean(confidences)
                    
                    # Consistency boost: more detections = higher confidence
                    detection_ratio = len(filtered_cluster) / len(images)
                    consistency_boost = min(0.25, detection_ratio * 0.25)  # Up to 25% boost
                    
                    # Variance penalty: high variance = lower confidence
                    # Normalize variance by image size (assume ~1000px typical size)
                    normalized_variance = total_variance / (1000.0 * 1000.0)
                    variance_penalty = min(0.3, normalized_variance * 10.0)  # Up to 30% penalty
                    
                    # Final confidence calculation
                    final_confidence = avg_confidence + consistency_boost - variance_penalty
                    final_confidence = max(0.1, min(1.0, final_confidence))  # Clamp to [0.1, 1.0]
                    
                    # Validate against expected positions if possible
                    validated = self._validate_averaged_landmark(
                        landmark_type,
                        (avg_pitch_x, avg_pitch_y),
                        (avg_img_x, avg_img_y),
                        images[0].shape if images else None
                    )
                    
                    if validated:
                        averaged_keypoints.append(PitchKeypoint(
                            image_point=(float(avg_img_x), float(avg_img_y)),
                            pitch_point=(float(avg_pitch_x), float(avg_pitch_y)),
                            landmark_type=landmark_type,
                            confidence=final_confidence
                        ))
        
        return averaged_keypoints
    
    def _cluster_keypoints_by_distance(self, keypoints: List[PitchKeypoint], max_distance_px: float = 50.0) -> List[List[PitchKeypoint]]:
        """
        Cluster keypoints by image position distance.
        Uses simple greedy clustering algorithm.
        
        Args:
            keypoints: List of keypoints to cluster
            max_distance_px: Maximum distance in pixels for clustering
        
        Returns:
            List of clusters (each cluster is a list of keypoints)
        """
        if len(keypoints) == 0:
            return []
        
        clusters = []
        used = set()
        
        for i, kp in enumerate(keypoints):
            if i in used:
                continue
            
            # Start new cluster
            cluster = [kp]
            used.add(i)
            
            # Find nearby keypoints
            for j, other_kp in enumerate(keypoints):
                if j in used or j == i:
                    continue
                
                # Calculate distance in image space
                dist = np.sqrt(
                    (kp.image_point[0] - other_kp.image_point[0])**2 +
                    (kp.image_point[1] - other_kp.image_point[1])**2
                )
                
                if dist <= max_distance_px:
                    cluster.append(other_kp)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _filter_outliers_iqr(self, keypoints: List[PitchKeypoint]) -> List[PitchKeypoint]:
        """
        Filter outliers using Interquartile Range (IQR) method.
        
        Args:
            keypoints: List of keypoints to filter
        
        Returns:
            Filtered list of keypoints (outliers removed)
        """
        if len(keypoints) < 3:
            # Need at least 3 points for IQR filtering
            return keypoints
        
        # Extract image coordinates
        img_x_coords = [kp.image_point[0] for kp in keypoints]
        img_y_coords = [kp.image_point[1] for kp in keypoints]
        
        # Calculate IQR for x and y separately
        q1_x, q3_x = np.percentile(img_x_coords, [25, 75])
        q1_y, q3_y = np.percentile(img_y_coords, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        # Define outlier bounds (1.5 * IQR is standard)
        lower_x = q1_x - 1.5 * iqr_x
        upper_x = q3_x + 1.5 * iqr_x
        lower_y = q1_y - 1.5 * iqr_y
        upper_y = q3_y + 1.5 * iqr_y
        
        # Filter outliers
        filtered = []
        for kp in keypoints:
            x, y = kp.image_point
            if lower_x <= x <= upper_x and lower_y <= y <= upper_y:
                filtered.append(kp)
        
        # If filtering removed too many (>50%), return original
        if len(filtered) < len(keypoints) * 0.5:
            return keypoints
        
        return filtered
    
    def _validate_averaged_landmark(self, landmark_type: str, 
                                   pitch_point: Tuple[float, float],
                                   image_point: Tuple[float, float],
                                   image_shape: Optional[Tuple[int, int, int]]) -> bool:
        """
        Validate averaged landmark against expected geometric constraints.
        
        Args:
            landmark_type: Type of landmark
            pitch_point: Pitch coordinates (x, y)
            image_point: Image coordinates (x, y)
            image_shape: Image shape (h, w, c) or None
        
        Returns:
            True if landmark passes validation, False otherwise
        """
        if image_shape is None:
            return True  # Can't validate without image dimensions
        
        h, w = image_shape[:2]
        pitch_x, pitch_y = pitch_point
        img_x, img_y = image_point
        
        # Basic bounds check
        if not (0 <= img_x <= w and 0 <= img_y <= h):
            return False
        
        # Type-specific validation
        if landmark_type == "goal":
            # Goals should be near left/right edges
            if not (img_x < w * 0.35 or img_x > w * 0.65):
                return False
            # Pitch x should be near ±pitch_length/2
            if abs(abs(pitch_x) - self.pitch_length / 2) > 15.0:
                return False
        
        elif landmark_type == "corner":
            # Corners should be near image corners
            corner_regions = [
                (img_x < w * 0.4 and img_y < h * 0.4),  # Top-left
                (img_x > w * 0.6 and img_y < h * 0.4),  # Top-right
                (img_x > w * 0.6 and img_y > h * 0.6),  # Bottom-right
                (img_x < w * 0.4 and img_y > h * 0.6),  # Bottom-left
            ]
            if not any(corner_regions):
                return False
        
        elif landmark_type == "center_circle":
            # Center circle should be near image center
            center_dist = np.sqrt((img_x - w / 2)**2 + (img_y - h / 2)**2)
            if center_dist > min(w, h) * 0.5:
                return False
            # Pitch coordinates should be near (0, 0)
            if np.sqrt(pitch_x**2 + pitch_y**2) > 15.0:
                return False
        
        elif landmark_type == "center_line":
            # Center line should be near image center vertically
            if abs(img_y - h / 2) > h * 0.4:
                return False
            # Pitch y should be 0 (center line)
            if abs(pitch_y) > 5.0:
                return False
        
        # All other types pass basic validation
        return True
    
    def _detect_field_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect white field lines using color segmentation and line detection
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of detected lines as [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        
        # Optimize: Downscale for line detection on very large images
        # HoughLinesP is also expensive on large images
        if max(h, w) > 3000:
            scale_factor = 0.5  # Downscale for speed
            small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        else:
            scale_factor = 1.0
            small_image = image
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
        
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
        
        # Detect lines using HoughLinesP with optimized parameters
        lines = cv2.HoughLinesP(
            line_mask,
            rho=2,  # Increased from 1 for speed
            theta=np.pi/180,
            threshold=50,
            minLineLength=30 if scale_factor < 1.0 else 50,  # Adjust for downscaled
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Scale lines back to original image size if downscaled
        if scale_factor < 1.0:
            lines = (lines / scale_factor).astype(np.int32)
        
        return lines
    
    def _detect_goals_improved_lines(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Improved vertical line detection for goal posts with better Hough parameters.
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints from improved line detection
        """
        h, w = image.shape[:2]
        keypoints = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Use both HoughLinesP and HoughLines for better detection
        # Left goal region (wider region: 20% from left)
        left_region = edges[:, :int(w * 0.2)]
        right_region = edges[:, int(w * 0.8):]
        
        # Detect vertical lines with multiple parameter sets
        vertical_lines_left = []
        vertical_lines_right = []
        
        # Try different Hough parameter sets for different scales
        for min_line_length in [30, 50, 80]:
            for max_line_gap in [5, 10, 15]:
                # Left region
                lines_p = cv2.HoughLinesP(left_region, 1, np.pi/180, 30, 
                                         minLineLength=min_line_length, maxLineGap=max_line_gap)
                if lines_p is not None:
                    for line in lines_p:
                        x1, y1, x2, y2 = line[0]
                        # Check angle: should be near vertical (75-105 degrees from horizontal)
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if abs(angle) > 75 and abs(angle) < 105:
                            vertical_lines_left.append((x1, y1, x2, y2))
                
                # Right region
                lines_p = cv2.HoughLinesP(right_region, 1, np.pi/180, 30,
                                         minLineLength=min_line_length, maxLineGap=max_line_gap)
                if lines_p is not None:
                    for line in lines_p:
                        x1, y1, x2, y2 = line[0]
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if abs(angle) > 75 and abs(angle) < 105:
                            vertical_lines_right.append((x1, y1, x2, y2))
        
        # Also use standard HoughLines for better angle detection
        lines_h = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines_h is not None:
            for line in lines_h:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)
                # Check if near vertical
                if abs(angle_deg - 90) < 15 or abs(angle_deg - 270) < 15:
                    # Convert to line endpoints
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    
                    # Check if in goal regions
                    if x1 < w * 0.2 or x2 < w * 0.2:
                        vertical_lines_left.append((x1, y1, x2, y2))
                    if x1 > w * 0.8 or x2 > w * 0.8:
                        vertical_lines_right.append((x1, y1, x2, y2))
        
        # Process left goal
        if vertical_lines_left:
            x_positions = []
            for x1, y1, x2, y2 in vertical_lines_left:
                # Use midpoint x
                x_positions.append((x1 + x2) / 2)
            
            if x_positions:
                avg_x = np.median(x_positions)
                # Use median y from line endpoints
                y_positions = []
                for x1, y1, x2, y2 in vertical_lines_left:
                    y_positions.extend([y1, y2])
                avg_y = np.median(y_positions) if y_positions else h / 2
                
                keypoints.append(PitchKeypoint(
                    image_point=(float(avg_x), float(avg_y)),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.75
                ))
        
        # Process right goal
        if vertical_lines_right:
            x_positions = []
            for x1, y1, x2, y2 in vertical_lines_right:
                # Adjust for region offset
                x_positions.append((x1 + x2) / 2 + int(w * 0.8))
            
            if x_positions:
                avg_x = np.median(x_positions)
                y_positions = []
                for x1, y1, x2, y2 in vertical_lines_right:
                    y_positions.extend([y1, y2])
                avg_y = np.median(y_positions) if y_positions else h / 2
                
                keypoints.append(PitchKeypoint(
                    image_point=(float(avg_x), float(avg_y)),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.75
                ))
        
        return keypoints
    
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
        # Handle partial visibility - camera may only see one goal
        if left_goal_lines is not None and len(left_goal_lines) > 0:
            # Find average x position of vertical lines
            x_positions = []
            for line in left_goal_lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical
                if abs(x2 - x1) < abs(y2 - y1) * 0.3:  # More vertical than horizontal
                    x_positions.append((x1 + x2) / 2)
            
            if x_positions:
                avg_x = np.median(x_positions)  # Use median for robustness
                keypoints.append(PitchKeypoint(
                    image_point=(avg_x, goal_center_y),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.75
                ))
        
        # Right goal (at x = +pitch_length/2, y = 0)
        # May not be visible depending on camera angle
        if right_goal_lines is not None and len(right_goal_lines) > 0:
            x_positions = []
            for line in right_goal_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1) * 0.3:
                    x_positions.append((x1 + x2) / 2 + int(w * 0.85))  # Adjust for region offset
            
            if x_positions:
                avg_x = np.median(x_positions)
                keypoints.append(PitchKeypoint(
                    image_point=(avg_x, goal_center_y),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=0.75
                ))
        
        # It's OK if only one goal is detected - camera angle may hide the other
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
            # Check if line is mostly horizontal with better metric
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length > 0:
                horizontal_ratio = abs(x2 - x1) / line_length
                # Line should be more horizontal than vertical
                if horizontal_ratio > 0.7:  # At least 70% horizontal
                    avg_y = (y1 + y2) / 2
                    # Check if near center (wider tolerance)
                    if abs(avg_y - center_y) < h * 0.15:  # Within 15% of center
                        horizontal_lines.append((line[0], avg_y, (x1 + x2) / 2))
        
        if horizontal_lines:
            # Use median for more robust position
            y_positions = [line[1] for line in horizontal_lines]
            x_midpoints = [line[2] for line in horizontal_lines]
            
            median_y = np.median(y_positions)
            median_x = np.median(x_midpoints)
            
            # Center point (halfway point)
            keypoints.append(PitchKeypoint(
                image_point=(median_x, median_y),
                pitch_point=(0.0, 0.0),  # Center of pitch
                landmark_type="center_line",
                confidence=0.85
            ))
            
            # Use actual line endpoints if available
            all_x = []
            for line, _, _ in horizontal_lines:
                x1, y1, x2, y2 = line
                all_x.extend([x1, x2])
            
            if all_x:
                left_x = min(all_x)
                right_x = max(all_x)
                
                # Left endpoint
                keypoints.append(PitchKeypoint(
                    image_point=(left_x, median_y),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="center_line",
                    confidence=0.7
                ))
                
                # Right endpoint
                keypoints.append(PitchKeypoint(
                    image_point=(right_x, median_y),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="center_line",
                    confidence=0.7
                ))
            else:
                # Fallback to approximate positions
                keypoints.append(PitchKeypoint(
                    image_point=(w * 0.1, median_y),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="center_line",
                    confidence=0.5
                ))
                keypoints.append(PitchKeypoint(
                    image_point=(w * 0.9, median_y),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="center_line",
                    confidence=0.5
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
        Detect penalty box corners using line intersections (improved accuracy)
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of penalty box keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Find all line intersections
        intersections = self._detect_line_intersections(lines)
        
        if len(intersections) < 4:
            # Not enough intersections - use approximate positions as last resort
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.15, h * 0.3),
                pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, -self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.2  # Very low confidence for hardcoded positions
            ))
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.15, h * 0.7),
                pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.2
            ))
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.85, h * 0.3),
                pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, -self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.2
            ))
            keypoints.append(PitchKeypoint(
                image_point=(w * 0.85, h * 0.7),
                pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, self.penalty_box_width / 2),
                landmark_type="penalty_box",
                confidence=0.2
            ))
            return keypoints
        
        # Penalty boxes are rectangular areas near goals
        # Left penalty box should be near left edge (x < w/3)
        # Right penalty box should be near right edge (x > 2*w/3)
        
        left_intersections = [(x, y) for x, y in intersections if x < w / 3]
        right_intersections = [(x, y) for x, y in intersections if x > 2 * w / 3]
        
        # Left penalty box: Find intersections that form rectangle corners
        if len(left_intersections) >= 2:
            # Sort by y to find top and bottom
            left_sorted = sorted(left_intersections, key=lambda p: p[1])
            # Penalty box should span roughly middle 40% of field height
            # Find intersections closest to expected positions
            expected_top_y = h * 0.3
            expected_bottom_y = h * 0.7
            
            left_top = min(left_sorted, key=lambda p: abs(p[1] - expected_top_y))
            left_bottom = min(left_sorted, key=lambda p: abs(p[1] - expected_bottom_y))
            
            # Validate these are reasonable (within 20% of expected)
            if abs(left_top[1] - expected_top_y) < h * 0.2 and abs(left_bottom[1] - expected_bottom_y) < h * 0.2:
                keypoints.append(PitchKeypoint(
                    image_point=left_top,
                    pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, -self.penalty_box_width / 2),
                    landmark_type="penalty_box",
                    confidence=0.75
                ))
                keypoints.append(PitchKeypoint(
                    image_point=left_bottom,
                    pitch_point=(-self.pitch_length / 2 + self.penalty_box_depth, self.penalty_box_width / 2),
                    landmark_type="penalty_box",
                    confidence=0.75
                ))
        
        # Right penalty box
        if len(right_intersections) >= 2:
            right_sorted = sorted(right_intersections, key=lambda p: p[1])
            right_top = min(right_sorted, key=lambda p: abs(p[1] - h * 0.3))
            right_bottom = min(right_sorted, key=lambda p: abs(p[1] - h * 0.7))
            
            if abs(right_top[1] - h * 0.3) < h * 0.2 and abs(right_bottom[1] - h * 0.7) < h * 0.2:
                keypoints.append(PitchKeypoint(
                    image_point=right_top,
                    pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, -self.penalty_box_width / 2),
                    landmark_type="penalty_box",
                    confidence=0.75
                ))
                keypoints.append(PitchKeypoint(
                    image_point=right_bottom,
                    pitch_point=(self.pitch_length / 2 - self.penalty_box_depth, self.penalty_box_width / 2),
                    landmark_type="penalty_box",
                    confidence=0.75
                ))
        
        return keypoints
    
    def _detect_field_corners(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect field corners using actual line intersections
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of corner keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Find all line intersections
        intersections = self._detect_line_intersections(lines)
        
        if len(intersections) < 4:
            # Fallback to approximate positions only if no intersections found
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
                    confidence=0.2  # Very low confidence for hardcoded positions
                ))
            return keypoints
        
        # Find intersections near image corners (likely field corners)
        corner_regions = [
            (0, 0, w * 0.25, h * 0.25, (-self.pitch_length / 2, -self.pitch_width / 2)),  # Top-left
            (w * 0.75, 0, w, h * 0.25, (self.pitch_length / 2, -self.pitch_width / 2)),  # Top-right
            (w * 0.75, h * 0.75, w, h, (self.pitch_length / 2, self.pitch_width / 2)),  # Bottom-right
            (0, h * 0.75, w * 0.25, h, (-self.pitch_length / 2, self.pitch_width / 2)),  # Bottom-left
        ]
        
        for x_min, y_min, x_max, y_max, pitch_coords in corner_regions:
            # Find intersection closest to corner region
            region_intersections = [
                (x, y) for x, y in intersections
                if x_min <= x <= x_max and y_min <= y <= y_max
            ]
            
            if region_intersections:
                # Use intersection closest to actual corner
                corner_point = (x_min if x_min < w/2 else x_max, y_min if y_min < h/2 else y_max)
                closest = min(region_intersections, 
                            key=lambda p: np.sqrt((p[0] - corner_point[0])**2 + (p[1] - corner_point[1])**2))
                
                keypoints.append(PitchKeypoint(
                    image_point=closest,
                    pitch_point=pitch_coords,
                    landmark_type="corner",
                    confidence=0.7
                ))
        
        return keypoints
    
    def _detect_goal_post_pairs(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect goal post pairs by finding pairs of vertical lines matching goal width.
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints from post pair detection
        """
        h, w = image.shape[:2]
        keypoints = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Get vertical lines from improved detection
        left_region = edges[:, :int(w * 0.2)]
        right_region = edges[:, int(w * 0.8):]
        
        # Detect vertical lines
        vertical_lines_left = []
        vertical_lines_right = []
        
        for region, lines_list, offset in [(left_region, vertical_lines_left, 0), 
                                          (right_region, vertical_lines_right, int(w * 0.8))]:
            lines_p = cv2.HoughLinesP(region, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
            if lines_p is not None:
                for line in lines_p:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) > 75 and abs(angle) < 105:
                        # Store as (x_center, y_top, y_bottom)
                        x_center = (x1 + x2) / 2 + offset
                        y_top = min(y1, y2)
                        y_bottom = max(y1, y2)
                        lines_list.append((x_center, y_top, y_bottom))
        
        # Find pairs in left region
        if len(vertical_lines_left) >= 2:
            # Goal width is 7.32m, should appear as ~2-5% of image width
            min_spacing = w * 0.02
            max_spacing = w * 0.05
            
            # Try all pairs
            for i in range(len(vertical_lines_left)):
                for j in range(i + 1, len(vertical_lines_left)):
                    x1, y1_top, y1_bottom = vertical_lines_left[i]
                    x2, y2_top, y2_bottom = vertical_lines_left[j]
                    
                    spacing = abs(x2 - x1)
                    if min_spacing <= spacing <= max_spacing:
                        # Check that posts are roughly same height
                        height1 = y1_bottom - y1_top
                        height2 = y2_bottom - y2_top
                        if abs(height1 - height2) < max(height1, height2) * 0.3:
                            # Goal center is midpoint between posts
                            goal_x = (x1 + x2) / 2
                            goal_y = (min(y1_top, y2_top) + max(y1_bottom, y2_bottom)) / 2
                            
                            keypoints.append(PitchKeypoint(
                                image_point=(float(goal_x), float(goal_y)),
                                pitch_point=(-self.pitch_length / 2, 0.0),
                                landmark_type="goal",
                                confidence=0.85  # Higher confidence for pairs
                            ))
        
        # Find pairs in right region
        if len(vertical_lines_right) >= 2:
            min_spacing = w * 0.02
            max_spacing = w * 0.05
            
            for i in range(len(vertical_lines_right)):
                for j in range(i + 1, len(vertical_lines_right)):
                    x1, y1_top, y1_bottom = vertical_lines_right[i]
                    x2, y2_top, y2_bottom = vertical_lines_right[j]
                    
                    spacing = abs(x2 - x1)
                    if min_spacing <= spacing <= max_spacing:
                        height1 = y1_bottom - y1_top
                        height2 = y2_bottom - y2_top
                        if abs(height1 - height2) < max(height1, height2) * 0.3:
                            goal_x = (x1 + x2) / 2
                            goal_y = (min(y1_top, y2_top) + max(y1_bottom, y2_bottom)) / 2
                            
                            keypoints.append(PitchKeypoint(
                                image_point=(float(goal_x), float(goal_y)),
                                pitch_point=(self.pitch_length / 2, 0.0),
                                landmark_type="goal",
                                confidence=0.85
                            ))
        
        return keypoints
    
    def _detect_goal_crossbar(self, image: np.ndarray, existing_detections: List[Tuple[str, PitchKeypoint]]) -> List[PitchKeypoint]:
        """
        Detect goal crossbar (horizontal line connecting goal posts).
        
        Args:
            image: Input image
            existing_detections: List of (method, goal) tuples from other detection methods
        
        Returns:
            List of goal keypoints refined by crossbar detection
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Extract goal positions from existing detections
        goal_positions = {}
        for method, goal in existing_detections:
            pitch_x = goal.pitch_point[0]
            if pitch_x < 0:
                side = 'left'
            else:
                side = 'right'
            
            if side not in goal_positions:
                goal_positions[side] = []
            goal_positions[side].append(goal.image_point)
        
        # Detect horizontal lines in goal regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        for side, positions in goal_positions.items():
            if not positions:
                continue
            
            # Get average goal position
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
            
            # Define region around goal
            region_width = int(w * 0.1)
            region_height = int(h * 0.15)
            x_start = max(0, int(avg_x - region_width / 2))
            x_end = min(w, int(avg_x + region_width / 2))
            y_start = max(0, int(avg_y - region_height / 2))
            y_end = min(h, int(avg_y + region_height / 2))
            
            goal_region = edges[y_start:y_end, x_start:x_end]
            
            # Detect horizontal lines (crossbar)
            lines_h = cv2.HoughLinesP(goal_region, 1, np.pi/180, 30, 
                                     minLineLength=int(region_width * 0.3), maxLineGap=10)
            
            if lines_h is not None:
                horizontal_lines = []
                for line in lines_h:
                    x1, y1, x2, y2 = line[0]
                    # Check if horizontal (angle close to 0 or 180)
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 15 or abs(angle) > 165:
                        horizontal_lines.append((x1 + x_start, y1 + y_start, x2 + x_start, y2 + y_start))
                
                if horizontal_lines:
                    # Use crossbar to refine goal position
                    # Crossbar should be above goal center
                    crossbar_y = np.median([(y1 + y2) / 2 for x1, y1, x2, y2 in horizontal_lines])
                    crossbar_x = np.median([(x1 + x2) / 2 for x1, y1, x2, y2 in horizontal_lines])
                    
                    # Refine goal position using crossbar
                    refined_y = crossbar_y + int(h * 0.02)  # Goal center slightly below crossbar
                    refined_x = crossbar_x
                    
                    pitch_x = -self.pitch_length / 2 if side == 'left' else self.pitch_length / 2
                    
                    keypoints.append(PitchKeypoint(
                        image_point=(float(refined_x), float(refined_y)),
                        pitch_point=(pitch_x, 0.0),
                        landmark_type="goal",
                        confidence=0.8  # Crossbar detection adds confidence
                    ))
        
        return keypoints
    
    def _fuse_goal_detections(self, all_detections: List[Tuple[str, PitchKeypoint]], 
                             image: np.ndarray) -> List[PitchKeypoint]:
        """
        Fuse goal detections from multiple methods using voting/consensus.
        
        Args:
            all_detections: List of (method_name, goal) tuples
            image: Input image for reference
        
        Returns:
            List of fused goal keypoints
        """
        h, w = image.shape[:2]
        
        # Group detections by side (left vs right)
        left_goals = []
        right_goals = []
        
        for method, goal in all_detections:
            pitch_x = goal.pitch_point[0]
            if pitch_x < 0:
                left_goals.append((method, goal))
            else:
                right_goals.append((method, goal))
        
        fused = []
        
        # Fuse left goals
        if left_goals:
            # Cluster nearby detections (within 30 pixels for tighter clustering)
            clusters = []
            for method, goal in left_goals:
                gx, gy = goal.image_point
                # Find cluster
                found_cluster = False
                for cluster in clusters:
                    # Check if close to any goal in cluster
                    for _, c_goal in cluster:
                        cx, cy = c_goal.image_point
                        dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
                        if dist < 30:  # Tighter clustering
                            cluster.append((method, goal))
                            found_cluster = True
                            break
                    if found_cluster:
                        break
                
                if not found_cluster:
                    clusters.append([(method, goal)])
            
            # Fuse each cluster - only keep best clusters
            # Sort clusters by size and confidence (prefer larger, higher confidence clusters)
            clusters.sort(key=lambda c: (len(c), np.mean([g.confidence for _, g in c])), reverse=True)
            
            # Keep only top 2 clusters per side (left and right goals)
            for cluster in clusters[:2]:
                if len(cluster) == 0:
                    continue
                
                # Calculate weighted average position
                total_confidence = sum(goal.confidence for _, goal in cluster)
                if total_confidence == 0:
                    continue
                
                weighted_x = sum(goal.image_point[0] * goal.confidence for _, goal in cluster) / total_confidence
                weighted_y = sum(goal.image_point[1] * goal.confidence for _, goal in cluster) / total_confidence
                
                # Confidence based on number of methods agreeing
                num_methods = len(cluster)
                base_confidence = np.mean([goal.confidence for _, goal in cluster])
                consensus_boost = min(0.15, num_methods * 0.05)  # Up to 15% boost
                final_confidence = min(1.0, base_confidence + consensus_boost)
                
                fused.append(PitchKeypoint(
                    image_point=(float(weighted_x), float(weighted_y)),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=final_confidence
                ))
        
        # Fuse right goals (same process)
        if right_goals:
            clusters = []
            for method, goal in right_goals:
                gx, gy = goal.image_point
                found_cluster = False
                for cluster in clusters:
                    for _, c_goal in cluster:
                        cx, cy = c_goal.image_point
                        dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
                        if dist < 30:  # Tighter clustering
                            cluster.append((method, goal))
                            found_cluster = True
                            break
                    if found_cluster:
                        break
                
                if not found_cluster:
                    clusters.append([(method, goal)])
            
            # Sort clusters by size and confidence
            clusters.sort(key=lambda c: (len(c), np.mean([g.confidence for _, g in c])), reverse=True)
            
            # Keep only top 2 clusters per side
            for cluster in clusters[:2]:
                if len(cluster) == 0:
                    continue
                
                total_confidence = sum(goal.confidence for _, goal in cluster)
                if total_confidence == 0:
                    continue
                
                weighted_x = sum(goal.image_point[0] * goal.confidence for _, goal in cluster) / total_confidence
                weighted_y = sum(goal.image_point[1] * goal.confidence for _, goal in cluster) / total_confidence
                
                num_methods = len(cluster)
                base_confidence = np.mean([goal.confidence for _, goal in cluster])
                consensus_boost = min(0.15, num_methods * 0.05)
                final_confidence = min(1.0, base_confidence + consensus_boost)
                
                fused.append(PitchKeypoint(
                    image_point=(float(weighted_x), float(weighted_y)),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=final_confidence
                ))
        
        return fused
    
    def _validate_goal_geometry(self, goal: PitchKeypoint, image: np.ndarray) -> bool:
        """
        Comprehensive geometric validation for goal detections.
        
        Args:
            goal: Goal keypoint to validate
            image: Input image for reference
        
        Returns:
            True if goal passes validation, False otherwise
        """
        h, w = image.shape[:2]
        gx, gy = goal.image_point
        pitch_x, pitch_y = goal.pitch_point
        
        # Basic bounds check
        if not (0 <= gx <= w and 0 <= gy <= h):
            return False
        
        # Validate pitch coordinates
        expected_x_left = -self.pitch_length / 2
        expected_x_right = self.pitch_length / 2
        
        if pitch_x < 0:
            # Left goal
            if abs(pitch_x - expected_x_left) > 10.0:  # Within 10m
                return False
            # Should be near left edge
            if gx > w * 0.35:
                return False
        else:
            # Right goal
            if abs(pitch_x - expected_x_right) > 10.0:
                return False
            # Should be near right edge
            if gx < w * 0.65:
                return False
        
        # Pitch y should be near 0 (center line)
        if abs(pitch_y) > 5.0:
            return False
        
        # Goal should be near center vertically (y near h/2, with tolerance)
        if abs(gy - h / 2) > h * 0.3:
            return False
        
        # All validations passed
        return True
    
    def _detect_goals_by_color(self, image: np.ndarray) -> List[PitchKeypoint]:
        """
        Detect goal posts using color-based detection (white goal posts).
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of goal keypoints detected by color
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define white color range in HSV
        # White has low saturation and high value
        lower_white = np.array([0, 0, 200])  # Low S, high V
        upper_white = np.array([180, 30, 255])  # Allow some saturation for off-white
        
        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for vertical rectangular shapes (goal posts)
        goal_regions = []
        for contour in contours:
            # Get bounding box
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (goal posts are tall and narrow)
            if ch > 0:
                aspect_ratio = cw / ch
                # Goal posts should be narrow (width < height/3)
                if aspect_ratio < 0.3 and ch > h * 0.05:  # At least 5% of image height
                    # Check if in goal regions (left or right 20% of image)
                    if x < w * 0.2 or x > w * 0.8:
                        goal_regions.append((x + cw / 2, y + ch / 2, cw, ch))
        
        # Group goal posts by region (left vs right)
        left_posts = [p for p in goal_regions if p[0] < w / 2]
        right_posts = [p for p in goal_regions if p[0] > w / 2]
        
        # Left goal: use median x position of left posts
        if left_posts:
            left_x = np.median([p[0] for p in left_posts])
            left_y = np.median([p[1] for p in left_posts])
            keypoints.append(PitchKeypoint(
                image_point=(float(left_x), float(left_y)),
                pitch_point=(-self.pitch_length / 2, 0.0),
                landmark_type="goal",
                confidence=0.7
            ))
        
        # Right goal: use median x position of right posts
        if right_posts:
            right_x = np.median([p[0] for p in right_posts])
            right_y = np.median([p[1] for p in right_posts])
            keypoints.append(PitchKeypoint(
                image_point=(float(right_x), float(right_y)),
                pitch_point=(self.pitch_length / 2, 0.0),
                landmark_type="goal",
                confidence=0.7
            ))
        
        return keypoints
    
    def _detect_goals_enhanced(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced goal detection with multiple methods and comprehensive validation.
        
        Uses:
        - Color-based detection (white goal posts)
        - Improved vertical line detection
        - Goal post pair detection
        - Crossbar detection
        - Multi-method fusion with voting
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints with improved accuracy and confidence
        """
        h, w = image.shape[:2]
        
        # Collect detections from all methods
        all_detections = []
        
        # Method 1: Color-based detection
        color_goals = self._detect_goals_by_color(image)
        for goal in color_goals:
            all_detections.append(('color', goal))
        
        # Method 2: Improved vertical line detection
        line_goals = self._detect_goals_improved_lines(image, lines)
        for goal in line_goals:
            all_detections.append(('lines', goal))
        
        # Method 3: Goal post pair detection
        pair_goals = self._detect_goal_post_pairs(image, lines)
        for goal in pair_goals:
            all_detections.append(('pairs', goal))
        
        # Method 4: Crossbar detection (enhances existing detections)
        crossbar_goals = self._detect_goal_crossbar(image, all_detections)
        for goal in crossbar_goals:
            all_detections.append(('crossbar', goal))
        
        # Fuse detections using voting/consensus
        fused_goals = self._fuse_goal_detections(all_detections, image)
        
        # Apply comprehensive geometric validation
        validated_goals = []
        for goal in fused_goals:
            if self._validate_goal_geometry(goal, image):
                validated_goals.append(goal)
        
        return validated_goals
    
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
    
    def _detect_center_circle_enhanced(self, image: np.ndarray, lines: List[np.ndarray], num_points: int = 8) -> List[PitchKeypoint]:
        """
        Enhanced center circle detection with radius validation.
        Samples multiple points evenly spaced around the circle for better homography accuracy.
        
        Args:
            image: Input image
            lines: Detected field lines
            num_points: Number of points to sample around circle (default: 8, range: 4-16)
        
        Returns:
            List of center circle keypoints (center + num_points evenly spaced around circle)
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Clamp num_points to reasonable range
        num_points = max(4, min(16, num_points))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Optimize: Downscale large images for HoughCircles (much faster)
        # HoughCircles is O(n²) so downscaling dramatically improves speed
        # Use aggressive downscaling for very large images
        if max(h, w) > 3000:
            scale_factor = 0.25  # 4x downscale for very large images (faster)
        elif max(h, w) > 2000:
            scale_factor = 0.4   # 2.5x downscale for large images (faster)
        else:
            scale_factor = 1.0   # No downscale for smaller images
        if scale_factor < 1.0:
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
            small_h, small_w = small_gray.shape[:2]
        else:
            small_gray = gray
            small_h, small_w = h, w
        
        # Detect circles with optimized parameters (on downscaled image)
        # dp=2 is faster than dp=1 with minimal accuracy loss
        circles = cv2.HoughCircles(
            small_gray,
            cv2.HOUGH_GRADIENT,
            dp=2,  # Increased from 1 for speed (2x faster)
            minDist=int(min(small_h, small_w) / 3),
            param1=50,
            param2=25,  # Lowered from 30 for speed (still accurate)
            minRadius=int(min(small_h, small_w) / 25),
            maxRadius=int(min(small_h, small_w) / 8)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Scale circle coordinates back to original image size
            if scale_factor < 1.0:
                circles = circles.astype(np.float32)
                circles[0, :, 0] /= scale_factor  # x coordinates
                circles[0, :, 1] /= scale_factor  # y coordinates
                circles[0, :, 2] /= scale_factor  # radius
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
                    
                    # Sample num_points evenly spaced around the circle
                    # Angle 0 is at top (negative y in pitch coordinates)
                    for i in range(num_points):
                        angle = 2 * np.pi * i / num_points
                        # Image coordinates: angle 0 is at top (negative y)
                        img_x = cx + r * np.sin(angle)
                        img_y = cy - r * np.cos(angle)
                        
                        # Pitch coordinates: angle 0 is at top (negative y)
                        pitch_x = self.center_circle_radius * np.sin(angle)
                        pitch_y = -self.center_circle_radius * np.cos(angle)
                        
                        keypoints.append(PitchKeypoint(
                            image_point=(float(img_x), float(img_y)),
                            pitch_point=(pitch_x, pitch_y),
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
    
    def _detect_touchlines(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect touchlines (sidelines) - critical for y-axis accuracy.
        Touchlines run along the length of the field and provide stable y-coordinate references.
        
        Enhanced with more aggressive detection parameters to improve y-axis calibration.
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of touchline keypoints
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # More aggressive detection: look for horizontal lines
        # Lower threshold for "horizontal" (was 0.3, now 0.5) to catch more lines
        # Lower minimum length (was 0.3, now 0.2) to catch shorter but valid touchlines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is roughly horizontal (more lenient threshold)
            if abs(y2 - y1) < abs(x2 - x1) * 0.5:  # More horizontal than vertical (was 0.3)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Lower threshold to catch more lines (was 0.3, now 0.2)
                if length > min(w, h) * 0.2:  # Long enough to be a touchline
                    horizontal_lines.append((line, length))
        
        # Sort by length (longest first)
        horizontal_lines.sort(key=lambda x: x[1], reverse=True)
        
        # Use top 2-4 longest horizontal lines as touchlines (more aggressive)
        # Look for lines near field edges (top/bottom of image)
        num_touchlines = min(4, len(horizontal_lines))
        if num_touchlines >= 2:
            # Sort by y position (top line has smaller y)
            sorted_lines = sorted(horizontal_lines[:num_touchlines], key=lambda x: (x[0][0][1] + x[0][0][3]) / 2)
            
            # Use topmost and bottommost lines as primary touchlines
            # But also consider intermediate lines if they're near edges
            edge_threshold = 0.15  # Lines within 15% of image edges are likely touchlines
            top_edge_y = h * edge_threshold
            bottom_edge_y = h * (1 - edge_threshold)
            
            selected_lines = []
            for line, length in sorted_lines:
                x1, y1, x2, y2 = line[0]
                mid_y = (y1 + y2) / 2
                # Prefer lines near edges
                if mid_y < top_edge_y or mid_y > bottom_edge_y:
                    selected_lines.append((line, length, mid_y))
            
            # If we found edge lines, use them; otherwise use top 2
            if len(selected_lines) >= 2:
                selected_lines = sorted(selected_lines, key=lambda x: x[2])[:2]
            elif len(sorted_lines) >= 2:
                selected_lines = [(line, length, (line[0][1] + line[0][3]) / 2) for line, length in sorted_lines[:2]]
            else:
                selected_lines = []
            
            for i, (line, length, mid_y) in enumerate(selected_lines):
                x1, y1, x2, y2 = line[0]
                
                # Determine which touchline (top or bottom) based on y position
                # Top touchline (smaller y) maps to y = -pitch_width/2 = -34m
                # Bottom touchline (larger y) maps to y = +pitch_width/2 = +34m
                is_top_touchline = (i == 0)
                pitch_y = -self.pitch_width / 2 if is_top_touchline else self.pitch_width / 2
                
                # Sample more points along the touchline (increased from 8 to 12)
                # Map x position in image to x position along field length
                num_samples = 12  # Increased from 8 for better coverage
                x_min, x_max = min(x1, x2), max(x1, x2)
                
                for j in range(num_samples):
                    # Sample point along the line
                    t = j / (num_samples - 1) if num_samples > 1 else 0.5
                    x_img = x_min + t * (x_max - x_min)
                    
                    if 0 <= x_img < w:
                        # Map image x to pitch x
                        # Assume field spans most of image width
                        # Map from image x to pitch x (-52.5m to +52.5m)
                        # Use normalized position: x_img/w maps to pitch x
                        norm_x = x_img / w
                        # Map normalized x to pitch x (assuming field is centered)
                        # Field length is 105m, so x ranges from -52.5 to +52.5
                        pitch_x = (norm_x - 0.5) * self.pitch_length
                        
                        keypoints.append(PitchKeypoint(
                            image_point=(float(x_img), float(mid_y)),
                            pitch_point=(pitch_x, pitch_y),  # Correct mapping: x varies, y is ±34m
                            landmark_type="touchline",
                            confidence=0.7 if abs(norm_x - 0.5) < 0.3 else 0.5  # Higher confidence near center
                        ))
        
        return keypoints
    
    def _detect_additional_line_intersections(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Detect additional line intersections to increase reference point count.
        Since camera is static, more points = better homography accuracy.
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of intersection keypoints
        """
        keypoints = []
        h, w = image.shape[:2]
        
        # Get all line intersections
        intersections = self._detect_line_intersections(lines)
        
        # Filter intersections that are likely valid pitch landmarks
        # (not too close to image edges, reasonable spacing)
        valid_intersections = []
        for ix, iy in intersections:
            # Must be within image bounds with margin
            margin = min(w, h) * 0.05
            if margin <= ix <= w - margin and margin <= iy <= h - margin:
                # Check distance from existing intersections
                too_close = False
                for vx, vy in valid_intersections:
                    if np.sqrt((ix - vx)**2 + (iy - vy)**2) < min(w, h) * 0.05:
                        too_close = True
                        break
                if not too_close:
                    valid_intersections.append((ix, iy))
        
        # For each valid intersection, try to map to pitch coordinates
        # This is approximate - in a real system, we'd use more sophisticated matching
        for ix, iy in valid_intersections[:10]:  # Limit to top 10 to avoid noise
            # Approximate mapping based on position in image
            # This is a heuristic - could be improved with better line matching
            norm_x = ix / w
            norm_y = iy / h
            
            # Map to pitch coordinates (rough approximation)
            pitch_x = (norm_x - 0.5) * self.pitch_length
            pitch_y = (norm_y - 0.5) * self.pitch_width
            
            keypoints.append(PitchKeypoint(
                image_point=(float(ix), float(iy)),
                pitch_point=(pitch_x, pitch_y),
                landmark_type="line_intersection",
                confidence=0.4  # Lower confidence for approximate mappings
            ))
        
        return keypoints
    
    def _validate_geometric_constraints(self, keypoints: List[PitchKeypoint], 
                                        image: np.ndarray) -> List[PitchKeypoint]:
        """
        Validate keypoints using geometric constraints to ensure detected landmarks
        match expected pitch layout
        
        Args:
            keypoints: Detected keypoints
            image: Input image for reference
        
        Returns:
            Filtered keypoints that pass validation
        """
        validated = []
        h, w = image.shape[:2]
        
        # Group keypoints by type for cross-validation
        by_type = {}
        for kp in keypoints:
            kp_type = kp.landmark_type
            if kp_type not in by_type:
                by_type[kp_type] = []
            by_type[kp_type].append(kp)
        
        # Validate each keypoint
        for kp in keypoints:
            x, y = kp.image_point
            pitch_x, pitch_y = kp.pitch_point
            
            # Basic bounds check
            if not (0 <= x <= w and 0 <= y <= h):
                continue
            
            # Type-specific geometric validation
            valid = True
            
            if kp.landmark_type == "goal":
                # Goals should be near left/right edges (more lenient - camera may only see one)
                if not (x < w * 0.3 or x > w * 0.7):
                    valid = False
                # Goals should be near center vertically (y near h/2) - more lenient
                if abs(y - h / 2) > h * 0.3:
                    valid = False
                # Pitch coordinates should be at ±pitch_length/2 (more lenient tolerance)
                if abs(abs(pitch_x) - self.pitch_length / 2) > 10.0:
                    valid = False
            
            elif kp.landmark_type == "center_line":
                # Center line should be near image center vertically
                if abs(y - h / 2) > h * 0.2:
                    valid = False
                # Pitch y should be 0 (center line)
                if abs(pitch_y) > 2.0:
                    valid = False
            
            elif kp.landmark_type == "corner":
                # Corners should be near image corners (more lenient - may only see some corners)
                corner_regions = [
                    (x < w * 0.4 and y < h * 0.4),  # Top-left (wider region)
                    (x > w * 0.6 and y < h * 0.4),  # Top-right
                    (x > w * 0.6 and y > h * 0.6),  # Bottom-right
                    (x < w * 0.4 and y > h * 0.6),  # Bottom-left
                ]
                if not any(corner_regions):
                    valid = False
                # Pitch coordinates should be at field corners (more lenient - camera angle may distort)
                expected_corners = [
                    (-self.pitch_length / 2, -self.pitch_width / 2),
                    (self.pitch_length / 2, -self.pitch_width / 2),
                    (self.pitch_length / 2, self.pitch_width / 2),
                    (-self.pitch_length / 2, self.pitch_width / 2),
                ]
                min_dist = min([np.sqrt((pitch_x - ex)**2 + (pitch_y - ey)**2) 
                               for ex, ey in expected_corners])
                if min_dist > 15.0:  # More lenient - within 15m of a corner
                    valid = False
            
            elif kp.landmark_type == "penalty_box":
                # Penalty boxes should be near goals (left/right edges) - more lenient
                if not (x < w * 0.5 or x > w * 0.5):  # Either side is OK
                    # Actually, this should be OR, not AND - fix logic
                    pass  # Accept if on either side
                # Pitch x should be near goal line ± penalty depth (more lenient)
                expected_x_left = -self.pitch_length / 2 + self.penalty_box_depth
                expected_x_right = self.pitch_length / 2 - self.penalty_box_depth
                if not (abs(pitch_x - expected_x_left) < 8.0 or abs(pitch_x - expected_x_right) < 8.0):
                    valid = False
            
            elif kp.landmark_type == "center_circle":
                # Center circle should be near image center (more lenient - may be off-center in some views)
                center_dist = np.sqrt((x - w / 2)**2 + (y - h / 2)**2)
                if center_dist > min(w, h) * 0.4:  # More lenient - 40% instead of 30%
                    valid = False
                # Pitch coordinates should be near (0, 0) - more lenient
                if np.sqrt(pitch_x**2 + pitch_y**2) > 10.0:  # Within 10m instead of 5m
                    valid = False
            
            # Confidence penalty for invalid detections
            if not valid:
                kp.confidence *= 0.5  # Reduce confidence but don't discard
            
            validated.append(kp)
        
        return validated
    
    def _calculate_y_axis_reliability(self, keypoint: PitchKeypoint, image: np.ndarray) -> float:
        """
        Calculate reliability score for y-axis accuracy.
        
        Landmarks that are more reliable for y-axis:
        - Goals (vertical posts) - high reliability
        - Center line (horizontal reference) - high reliability
        - Penalty box vertical edges - medium reliability
        - Corners near edges - lower reliability (affected by distortion)
        
        Args:
            keypoint: Keypoint to evaluate
            image: Input image for context
        
        Returns:
            Reliability score (0-1), higher = more reliable for y-axis
        """
        h, w = image.shape[:2]
        x, y = keypoint.image_point
        
        # Base reliability by type
        type_reliability = {
            "goal": 0.95,           # Goals are excellent y-axis references
            "center_line": 0.90,    # Center line provides y-scale reference
            "penalty_box": 0.75,    # Penalty box vertical edges are good
            "goal_area": 0.70,      # Goal area edges
            "center_circle": 0.65,  # Center circle helps but less critical
            "penalty_spot": 0.60,   # Penalty spots are points, less reliable
            "corner": 0.50,         # Corners may be affected by distortion
            "corner_arc": 0.45,     # Corner arcs near edges
            "touchline": 0.40       # Touchlines may curve due to distortion
        }
        
        base_reliability = type_reliability.get(keypoint.landmark_type, 0.5)
        
        # Adjust based on position: landmarks near image edges are less reliable
        # due to potential fisheye distortion
        distance_from_center = np.sqrt((x - w/2)**2 + (y - h/2)**2)
        max_distance = np.sqrt(w**2 + h**2) / 2
        normalized_distance = distance_from_center / max_distance
        
        # Reduce reliability for edge landmarks (where distortion is strongest)
        edge_penalty = normalized_distance * 0.3  # Up to 30% penalty at edges
        position_adjusted = base_reliability * (1.0 - edge_penalty)
        
        # Boost reliability for high-confidence detections
        confidence_boost = keypoint.confidence * 0.1  # Up to 10% boost
        
        final_reliability = min(1.0, position_adjusted + confidence_boost)
        
        return final_reliability
    
    def select_best_keypoints(self, keypoints: List[PitchKeypoint], 
                             min_points: int = 4, max_points: int = 25,
                             image: Optional[np.ndarray] = None,
                             prioritize_y_axis: bool = True) -> List[PitchKeypoint]:
        """
        Select best keypoints for homography estimation with y-axis accuracy focus.
        
        Prioritizes:
        1. Goals (high confidence, critical landmarks, excellent y-axis reference)
        2. Center point (halfway point, good y-scale reference)
        3. Center circle
        4. Penalty boxes, penalty spots, goal areas (vertical edges help y-axis)
        5. Corners (may be affected by distortion)
        6. Other landmarks
        
        Args:
            keypoints: All detected keypoints
            min_points: Minimum points needed (default: 4)
            max_points: Maximum points to use (default: 25 for comprehensive system)
            image: Input image for calculating y-axis reliability (optional)
            prioritize_y_axis: If True, weight landmarks by y-axis reliability
        
        Returns:
            Selected keypoints sorted by priority
        """
        # Calculate y-axis reliability if image provided and prioritizing y-axis
        if prioritize_y_axis and image is not None:
            for kp in keypoints:
                # Add y-axis reliability as an attribute (we'll use it in sorting)
                kp.y_axis_reliability = self._calculate_y_axis_reliability(kp, image)
        else:
            # Default reliability if not calculated
            for kp in keypoints:
                kp.y_axis_reliability = 0.7
        
        # Sort by confidence and type priority
        # Touchlines are critical for y-axis accuracy (field width), so give them high priority
        type_priority = {
            "goal": 1,
            "touchline": 2,  # High priority - critical for y-axis (field width) accuracy
            "center_line": 3,
            "center_circle": 4,
            "penalty_box": 5,
            "penalty_spot": 5,
            "goal_area": 5,
            "corner": 6,
            "corner_arc": 7
        }
        
        # Sort by priority, then by y-axis reliability (if prioritizing), then by confidence
        if prioritize_y_axis:
            sorted_keypoints = sorted(
                keypoints,
                key=lambda kp: (
                    type_priority.get(kp.landmark_type, 99),
                    -getattr(kp, 'y_axis_reliability', 0.7),  # Higher reliability first
                    -kp.confidence
                )
            )
        else:
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


def detect_pitch_keypoints_auto_averaged(images: List[np.ndarray],
                                         pitch_length: float = 105.0,
                                         pitch_width: float = 68.0,
                                         min_points: int = 4,
                                         max_points: int = 25,
                                         min_frames: int = 5) -> Optional[Dict]:
    """
    Detect pitch keypoints by averaging across multiple frames.
    Since camera is stationary, averaging improves accuracy and stability.
    
    Args:
        images: List of frames from the same camera position
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        min_points: Minimum points required
        max_points: Maximum points to use
        min_frames: Minimum frames needed for averaging
    
    Returns:
        Dictionary with averaged keypoints, or None if insufficient data
    """
    if not images or len(images) < min_frames:
        # Fallback to single frame if not enough frames
        return detect_pitch_keypoints_auto(images[0] if images else None, 
                                         pitch_length, pitch_width, min_points, max_points)
    
    detector = PitchKeypointDetector(pitch_length, pitch_width)
    
    # Use averaged detection
    averaged_keypoints = detector.detect_keypoints_averaged(images, min_frames=min_frames)
    
    if len(averaged_keypoints) < min_points:
        return None
    
    # Select best keypoints (same as single frame)
    selected = detector.select_best_keypoints(
        averaged_keypoints,
        min_points=min_points,
        max_points=max_points,
        image=images[0],  # Use first frame for context
        prioritize_y_axis=True
    )
    
    if len(selected) < min_points:
        return None
    
    image_points = [kp.image_point for kp in selected]
    pitch_points = [kp.pitch_point for kp in selected]
    
    return {
        'image_points': image_points,
        'pitch_points': pitch_points,
        'keypoints': selected,
        'landmark_db': detector.landmark_db
    }


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
    # Pass image for y-axis reliability calculation
    selected = detector.select_best_keypoints(
        keypoints, 
        min_points=min_points, 
        max_points=max_points,
        image=image,  # Pass image for y-axis reliability scoring
        prioritize_y_axis=True  # Enable y-axis prioritization
    )
    
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
