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
from src.analysis.line_clustering import merged_lines_and_intersection_keypoints

try:
    from skimage.morphology import skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


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
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, 
                 enable_zero_shot: bool = True,
                 use_semantic_segmentation: bool = False,
                 segmentation_config: Optional[Dict] = None):
        """
        Initialize detector
        
        Args:
            pitch_length: Standard pitch length in meters
            pitch_width: Standard pitch width in meters
            enable_zero_shot: Enable zero-shot detection (requires transformers library)
            use_semantic_segmentation: Enable semantic segmentation for line detection
            segmentation_config: Configuration dict for segmentation (model_path, model_type, etc.)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.enable_zero_shot = enable_zero_shot
        self.use_semantic_segmentation = use_semantic_segmentation
        self._zero_shot_detector = None
        
        # Initialize semantic segmentation if enabled
        self._line_segmenter = None
        if use_semantic_segmentation:
            try:
                from src.analysis.pitch_line_segmentation import PitchLineSegmenter
                
                # Get config values with defaults
                config = segmentation_config or {}
                model_path = config.get('model_path', None)
                model_type = config.get('model_type', 'deeplabv3')
                threshold = config.get('threshold', 0.5)
                use_pretrained = config.get('use_pretrained', True)
                device = config.get('device', None)
                
                self._line_segmenter = PitchLineSegmenter(
                    model_path=model_path,
                    model_type=model_type,
                    device=device,
                    threshold=threshold,
                    use_pretrained=use_pretrained
                )
            except ImportError as e:
                print(f"Warning: Could not import PitchLineSegmenter: {e}")
                print("Falling back to color-based line detection")
                self.use_semantic_segmentation = False
        
        # Temporal tracking for goal positions (for multi-frame fusion)
        self._goal_history = []  # List of recent goal detections
        self._max_history_size = 10  # Keep last 10 frames
        
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
    
    def detect_all_keypoints(self, image: np.ndarray, frame_buffer: Optional[List[np.ndarray]] = None) -> List[PitchKeypoint]:
        """
        Detect all available pitch keypoints from image.
        Supports temporal averaging if frame_buffer is provided (static camera).
        
        Args:
            image: Input image (BGR format)
            frame_buffer: Optional list of recent frames for temporal averaging (static camera)
        
        Returns:
            List of detected keypoints with their pitch coordinates
        """
        keypoints = []
        
        # 1. Detect field lines (white lines on green field)
        # Use frame averaging if available (static camera)
        lines = self._detect_field_lines(image, frame_buffer=frame_buffer)
        
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
        # Use frame averaging if available (static camera)
        center_circle = self._detect_center_circle_enhanced(image, lines, frame_buffer=frame_buffer)
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
        
        # 9. Virtual keypoints from merged lines (VP + DBSCAN + L_long cap L_trans)
        h, w = image.shape[:2]
        if lines is not None and len(lines) >= 4:
            try:
                _, _, intersection_pts = merged_lines_and_intersection_keypoints(
                    lines, w, h,
                    min_distance_between_keypoints=min(w, h) * 0.05,
                    extend_bounds_factor=1.5
                )
                for ix, iy in intersection_pts[:15]:
                    norm_x = ix / w
                    norm_y = iy / h
                    pitch_x = (norm_x - 0.5) * self.pitch_length
                    pitch_y = (norm_y - 0.5) * self.pitch_width
                    keypoints.append(PitchKeypoint(
                        image_point=(float(ix), float(iy)),
                        pitch_point=(pitch_x, pitch_y),
                        landmark_type="virtual_keypoint",
                        confidence=0.5
                    ))
            except Exception:
                pass
        
        # 10. Validate geometric constraints
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
    
    def _prepare_mask_for_hough(self, line_mask: np.ndarray) -> np.ndarray:
        """
        Prepare binary line mask for Hough: morphological closing then skeletonization.
        Skeletonization yields 1-pixel-wide centerlines for precise rho/theta.
        """
        # Bridge 1-2 px gaps with closing (linear/elliptical kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        if not _SKIMAGE_AVAILABLE:
            return closed
        # Skeletonize to 1-pixel centerlines
        binary = (closed > 0).astype(np.uint8)
        skel = skeletonize(binary.astype(bool))
        return (skel.astype(np.uint8)) * 255
    
    def _detect_field_lines(self, image: np.ndarray, frame_buffer: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Detect white field lines using semantic segmentation or color-based detection.
        
        If semantic segmentation is enabled, uses segmentation model to generate
        a binary mask of pitch lines, then applies Hough transform on the mask.
        Supports temporal averaging if frame_buffer is provided (static camera).
        Otherwise, falls back to color-based detection.
        
        Args:
            image: Input image (BGR)
            frame_buffer: Optional list of recent frames for temporal averaging (static camera)
        
        Returns:
            List of detected lines as [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        
        # Use semantic segmentation if enabled and available
        if self.use_semantic_segmentation and self._line_segmenter is not None:
            try:
                # Get binary mask from segmentation model
                # Use temporal averaging if available (static camera)
                if frame_buffer is not None and len(frame_buffer) >= 3:
                    line_mask = self._line_segmenter.segment_pitch_lines_averaged(frame_buffer)
                else:
                    line_mask = self._line_segmenter.segment_pitch_lines(image)
                
                # Optimize: Downscale for line detection on very large images
                if max(h, w) > 3000:
                    scale_factor = 0.5
                    small_mask = cv2.resize(line_mask, None, fx=scale_factor, fy=scale_factor)
                else:
                    scale_factor = 1.0
                    small_mask = line_mask
                
                # Prepare mask: closing + skeletonization for precise Hough (framework)
                hough_mask = self._prepare_mask_for_hough(small_mask)
                min_side = min(hough_mask.shape[0], hough_mask.shape[1])
                min_line_len = min(100, int(min_side * 0.08))  # ~100 px or 8% of short side
                # Hough: rho=1, higher minLineLength, maxLineGap 20-50 for secondary hallucination
                lines = cv2.HoughLinesP(
                    hough_mask,
                    rho=1,
                    theta=np.pi/180,
                    threshold=40,
                    minLineLength=max(30, min_line_len),
                    maxLineGap=30
                )
                
                if lines is not None and len(lines) > 0:
                    # Scale lines back to original image size if downscaled
                    if scale_factor < 1.0:
                        lines = (lines / scale_factor).astype(np.int32)
                    return lines
                else:
                    # If segmentation fails, fall through to color-based
                    pass
            except Exception as e:
                print(f"Warning: Semantic segmentation failed: {e}")
                print("Falling back to color-based line detection")
        
        # Fallback to color-based detection (original method)
        # Optimize: Downscale for line detection on very large images
        # HoughLinesP is also expensive on large images
        if max(h, w) > 3000:
            scale_factor = 0.5  # Downscale for speed
            small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        else:
            scale_factor = 1.0
            small_image = image
        
        # Simple approach: mask any shade of white vs everything else
        # Use multiple color spaces to catch all white/off-white variations
        
        # Method 1: Grayscale - simplest, catches all bright pixels
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        gray_mask = (gray > 100).astype(np.uint8) * 255  # Low threshold to catch all whites
        
        # Method 2: LAB lightness - perceptual lightness (catches off-white better)
        lab = cv2.cvtColor(small_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Lightness channel
        lab_mask = (l_channel > 120).astype(np.uint8) * 255  # Catch all light colors
        
        # Method 3: HSV - catch white with any hue (low saturation, high value)
        hsv = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
        # Very broad range: any hue, low saturation, high value = white/off-white
        lower_white = np.array([0, 0, 100])  # Very low threshold for value
        upper_white = np.array([180, 100, 255])  # High saturation tolerance
        hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all methods (OR) - if ANY method says it's white, it's white
        line_mask = cv2.bitwise_or(gray_mask, lab_mask)
        line_mask = cv2.bitwise_or(line_mask, hsv_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)
        
        # Prepare mask: closing + skeletonization for precise Hough (framework)
        hough_mask = self._prepare_mask_for_hough(line_mask)
        min_side = min(hough_mask.shape[0], hough_mask.shape[1])
        min_line_len = min(100, int(min_side * 0.08))
        # Hough: rho=1, minLineLength ~100, maxLineGap 20-50
        lines = cv2.HoughLinesP(
            hough_mask,
            rho=1,
            theta=np.pi/180,
            threshold=40,
            minLineLength=max(30, min_line_len),
            maxLineGap=30
        )
        
        if lines is None:
            return []
        
        # Scale lines back to original image size if downscaled
        if scale_factor < 1.0:
            lines = (lines / scale_factor).astype(np.int32)
        
        return lines
    
    def _detect_goals_improved_lines(self, image: np.ndarray, lines: List[np.ndarray]) -> List[PitchKeypoint]:
        """
        Enhanced goal detection using improved Hough transform with geometric validation.
        
        Uses field masking, probabilistic Hough transform, and spatial logic to detect
        goal structures (2 vertical posts + 1 horizontal crossbar).
        
        Args:
            image: Input image
            lines: Detected field lines
        
        Returns:
            List of goal keypoints from enhanced line detection
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Step 1: Field masking - detect green field, then invert to find goal posts
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Green field detection (HSV range: 35-85 for hue)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        field_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert mask to isolate non-field objects (goal posts, lines, etc.)
        non_field_mask = cv2.bitwise_not(field_mask)
        
        # Step 2: Edge detection on masked image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply mask to gray image
        masked_gray = cv2.bitwise_and(gray, non_field_mask)
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
        
        # Step 3: Enhanced Probabilistic Hough Line Transform
        # Use optimized parameters for goal detection
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if hough_lines is None:
            return []
        
        # Step 4: Geometric filtering - separate vertical and horizontal lines
        vertical_lines = []  # Goal posts
        horizontal_lines = []  # Crossbars
        
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle (0 = horizontal, 90 = vertical)
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            else:
                angle = 90.0
            
            # Normalize angle to 0-180 range
            angle = abs(angle)
            if angle > 90:
                angle = 180 - angle
            
            # Filter for vertical posts (80-100 degrees, with tolerance for perspective)
            if 75 <= angle <= 105:
                vertical_lines.append((x1, y1, x2, y2, angle))
            # Filter for horizontal crossbars (0-15 or 165-180 degrees)
            elif angle <= 15 or angle >= 165:
                horizontal_lines.append((x1, y1, x2, y2, angle))
        
        # Step 5: Spatial logic - detect goal structure (2 vertical + 1 horizontal)
        # Group vertical lines by x-position (goal posts should be close together)
        left_region_lines = [l for l in vertical_lines if l[0] < w / 2]
        right_region_lines = [l for l in vertical_lines if l[0] >= w / 2]
        
        # Find goal post pairs in each region
        def find_goal_structure(vertical_lines_list, horizontal_lines_list, pitch_x):
            """Find goal structure: 2 vertical posts connected by horizontal crossbar"""
            if len(vertical_lines_list) < 2:
                return None
            
            # Sort by x position
            sorted_vert = sorted(vertical_lines_list, key=lambda l: (l[0] + l[2]) / 2)
            
            # Find pairs of vertical lines that could be goal posts
            for i in range(len(sorted_vert) - 1):
                post1 = sorted_vert[i]
                post2 = sorted_vert[i + 1]
                
                # Calculate centers
                x1_center = (post1[0] + post1[2]) / 2
                y1_center = (post1[1] + post1[3]) / 2
                x2_center = (post2[0] + post2[2]) / 2
                y2_center = (post2[1] + post2[3]) / 2
                
                # Check if posts are reasonably spaced (goal width ~7.32m)
                post_distance = abs(x2_center - x1_center)
                if 30 < post_distance < 500:  # Reasonable goal width in pixels
                    # Check if there's a horizontal line connecting them (crossbar)
                    top_y = min(y1_center, y2_center)
                    bottom_y = max(y1_center, y2_center)
                    
                    has_crossbar = False
                    for h_line in horizontal_lines_list:
                        h_x1, h_y1, h_x2, h_y2 = h_line[:4]
                        h_x_center = (h_x1 + h_x2) / 2
                        h_y_center = (h_y1 + h_y2) / 2
                        
                        # Check if horizontal line is near top of posts
                        if (min(x1_center, x2_center) - 20 < h_x_center < max(x1_center, x2_center) + 20 and
                            top_y - 50 < h_y_center < top_y + 50):
                            has_crossbar = True
                            break
                    
                    # Return goal center (between posts)
                    goal_x = (x1_center + x2_center) / 2
                    goal_y = (y1_center + y2_center) / 2
                    
                    return (goal_x, goal_y, has_crossbar)
            
            return None
        
        # Detect left goal structure
        left_structure = find_goal_structure(left_region_lines, horizontal_lines, -self.pitch_length / 2)
        if left_structure:
            goal_x, goal_y, has_crossbar = left_structure
            confidence = 0.85 if has_crossbar else 0.75
            keypoints.append(PitchKeypoint(
                image_point=(float(goal_x), float(goal_y)),
                pitch_point=(-self.pitch_length / 2, 0.0),
                landmark_type="goal",
                confidence=confidence
            ))
        
        # Detect right goal structure
        right_structure = find_goal_structure(right_region_lines, horizontal_lines, self.pitch_length / 2)
        if right_structure:
            goal_x, goal_y, has_crossbar = right_structure
            confidence = 0.85 if has_crossbar else 0.75
            keypoints.append(PitchKeypoint(
                image_point=(float(goal_x), float(goal_y)),
                pitch_point=(self.pitch_length / 2, 0.0),
                landmark_type="goal",
                confidence=confidence
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
        
        Improved with better deduplication, outlier filtering, and stricter clustering.
        
        Args:
            all_detections: List of (method_name, goal) tuples
            image: Input image for reference
        
        Returns:
            List of fused goal keypoints (max 2-4 total: 1-2 per side)
        """
        h, w = image.shape[:2]
        
        # Step 1: Filter out obvious outliers before clustering
        # Goals should be near the left/right edges of the image
        filtered_detections = []
        for method, goal in all_detections:
            gx, gy = goal.image_point
            
            # Filter by position: goals should be in edge regions
            # Left goal: x < 30% of width OR x < 50% and pitch_x < 0
            # Right goal: x > 70% of width OR x > 50% and pitch_x > 0
            pitch_x = goal.pitch_point[0]
            is_left = pitch_x < 0
            is_right = pitch_x > 0
            
            # Position validation
            if is_left and gx > w * 0.5:
                continue  # Left goal too far right
            if is_right and gx < w * 0.5:
                continue  # Right goal too far left
            
            # Filter by confidence
            if goal.confidence < 0.5:
                continue
            
            # Filter by vertical position (goals shouldn't be at extreme top/bottom)
            if gy < h * 0.05 or gy > h * 0.95:
                continue
            
            filtered_detections.append((method, goal))
        
        if len(filtered_detections) == 0:
            return []
        
        # Step 2: Group by side
        left_goals = []
        right_goals = []
        
        for method, goal in filtered_detections:
            pitch_x = goal.pitch_point[0]
            if pitch_x < 0:
                left_goals.append((method, goal))
            else:
                right_goals.append((method, goal))
        
        fused = []
        
        # Step 3: Fuse left goals with improved clustering
        if left_goals:
            # Use larger clustering distance (100 pixels) to merge nearby detections
            # This helps merge detections from different methods that are close
            clusters = []
            for method, goal in left_goals:
                gx, gy = goal.image_point
                found_cluster = False
                
                # Try to find existing cluster
                for cluster in clusters:
                    # Check distance to cluster center
                    cluster_centers = [(g.image_point[0], g.image_point[1]) for _, g in cluster]
                    avg_x = np.mean([cx for cx, _ in cluster_centers])
                    avg_y = np.mean([cy for _, cy in cluster_centers])
                    
                    dist = np.sqrt((gx - avg_x)**2 + (gy - avg_y)**2)
                    if dist < 100:  # Larger clustering distance
                        cluster.append((method, goal))
                        found_cluster = True
                        break
                
                if not found_cluster:
                    clusters.append([(method, goal)])
            
            # Sort clusters by quality score (consensus + confidence + method priority)
            def cluster_score(cluster):
                num_methods = len(cluster)
                avg_conf = np.mean([g.confidence for _, g in cluster])
                has_zero_shot = any(m == 'zero_shot' for m, _ in cluster)
                has_color = any(m == 'color' for m, _ in cluster)
                
                # Score: method priority > consensus > confidence
                method_bonus = 3.0 if has_zero_shot else (2.0 if has_color else 1.0)
                return method_bonus * num_methods * avg_conf
            
            clusters.sort(key=cluster_score, reverse=True)
            
            # Keep only the BEST 1-2 clusters per side (not all clusters)
            # Prefer clusters with multiple methods agreeing
            best_clusters = []
            for cluster in clusters:
                if len(cluster) >= 2:  # Require at least 2 detections agreeing
                    best_clusters.append(cluster)
                elif len(best_clusters) == 0 and len(cluster) == 1:
                    # Allow single detection only if no better clusters exist
                    method, goal = cluster[0]
                    if method in ('zero_shot', 'color') and goal.confidence > 0.7:
                        best_clusters.append(cluster)
                
                if len(best_clusters) >= 2:  # Max 2 goals per side
                    break
            
            # Fuse each best cluster
            for cluster in best_clusters:
                if len(cluster) == 0:
                    continue
                
                # Calculate weighted average position, prioritizing zero-shot and color-based detections
                # Zero-shot has highest priority (semantic understanding), then color-based
                weighted_sum_x = 0
                weighted_sum_y = 0
                total_weight = 0
                
                for method, goal in cluster:
                    weight = goal.confidence
                    if method == 'zero_shot':  # Highest priority: semantic understanding
                        weight *= 3.0
                    elif method == 'color':  # Second priority: reliable in good conditions
                        weight *= 2.0
                    weighted_sum_x += goal.image_point[0] * weight
                    weighted_sum_y += goal.image_point[1] * weight
                    total_weight += weight
                
                if total_weight == 0:
                    continue
                
                weighted_x = weighted_sum_x / total_weight
                weighted_y = weighted_sum_y / total_weight
                
                # Confidence based on number of methods agreeing, boost if zero-shot or color method agrees
                num_methods = len(cluster)
                has_zero_shot = any(method == 'zero_shot' for method, _ in cluster)
                has_color = any(method == 'color' for method, _ in cluster)
                base_confidence = np.mean([goal.confidence for _, goal in cluster])
                consensus_boost = min(0.15, num_methods * 0.05)  # Up to 15% boost
                zero_shot_boost = 0.15 if has_zero_shot else 0  # Highest boost for zero-shot
                color_boost = 0.1 if has_color else 0  # Extra boost if color method agrees
                final_confidence = min(1.0, base_confidence + consensus_boost + zero_shot_boost + color_boost)
                
                fused.append(PitchKeypoint(
                    image_point=(float(weighted_x), float(weighted_y)),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=final_confidence
                ))
        
        # Step 4: Fuse right goals (same improved process)
        if right_goals:
            # Use larger clustering distance (100 pixels) to merge nearby detections
            clusters = []
            for method, goal in right_goals:
                gx, gy = goal.image_point
                found_cluster = False
                
                # Try to find existing cluster
                for cluster in clusters:
                    # Check distance to cluster center
                    cluster_centers = [(g.image_point[0], g.image_point[1]) for _, g in cluster]
                    avg_x = np.mean([cx for cx, _ in cluster_centers])
                    avg_y = np.mean([cy for _, cy in cluster_centers])
                    
                    dist = np.sqrt((gx - avg_x)**2 + (gy - avg_y)**2)
                    if dist < 100:  # Larger clustering distance
                        cluster.append((method, goal))
                        found_cluster = True
                        break
                
                if not found_cluster:
                    clusters.append([(method, goal)])
            
            # Sort clusters by quality score
            def cluster_score(cluster):
                num_methods = len(cluster)
                avg_conf = np.mean([g.confidence for _, g in cluster])
                has_zero_shot = any(m == 'zero_shot' for m, _ in cluster)
                has_color = any(m == 'color' for m, _ in cluster)
                
                method_bonus = 3.0 if has_zero_shot else (2.0 if has_color else 1.0)
                return method_bonus * num_methods * avg_conf
            
            clusters.sort(key=cluster_score, reverse=True)
            
            # Keep only the BEST 1-2 clusters per side
            best_clusters = []
            for cluster in clusters:
                if len(cluster) >= 2:  # Require at least 2 detections agreeing
                    best_clusters.append(cluster)
                elif len(best_clusters) == 0 and len(cluster) == 1:
                    # Allow single detection only if no better clusters exist
                    method, goal = cluster[0]
                    if method in ('zero_shot', 'color') and goal.confidence > 0.7:
                        best_clusters.append(cluster)
                
                if len(best_clusters) >= 2:  # Max 2 goals per side
                    break
            
            # Fuse each best cluster
            for cluster in best_clusters:
                if len(cluster) == 0:
                    continue
                
                # Calculate weighted average position
                weighted_sum_x = 0
                weighted_sum_y = 0
                total_weight = 0
                
                for method, goal in cluster:
                    weight = goal.confidence
                    if method == 'zero_shot':
                        weight *= 3.0
                    elif method == 'color':
                        weight *= 2.0
                    weighted_sum_x += goal.image_point[0] * weight
                    weighted_sum_y += goal.image_point[1] * weight
                    total_weight += weight
                
                if total_weight == 0:
                    continue
                
                weighted_x = weighted_sum_x / total_weight
                weighted_y = weighted_sum_y / total_weight
                
                # Confidence calculation
                num_methods = len(cluster)
                has_zero_shot = any(method == 'zero_shot' for method, _ in cluster)
                has_color = any(method == 'color' for method, _ in cluster)
                base_confidence = np.mean([goal.confidence for _, goal in cluster])
                consensus_boost = min(0.15, num_methods * 0.05)
                zero_shot_boost = 0.15 if has_zero_shot else 0
                color_boost = 0.1 if has_color else 0
                final_confidence = min(1.0, base_confidence + consensus_boost + zero_shot_boost + color_boost)
                
                fused.append(PitchKeypoint(
                    image_point=(float(weighted_x), float(weighted_y)),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal",
                    confidence=final_confidence
                ))
        
        # Step 5: Final deduplication - remove any remaining duplicates
        # Check for goals that are too close to each other (within 20 pixels)
        if len(fused) > 1:
            deduplicated = []
            used = set()
            for i, goal1 in enumerate(fused):
                if i in used:
                    continue
                
                g1x, g1y = goal1.image_point
                keep_goal = goal1
                
                # Check for nearby goals
                for j, goal2 in enumerate(fused[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    g2x, g2y = goal2.image_point
                    dist = np.sqrt((g1x - g2x)**2 + (g1y - g2y)**2)
                    
                    if dist < 20:  # Very close, likely duplicate
                        # Keep the one with higher confidence
                        if goal2.confidence > keep_goal.confidence:
                            keep_goal = goal2
                        used.add(j)
                
                deduplicated.append(keep_goal)
            
            fused = deduplicated
        
        # Step 6: Final filtering - ensure we have at most 4 goals total (2 per side)
        if len(fused) > 4:
            # Sort by confidence and keep top 4
            fused.sort(key=lambda g: g.confidence, reverse=True)
            fused = fused[:4]
        
        return fused
    
    def _validate_goal_geometry(self, goal: PitchKeypoint, image: np.ndarray) -> bool:
        """
        Comprehensive geometric validation for goal detections.
        
        Stricter validation to reduce false positives:
        - Goals must be near image edges (left/right)
        - Minimum confidence threshold
        - Reasonable vertical position
        - Valid pitch coordinates
        
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
        
        # Minimum confidence threshold
        if goal.confidence < 0.5:
            return False
        
        # Validate pitch coordinates
        expected_x_left = -self.pitch_length / 2
        expected_x_right = self.pitch_length / 2
        
        if pitch_x < 0:
            # Left goal - must be on left side of image
            if abs(pitch_x - expected_x_left) > 10.0:  # Within 10m
                return False
            # Stricter: left goal should be in left 40% of image
            if gx > w * 0.4:
                return False
        else:
            # Right goal - must be on right side of image
            if abs(pitch_x - expected_x_right) > 10.0:
                return False
            # Stricter: right goal should be in right 40% of image
            if gx < w * 0.6:
                return False
        
        # Pitch y should be near 0 (center line)
        if abs(pitch_y) > 5.0:
            return False
        
        # Goal should be in reasonable vertical position
        # Not at extreme top/bottom (stricter: within 25% of center)
        if abs(gy - h / 2) > h * 0.25:
            return False
        
        # All validations passed
        return True
    
    def _validate_goal_post_geometry(self, goal: PitchKeypoint, image: np.ndarray) -> bool:
        """
        Validation for individual goal posts (more relaxed than goal centers).
        
        Args:
            goal: Goal post keypoint to validate
            image: Input image for reference
        
        Returns:
            True if goal post passes validation, False otherwise
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
            # Left goal side
            if abs(pitch_x - expected_x_left) > 10.0:  # Within 10m
                return False
        else:
            # Right goal side
            if abs(pitch_x - expected_x_right) > 10.0:
                return False
        
        # Pitch y should be near 0 (center line)
        if abs(pitch_y) > 5.0:
            return False
        
        # Posts can be anywhere vertically (more relaxed than goal centers)
        # Just ensure they're not at extreme edges
        if gy < h * 0.05 or gy > h * 0.95:
            return False
        
        return True
    
    def _deduplicate_regions(self, regions: List[Tuple], w: int, h: int, 
                            distance_threshold: float = 30.0) -> List[Tuple]:
        """
        Deduplicate regions from multi-scale detection.
        
        Args:
            regions: List of (center_x, center_y, width, height, aspect_ratio) tuples
            w: Image width
            h: Image height
            distance_threshold: Maximum distance for considering regions as duplicates (pixels)
        
        Returns:
            Deduplicated list of regions
        """
        if len(regions) == 0:
            return []
        
        # Sort by area (largest first) to keep best detections
        regions_sorted = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
        
        deduplicated = []
        used = set()
        
        for i, region in enumerate(regions_sorted):
            if i in used:
                continue
            
            center_x, center_y = region[0], region[1]
            deduplicated.append(region)
            
            # Mark nearby regions as duplicates
            for j, other_region in enumerate(regions_sorted[i+1:], start=i+1):
                if j in used:
                    continue
                
                other_x, other_y = other_region[0], other_region[1]
                distance = np.sqrt((center_x - other_x)**2 + (center_y - other_y)**2)
                
                if distance < distance_threshold:
                    used.add(j)
        
        return deduplicated
    
    def _apply_temporal_smoothing(self, current_goals: List[PitchKeypoint], 
                                  image: np.ndarray) -> List[PitchKeypoint]:
        """
        Apply temporal smoothing and multi-frame fusion for goal positions.
        
        Uses Kalman filtering principles to predict goal positions during occlusion
        and aggregates detections across multiple frames for robustness.
        
        Args:
            current_goals: Current frame's goal detections
            image: Current image (for reference)
        
        Returns:
            Temporally smoothed goal keypoints
        """
        h, w = image.shape[:2]
        
        # Add current detections to history
        self._goal_history.append({
            'goals': current_goals.copy(),
            'frame_time': len(self._goal_history)  # Simple frame counter
        })
        
        # Keep only recent history
        if len(self._goal_history) > self._max_history_size:
            self._goal_history.pop(0)
        
        if len(self._goal_history) < 2:
            # Not enough history yet, return current detections
            return current_goals
        
        # Group goals by side (left vs right) across frames
        left_goals_history = []
        right_goals_history = []
        
        for frame_data in self._goal_history:
            for goal in frame_data['goals']:
                pitch_x = goal.pitch_point[0]
                if pitch_x < 0:
                    left_goals_history.append(goal)
                else:
                    right_goals_history.append(goal)
        
        # Temporal fusion: aggregate positions across frames
        smoothed_goals = []
        
        # Process left goals
        if left_goals_history:
            # Cluster nearby detections across frames
            left_clusters = self._cluster_goals_temporal(left_goals_history, w, h)
            for cluster in left_clusters:
                if len(cluster) >= 2:  # Need at least 2 detections across frames
                    # Weighted average position (more recent = higher weight)
                    weighted_x = 0
                    weighted_y = 0
                    total_weight = 0
                    
                    for i, goal in enumerate(cluster):
                        # More recent detections get higher weight
                        weight = goal.confidence * (i + 1) / len(cluster)
                        weighted_x += goal.image_point[0] * weight
                        weighted_y += goal.image_point[1] * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        avg_x = weighted_x / total_weight
                        avg_y = weighted_y / total_weight
                        avg_confidence = np.mean([g.confidence for g in cluster])
                        
                        smoothed_goals.append(PitchKeypoint(
                            image_point=(float(avg_x), float(avg_y)),
                            pitch_point=(-self.pitch_length / 2, 0.0),
                            landmark_type="goal",
                            confidence=min(1.0, avg_confidence * 1.1)  # Boost confidence from consensus
                        ))
        
        # Process right goals
        if right_goals_history:
            right_clusters = self._cluster_goals_temporal(right_goals_history, w, h)
            for cluster in right_clusters:
                if len(cluster) >= 2:
                    weighted_x = 0
                    weighted_y = 0
                    total_weight = 0
                    
                    for i, goal in enumerate(cluster):
                        weight = goal.confidence * (i + 1) / len(cluster)
                        weighted_x += goal.image_point[0] * weight
                        weighted_y += goal.image_point[1] * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        avg_x = weighted_x / total_weight
                        avg_y = weighted_y / total_weight
                        avg_confidence = np.mean([g.confidence for g in cluster])
                        
                        smoothed_goals.append(PitchKeypoint(
                            image_point=(float(avg_x), float(avg_y)),
                            pitch_point=(self.pitch_length / 2, 0.0),
                            landmark_type="goal",
                            confidence=min(1.0, avg_confidence * 1.1)
                        ))
        
        # If temporal smoothing found goals, use them; otherwise use current detections
        if len(smoothed_goals) > 0:
            return smoothed_goals
        else:
            return current_goals
    
    def _cluster_goals_temporal(self, goals: List[PitchKeypoint], w: int, h: int,
                                distance_threshold: float = 50.0) -> List[List[PitchKeypoint]]:
        """
        Cluster goal detections across frames that are spatially close.
        
        Args:
            goals: List of goal detections from multiple frames
            w: Image width
            h: Image height
            distance_threshold: Maximum distance for clustering (pixels)
        
        Returns:
            List of clusters, each containing spatially close goals
        """
        if len(goals) == 0:
            return []
        
        clusters = []
        
        for goal in goals:
            gx, gy = goal.image_point
            
            # Find existing cluster
            found_cluster = False
            for cluster in clusters:
                for c_goal in cluster:
                    cx, cy = c_goal.image_point
                    dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
                    if dist < distance_threshold:
                        cluster.append(goal)
                        found_cluster = True
                        break
                if found_cluster:
                    break
            
            if not found_cluster:
                clusters.append([goal])
        
        return clusters
    
    def _detect_goals_zero_shot(self, image: np.ndarray) -> List[PitchKeypoint]:
        """
        Detect goals using zero-shot vision-language models (Grounding DINO/OWL-ViT).
        
        This method provides semantic understanding of goals without requiring
        labeled training data. Uses Apache 2.0 licensed models.
        
        Args:
            image: Input image
        
        Returns:
            List of detected goal keypoints
        """
        if not self.enable_zero_shot:
            return []
        
        # Lazy load zero-shot detector
        if self._zero_shot_detector is None:
            try:
                from src.analysis.zero_shot_goal_detector import ZeroShotGoalDetector
                self._zero_shot_detector = ZeroShotGoalDetector(
                    model_name="grounding-dino-base",
                    box_threshold=0.35,
                    text_threshold=0.25,
                    enable_fp16=True
                )
            except ImportError:
                # transformers not available, disable zero-shot
                self.enable_zero_shot = False
                return []
            except Exception as e:
                print(f"Warning: Could not initialize zero-shot detector: {e}")
                self.enable_zero_shot = False
                return []
        
        # Run zero-shot detection
        try:
            goals = self._zero_shot_detector.detect_goals(
                image,
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width
            )
            return goals
        except Exception as e:
            print(f"Warning: Zero-shot detection failed: {e}")
            return []
    
    def _detect_goals_by_color(self, image: np.ndarray) -> List[PitchKeypoint]:
        """
        Detect goal posts using enhanced color-based detection with adaptive thresholds.
        
        Uses adaptive thresholding based on image statistics and multi-scale processing
        for better robustness to varying lighting conditions.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of goal keypoints detected by color
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Adaptive thresholding: analyze image statistics
        v_channel = hsv[:, :, 2]  # Value channel
        v_mean = np.mean(v_channel)
        v_std = np.std(v_channel)
        
        # Adaptive white detection: adjust thresholds based on image brightness
        # For bright images, use higher thresholds; for dark images, use lower
        if v_mean > 150:  # Bright image
            base_v_low = 200
            base_v_high = 255
        elif v_mean > 100:  # Medium brightness
            base_v_low = max(150, v_mean - v_std)
            base_v_high = min(255, v_mean + 2 * v_std)
        else:  # Dark image
            base_v_low = max(120, v_mean - 0.5 * v_std)
            base_v_high = min(255, v_mean + 2.5 * v_std)
        
        # Define adaptive white color range in HSV
        lower_white = np.array([0, 0, int(base_v_low)])
        upper_white = np.array([180, 30, int(base_v_high)])
        
        # Multi-scale detection: process at different scales
        scales = [1.0, 0.75, 1.25]  # Original, smaller, larger
        all_goal_regions = []
        
        for scale in scales:
            if scale != 1.0:
                scaled_h = int(h * scale)
                scaled_w = int(w * scale)
                scaled_hsv = cv2.resize(hsv, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            else:
                scaled_hsv = hsv
            
            # Create mask for white regions
            white_mask = cv2.inRange(scaled_hsv, lower_white, upper_white)
            
            # Enhanced morphological operations
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            
            # Close small gaps
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_small)
            # Remove noise
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_medium)
            
            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Scale back coordinates if needed
            for contour in contours:
                if scale != 1.0:
                    # Scale contour coordinates back to original size
                    contour = (contour / scale).astype(np.int32)
                
                x, y, cw, ch = cv2.boundingRect(contour)
                if ch > 0:
                    aspect_ratio = cw / ch
                    area = cw * ch
                    
                    # Filter for goal-like structures
                    # Goal posts: tall and narrow (aspect_ratio < 0.4)
                    # Crossbars: wide and short (aspect_ratio > 2.0)
                    # Large structures: entire goal frame
                    is_vertical_post = aspect_ratio < 0.4 and ch > h * 0.03
                    is_crossbar = aspect_ratio > 2.0 and cw > w * 0.02
                    is_large_structure = area > (h * w * 0.001) and 0.3 < aspect_ratio < 3.0
                    
                    if is_vertical_post or is_crossbar or is_large_structure:
                        center_x = x + cw / 2
                        center_y = y + ch / 2
                        all_goal_regions.append((center_x, center_y, cw, ch, aspect_ratio))
        
        # Deduplicate regions from multi-scale detection
        goal_regions = self._deduplicate_regions(all_goal_regions, w, h)
        
        # Group goal posts by region (left vs right)
        # goal_regions format: (center_x, center_y, width, height, aspect_ratio)
        # Extract just (x, y) for pair finding
        left_posts = [(p[0], p[1]) for p in goal_regions if p[0] < w / 2]
        right_posts = [(p[0], p[1]) for p in goal_regions if p[0] > w / 2]
        
        # Look for goal structures: pairs of posts (two vertical posts)
        # Goals typically have two posts, so look for pairs of posts close together
        def find_goal_posts(posts, pitch_x):
            """
            Find goal post pairs and return individual left and right posts.
            
            Args:
                posts: List of post positions (x, y) tuples
                pitch_x: Pitch x coordinate for this goal side (negative for left, positive for right)
            
            Returns:
                List of (left_post, right_post) tuples, or empty list if no pairs found
            """
            if len(posts) < 2:
                return []  # Need at least 2 posts to form a pair
            
            # Sort by x position
            sorted_posts = sorted(posts, key=lambda p: p[0])
            pairs = []
            
            # Find pairs that are close together (goal width ~7.32m, but in pixels)
            # Typical goal width in image: roughly 30-500 pixels depending on distance and camera angle
            for i in range(len(sorted_posts) - 1):
                p1 = sorted_posts[i]  # Left post (smaller x)
                p2 = sorted_posts[i + 1]  # Right post (larger x)
                dist = abs(p1[0] - p2[0])
                
                # Goal posts should be roughly 30-500 pixels apart
                if 30 < dist < 500:
                    pairs.append((p1, p2))
            
            return pairs
        
        # Find left goal post pairs
        left_goal_pairs = find_goal_posts(left_posts, -self.pitch_length / 2)
        
        # Score and sort pairs by quality (prefer pairs closer to expected goal position)
        # Goals should be near the left edge and in reasonable vertical position
        scored_left_pairs = []
        for left_post, right_post in left_goal_pairs:
            goal_y = (left_post[1] + right_post[1]) / 2
            goal_x = (left_post[0] + right_post[0]) / 2
            post_distance = abs(left_post[0] - right_post[0])
            
            # Score: prefer goals near left edge, reasonable vertical position, reasonable post distance
            edge_score = 1.0 / (1.0 + goal_x / (w * 0.1))  # Closer to left edge = higher score
            vertical_score = 1.0 - abs(goal_y - h / 2) / (h / 2)  # Closer to center = higher score
            distance_score = 1.0 if 50 < post_distance < 300 else 0.5  # Reasonable goal width
            
            score = edge_score * vertical_score * distance_score
            scored_left_pairs.append((score, left_post, right_post))
        
        # Sort by score and keep only top 1-2 pairs
        scored_left_pairs.sort(key=lambda x: x[0], reverse=True)
        for score, left_post, right_post in scored_left_pairs[:2]:  # Max 2 pairs
            goal_y = (left_post[1] + right_post[1]) / 2
            # Additional validation: reasonable vertical position
            if h * 0.1 < goal_y < h * 0.9:
                # Left post of the goal (smaller x coordinate)
                keypoints.append(PitchKeypoint(
                    image_point=(float(left_post[0]), float(left_post[1])),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal_post_left",
                    confidence=0.8
                ))
                # Right post of the goal (larger x coordinate)
                keypoints.append(PitchKeypoint(
                    image_point=(float(right_post[0]), float(right_post[1])),
                    pitch_point=(-self.pitch_length / 2, 0.0),
                    landmark_type="goal_post_right",
                    confidence=0.8
                ))
        
        # Find right goal post pairs
        right_goal_pairs = find_goal_posts(right_posts, self.pitch_length / 2)
        
        # Score and sort pairs by quality
        scored_right_pairs = []
        for left_post, right_post in right_goal_pairs:
            goal_y = (left_post[1] + right_post[1]) / 2
            goal_x = (left_post[0] + right_post[0]) / 2
            post_distance = abs(left_post[0] - right_post[0])
            
            # Score: prefer goals near right edge, reasonable vertical position, reasonable post distance
            edge_score = 1.0 / (1.0 + (w - goal_x) / (w * 0.1))  # Closer to right edge = higher score
            vertical_score = 1.0 - abs(goal_y - h / 2) / (h / 2)  # Closer to center = higher score
            distance_score = 1.0 if 50 < post_distance < 300 else 0.5  # Reasonable goal width
            
            score = edge_score * vertical_score * distance_score
            scored_right_pairs.append((score, left_post, right_post))
        
        # Sort by score and keep only top 1-2 pairs
        scored_right_pairs.sort(key=lambda x: x[0], reverse=True)
        for score, left_post, right_post in scored_right_pairs[:2]:  # Max 2 pairs
            goal_y = (left_post[1] + right_post[1]) / 2
            # Additional validation: goal should be in reasonable vertical position
            if h * 0.1 < goal_y < h * 0.9:  # Between 10% and 90% of image height
                # Left post of the goal (smaller x coordinate)
                keypoints.append(PitchKeypoint(
                    image_point=(float(left_post[0]), float(left_post[1])),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal_post_left",
                    confidence=0.8
                ))
                # Right post of the goal (larger x coordinate)
                keypoints.append(PitchKeypoint(
                    image_point=(float(right_post[0]), float(right_post[1])),
                    pitch_point=(self.pitch_length / 2, 0.0),
                    landmark_type="goal_post_right",
                    confidence=0.8
                ))
        
        # If no goals found by posts, try finding large white structures (entire goal frame)
        if not keypoints:
            # Look for large white rectangular structures that could be goals
            large_structures = []
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                aspect_ratio = cw / ch if ch > 0 else 0
                
                # Look for reasonably large structures in goal regions
                # Right side: x > w * 0.4, reasonable size, not too extreme aspect ratio
                if x > w * 0.4 and area > (h * w * 0.001) and 0.3 < aspect_ratio < 3.0:
                    center_y = y + ch / 2
                    # Prefer structures in middle-upper region (not at very bottom)
                    if h * 0.1 < center_y < h * 0.85:
                        large_structures.append((x + cw / 2, center_y, area))
            
            # Use the largest structure in reasonable position
            if large_structures:
                large_structures.sort(key=lambda s: s[2], reverse=True)  # Sort by area
                best_goal = large_structures[0]
                keypoints.append(PitchKeypoint(
                    image_point=(float(best_goal[0]), float(best_goal[1])),
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
        
        # Check if color-based detection found individual posts
        color_posts = [g for method, g in all_detections if method == 'color' and g.landmark_type in ('goal_post_left', 'goal_post_right')]
        
        # If we have individual posts from color detection, validate and pair them properly
        if len(color_posts) > 0:
            # Apply validation to individual posts
            validated_posts = []
            for goal in color_posts:
                # Use relaxed validation for individual posts
                if self._validate_goal_post_geometry(goal, image):
                    validated_posts.append(goal)
            
            if len(validated_posts) > 0:
                # Group by side
                left_side_posts = [g for g in validated_posts if g.pitch_point[0] < 0]
                right_side_posts = [g for g in validated_posts if g.pitch_point[0] > 0]
                
                # Pair posts on each side - only keep posts that are part of a valid pair
                paired_posts = []
                
                # Pair left side posts
                for i, post1 in enumerate(left_side_posts):
                    for post2 in left_side_posts[i+1:]:
                        # Check if these two posts could be from the same goal
                        dist = np.sqrt((post1.image_point[0] - post2.image_point[0])**2 + 
                                      (post1.image_point[1] - post2.image_point[1])**2)
                        # Goal posts should be 30-500 pixels apart horizontally
                        # And roughly at similar vertical positions (within 300 pixels for perspective)
                        h_dist = abs(post1.image_point[0] - post2.image_point[0])
                        v_dist = abs(post1.image_point[1] - post2.image_point[1])
                        
                        # More flexible: allow larger vertical distance for perspective views
                        # But ensure horizontal distance is reasonable for goal width
                        if 30 < h_dist < 500 and v_dist < 300:  # Valid goal pair
                            # Determine which is left and which is right post
                            if post1.image_point[0] < post2.image_point[0]:
                                left_post, right_post = post1, post2
                            else:
                                left_post, right_post = post2, post1
                            
                            # Only add if not already added
                            if left_post not in paired_posts:
                                paired_posts.append(left_post)
                            if right_post not in paired_posts:
                                paired_posts.append(right_post)
                            break  # Found a pair for post1
                
                # Pair right side posts
                for i, post1 in enumerate(right_side_posts):
                    for post2 in right_side_posts[i+1:]:
                        dist = np.sqrt((post1.image_point[0] - post2.image_point[0])**2 + 
                                      (post1.image_point[1] - post2.image_point[1])**2)
                        h_dist = abs(post1.image_point[0] - post2.image_point[0])
                        v_dist = abs(post1.image_point[1] - post2.image_point[1])
                        
                        # More flexible: allow larger vertical distance for perspective views
                        if 30 < h_dist < 500 and v_dist < 300:  # Valid goal pair
                            if post1.image_point[0] < post2.image_point[0]:
                                left_post, right_post = post1, post2
                            else:
                                left_post, right_post = post2, post1
                            
                            if left_post not in paired_posts:
                                paired_posts.append(left_post)
                            if right_post not in paired_posts:
                                paired_posts.append(right_post)
                            break
                
                # Only return if we have properly paired posts (at least 2 posts forming a pair)
                if len(paired_posts) >= 2:
                    # Limit to max 4 posts (2 pairs: one per side)
                    if len(paired_posts) <= 4:
                        return paired_posts
                    else:
                        # If too many, keep the best pairs (by confidence and position)
                        paired_posts.sort(key=lambda g: (g.confidence, -abs(g.image_point[0] - image.shape[1]/2)), reverse=True)
                        return paired_posts[:4]
        
        # Otherwise, use other methods and fuse
        # Method 2: Improved vertical line detection
        line_goals = self._detect_goals_improved_lines(image, lines)
        for goal in line_goals:
            all_detections.append(('lines', goal))
        
        # Method 3: Goal post pair detection
        pair_goals = self._detect_goal_post_pairs(image, lines)
        for goal in pair_goals:
            all_detections.append(('pairs', goal))
        
        # Method 4: Zero-shot detection (semantic understanding)
        if self.enable_zero_shot:
            zero_shot_goals = self._detect_goals_zero_shot(image)
            for goal in zero_shot_goals:
                all_detections.append(('zero_shot', goal))
        
        # Method 5: Crossbar detection (enhances existing detections)
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
        
        # Apply temporal smoothing and multi-frame fusion
        if len(validated_goals) > 0:
            smoothed_goals = self._apply_temporal_smoothing(validated_goals, image)
            return smoothed_goals
        
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
    
    def _detect_center_circle_enhanced(self, image: np.ndarray, lines: List[np.ndarray], num_points: int = 8, 
                                      frame_buffer: Optional[List[np.ndarray]] = None) -> List[PitchKeypoint]:
        """
        Enhanced center circle detection with radius validation.
        Samples multiple points evenly spaced around the circle for better homography accuracy.
        
        Uses segmentation mask first (better for off-white circles), then falls back to grayscale.
        Supports temporal averaging if frame_buffer is provided (static camera).
        
        Args:
            image: Input image
            lines: Detected field lines
            num_points: Number of points to sample around circle (default: 8, range: 4-16)
            frame_buffer: Optional list of recent frames for temporal averaging (static camera)
        
        Returns:
            List of center circle keypoints (center + num_points evenly spaced around circle)
        """
        h, w = image.shape[:2]
        keypoints = []
        
        # Clamp num_points to reasonable range
        num_points = max(4, min(16, num_points))
        
        # Method 1: Try HoughCircles on segmentation mask (better for off-white circles)
        # Use temporal averaging if available (static camera)
        segmentation_mask = None
        if self.use_semantic_segmentation and self._line_segmenter is not None:
            try:
                if frame_buffer is not None and len(frame_buffer) >= 3:
                    # Use averaged mask for better stability (static camera)
                    segmentation_mask = self._line_segmenter.segment_pitch_lines_averaged(frame_buffer)
                else:
                    # Single frame detection
                    segmentation_mask = self._line_segmenter.segment_pitch_lines(image)
            except:
                pass
        
        # Optimize: Downscale large images for HoughCircles (much faster)
        if max(h, w) > 3000:
            scale_factor = 0.25
        elif max(h, w) > 2000:
            scale_factor = 0.4
        else:
            scale_factor = 1.0
        
        circles = None
        
        # Try detection on segmentation mask first (where circle appears as white pixels)
        if segmentation_mask is not None:
            if scale_factor < 1.0:
                small_mask = cv2.resize(segmentation_mask, None, fx=scale_factor, fy=scale_factor)
                small_h, small_w = small_mask.shape[:2]
            else:
                small_mask = segmentation_mask
                small_h, small_w = h, w
            
            # HoughCircles on mask - circle should appear as white pixels
            # Lower param2 to detect weaker circles (20 instead of 25)
            circles = cv2.HoughCircles(
                small_mask,
                cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=int(min(small_h, small_w) / 3),
                param1=50,
                param2=20,  # Lowered to 20 for weaker circle detection
                minRadius=int(min(small_h, small_w) / 25),
                maxRadius=int(min(small_h, small_w) / 8)
            )
        
        # Fallback: Try on grayscale if mask detection failed
        if circles is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if scale_factor < 1.0:
                small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
                small_h, small_w = small_gray.shape[:2]
            else:
                small_gray = gray
                small_h, small_w = h, w
            
            # Detect circles with optimized parameters (on downscaled image)
            circles = cv2.HoughCircles(
                small_gray,
                cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=int(min(small_h, small_w) / 3),
                param1=50,
                param2=20,  # Lowered to 20 for weaker circle detection
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
                    # Higher confidence if detected from segmentation mask
                    mask_confidence = 0.85 if segmentation_mask is not None else 0.8
                    keypoints.append(PitchKeypoint(
                        image_point=(float(cx), float(cy)),
                        pitch_point=(0.0, 0.0),
                        landmark_type="center_circle",
                        confidence=mask_confidence
                    ))
                    
                    # Sample num_points evenly spaced around the circle
                    # Angle 0 is at top (negative y in pitch coordinates)
                    point_confidence = 0.75 if segmentation_mask is not None else 0.7
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
                            confidence=point_confidence
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
                                max_points: int = 25,
                                use_semantic_segmentation: bool = False,
                                segmentation_config: Optional[Dict] = None) -> Optional[Dict]:
    """
    Convenience function to automatically detect pitch keypoints
    
    Args:
        image: Input image (BGR)
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        min_points: Minimum points required (default: 4)
        max_points: Maximum points to use (default: 25 for comprehensive system)
        use_semantic_segmentation: Enable semantic segmentation for line detection
        segmentation_config: Configuration dict for segmentation (model_path, model_type, etc.)
    
    Returns:
        Dictionary with 'image_points' and 'pitch_points' arrays, or None if insufficient points
    """
    detector = PitchKeypointDetector(
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        use_semantic_segmentation=use_semantic_segmentation,
        segmentation_config=segmentation_config
    )
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
