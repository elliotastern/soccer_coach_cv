"""
Ball tracking with parabolic fit validation to filter false positives.
Checks if ball trajectories follow gravity curves to eliminate "ghost balls" (white socks, lines, etc.).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from src.types import Detection, TrackedObject

# Try to import scipy, fallback to numpy polyfit
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using numpy polyfit for parabolic fitting")


@dataclass
class BallTrackPoint:
    """Single point in a ball track"""
    frame_id: int
    x: float  # Center x coordinate
    y: float  # Center y coordinate
    timestamp: float  # Frame timestamp or frame number


class BallTracker:
    """
    Ball tracker with parabolic fit validation.
    Filters out tracks that don't follow gravity curves (false positives like white socks).
    """
    
    def __init__(self, min_track_length: int = 5, fit_threshold: float = 0.15):
        """
        Initialize ball tracker.
        
        Args:
            min_track_length: Minimum number of detections before checking parabolic fit (default: 5)
            fit_threshold: Maximum normalized residual for valid parabolic fit (default: 0.15 = 15%)
        """
        self.min_track_length = min_track_length
        self.fit_threshold = fit_threshold
        
        # Track history: track_id -> List[BallTrackPoint]
        self.track_history: Dict[int, List[BallTrackPoint]] = defaultdict(list)
        
        # Track status: track_id -> 'valid' | 'noise' | 'pending'
        self.track_status: Dict[int, str] = {}
        
        # Frame counter for timestamps
        self.frame_id = 0
    
    def _parabolic_2d_time(self, t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Parabolic function for vertical motion: y(t) = a*t^2 + b*t + c
        This models gravity-affected vertical motion.
        """
        return a * t**2 + b * t + c
    
    def _linear_time(self, t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Linear function for horizontal motion: x(t) = a*t + b
        This models constant-velocity horizontal motion.
        """
        return a * t + b
    
    def _fit_parabolic_trajectory(
        self, 
        points: List[BallTrackPoint]
    ) -> Tuple[bool, float]:
        """
        Fit a 2D parabolic trajectory to ball track points.
        
        Args:
            points: List of track points
        
        Returns:
            Tuple of (is_valid, normalized_residual)
            - is_valid: True if fit is good (follows gravity curve)
            - normalized_residual: Normalized residual (0-1, lower is better)
        """
        if len(points) < self.min_track_length:
            return True, 0.0  # Not enough points, assume valid
        
        # Extract coordinates and time
        times = np.array([p.frame_id for p in points], dtype=np.float32)
        x_coords = np.array([p.x for p in points], dtype=np.float32)
        y_coords = np.array([p.y for p in points], dtype=np.float32)
        
        # Normalize time to start from 0
        times = times - times[0]
        
        # Fit vertical motion (y) as parabola: y(t) = a*t^2 + b*t + c
        # For a ball in free flight, vertical motion should follow gravity
        try:
            if SCIPY_AVAILABLE:
                # Use scipy curve_fit for more robust fitting
                popt_y, _ = curve_fit(
                    self._parabolic_2d_time,
                    times,
                    y_coords,
                    p0=[0.1, 0.0, y_coords[0]],  # Initial guess: small acceleration, zero velocity, start position
                    maxfev=1000
                )
                a_y, b_y, c_y = popt_y
                
                # Fit horizontal motion (x) as linear: x(t) = a*t + b
                popt_x, _ = curve_fit(
                    self._linear_time,
                    times,
                    x_coords,
                    p0=[0.0, x_coords[0]],  # Initial guess: zero velocity, start position
                    maxfev=1000
                )
                a_x, b_x = popt_x
            else:
                # Fallback to numpy polyfit
                # Fit parabola to y(t): degree 2
                poly_y = np.polyfit(times, y_coords, deg=2)
                a_y, b_y, c_y = poly_y[0], poly_y[1], poly_y[2]
                
                # Fit line to x(t): degree 1
                poly_x = np.polyfit(times, x_coords, deg=1)
                a_x, b_x = poly_x[0], poly_x[1]
            
            # Calculate residuals for y
            y_pred = a_y * times**2 + b_y * times + c_y
            y_residual = np.mean(np.abs(y_coords - y_pred))
            y_range = np.max(y_coords) - np.min(y_coords)
            y_normalized_residual = y_residual / (y_range + 1e-6)  # Avoid division by zero
            
            # Calculate residuals for x
            x_pred = a_x * times + b_x
            x_residual = np.mean(np.abs(x_coords - x_pred))
            x_range = np.max(x_coords) - np.min(x_coords)
            x_normalized_residual = x_residual / (x_range + 1e-6)
            
            # Combined normalized residual (weighted average)
            # Vertical motion is more important for gravity check
            combined_residual = 0.7 * y_normalized_residual + 0.3 * x_normalized_residual
            
            # Check if fit is good
            is_valid = combined_residual < self.fit_threshold
            
            return is_valid, combined_residual
            
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            # Fit failed (e.g., singular matrix, too few points)
            # If fit fails, it's likely not a valid ball trajectory
            return False, 1.0
    
    def update(
        self, 
        tracked_objects: List[TrackedObject],
        frame_id: int
    ) -> List[TrackedObject]:
        """
        Update ball tracker and filter out invalid tracks.
        
        Args:
            tracked_objects: List of tracked objects from ByteTrack
            frame_id: Current frame ID
        
        Returns:
            Filtered list of tracked objects (balls only, with noise removed)
        """
        self.frame_id = frame_id
        
        # Separate balls from other objects
        ball_objects = [obj for obj in tracked_objects if obj.detection.class_name == 'ball']
        non_ball_objects = [obj for obj in tracked_objects if obj.detection.class_name != 'ball']
        
        # Update track history for balls
        for ball_obj in ball_objects:
            track_id = ball_obj.object_id
            x, y, w, h = ball_obj.detection.bbox
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Add point to track history
            point = BallTrackPoint(
                frame_id=frame_id,
                x=center_x,
                y=center_y,
                timestamp=frame_id  # Use frame_id as timestamp
            )
            self.track_history[track_id].append(point)
            
            # Keep only recent history (last 30 frames to avoid memory issues)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Check parabolic fit if we have enough points
            if len(self.track_history[track_id]) >= self.min_track_length:
                if track_id not in self.track_status or self.track_status[track_id] == 'pending':
                    is_valid, residual = self._fit_parabolic_trajectory(
                        self.track_history[track_id]
                    )
                    
                    if is_valid:
                        self.track_status[track_id] = 'valid'
                    else:
                        self.track_status[track_id] = 'noise'
                        # Mark for deletion
            else:
                # Not enough points yet, mark as pending
                if track_id not in self.track_status:
                    self.track_status[track_id] = 'pending'
        
        # Filter out noise tracks
        valid_ball_objects = [
            obj for obj in ball_objects
            if self.track_status.get(obj.object_id, 'pending') != 'noise'
        ]
        
        # Clean up old tracks (remove tracks not seen in recent frames)
        active_track_ids = {obj.object_id for obj in ball_objects}
        for track_id in list(self.track_history.keys()):
            if track_id not in active_track_ids:
                # Check if track is old (last seen more than 10 frames ago)
                if self.track_history[track_id]:
                    last_frame = self.track_history[track_id][-1].frame_id
                    if frame_id - last_frame > 10:
                        # Remove old track
                        del self.track_history[track_id]
                        if track_id in self.track_status:
                            del self.track_status[track_id]
        
        # Return filtered objects (valid balls + non-balls)
        return valid_ball_objects + non_ball_objects
    
    def reset(self):
        """Reset tracker state (call on scene cut)"""
        self.track_history.clear()
        self.track_status.clear()
        self.frame_id = 0


def create_ball_tracker_wrapper(base_tracker, min_track_length: int = 5, fit_threshold: float = 0.15):
    """
    Create a wrapper that adds parabolic fit validation to an existing tracker.
    
    Args:
        base_tracker: Base tracker (e.g., Tracker from tracker.py)
        min_track_length: Minimum track length before validation
        fit_threshold: Maximum normalized residual for valid fit
    
    Returns:
        Wrapped tracker with ball validation
    """
    ball_tracker = BallTracker(min_track_length=min_track_length, fit_threshold=fit_threshold)
    
    class TrackerWithBallValidation:
        """Wrapper that adds ball validation to base tracker"""
        
        def __init__(self, base_tracker, ball_tracker):
            self.base_tracker = base_tracker
            self.ball_tracker = ball_tracker
            self.frame_id = 0
        
        def update(self, detections, frame):
            """Update with ball validation"""
            # Get tracked objects from base tracker
            tracked_objects = self.base_tracker.update(detections, frame)
            
            # Apply ball validation
            filtered_objects = self.ball_tracker.update(tracked_objects, self.frame_id)
            self.frame_id += 1
            
            return filtered_objects
        
        def reset(self):
            """Reset both trackers"""
            self.base_tracker.reset()
            self.ball_tracker.reset()
            self.frame_id = 0
        
        def __getattr__(self, name):
            """Delegate other attributes to base tracker"""
            return getattr(self.base_tracker, name)
    
    return TrackerWithBallValidation(base_tracker, ball_tracker)
