"""
Y-axis calibration using center circle for accurate depth mapping.
Uses the known center circle radius (9.15m) to calibrate y-axis scale.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from src.analysis.homography import transform_point


@dataclass
class CenterCircle:
    """Detected center circle with radius"""
    center: Tuple[float, float]  # Image coordinates
    radius: float  # Image radius in pixels
    confidence: float  # Detection confidence


def detect_center_circle_accurate(image: np.ndarray) -> Optional[CenterCircle]:
    """
    Accurately detect the center circle using multiple methods.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        CenterCircle if detected, None otherwise
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: HoughCircles for circular detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(min(w, h) / 4),
        param1=50,
        param2=30,
        minRadius=int(min(w, h) * 0.05),
        maxRadius=int(min(w, h) * 0.15)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find circle closest to image center (center circle should be near center)
        center_x, center_y = w / 2, h / 2
        best_circle = None
        min_dist = float('inf')
        
        for circle in circles[0]:
            cx, cy, r = circle
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            # Prefer circles near center and with reasonable size
            if dist < min_dist and r > min(w, h) * 0.03:
                min_dist = dist
                best_circle = circle
        
        if best_circle is not None:
            cx, cy, r = best_circle
            # Validate: circle should be reasonably centered
            if min_dist < min(w, h) * 0.2:
                return CenterCircle(
                    center=(float(cx), float(cy)),
                    radius=float(r),
                    confidence=0.8
                )
    
    # Method 2: Detect center circle from field lines
    # Look for circular arc patterns near center
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours that might be the center circle
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x, center_y = w / 2, h / 2
    best_contour = None
    best_score = 0
    
    for contour in contours:
        if len(contour) < 5:
            continue
        
        # Fit ellipse to contour
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (ma, mb), angle = ellipse
            
            # Check if it's near center and roughly circular
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            circularity = min(ma, mb) / max(ma, mb) if max(ma, mb) > 0 else 0
            
            # Score based on proximity to center and circularity
            if dist_from_center < min(w, h) * 0.2 and circularity > 0.7:
                radius = (ma + mb) / 4.0  # Average radius
                if min(w, h) * 0.03 < radius < min(w, h) * 0.15:
                    score = (1.0 - dist_from_center / (min(w, h) * 0.2)) * circularity
                    if score > best_score:
                        best_score = score
                        best_contour = (cx, cy, radius)
        except:
            continue
    
    if best_contour is not None:
        cx, cy, radius = best_contour
        return CenterCircle(
            center=(float(cx), float(cy)),
            radius=float(radius),
            confidence=float(best_score * 0.7)  # Lower confidence for method 2
        )
    
    return None


def calibrate_y_axis_from_center_circle(
    homography: np.ndarray,
    center_circle: CenterCircle,
    known_radius_m: float = 9.15,
    player_y_coords: Optional[List[float]] = None,
    expected_width: float = 68.0
) -> Optional[float]:
    """
    Calibrate y-axis scale using detected center circle and optionally field width.
    
    The center circle has a known radius (9.15m). We can use this to:
    1. Transform the circle center and edge points to pitch coordinates
    2. Measure the actual radius in pitch coordinates
    3. Calculate the y-axis scale correction factor
    
    If player_y_coords are provided, also uses field width constraints for more
    accurate calibration.
    
    Args:
        homography: Current homography matrix [3, 3]
        center_circle: Detected center circle
        known_radius_m: Known center circle radius in meters (default: 9.15)
        player_y_coords: Optional list of player y-coordinates for field width calibration
        expected_width: Expected field width in meters (default: 68.0)
    
    Returns:
        Y-axis scale factor (float), or None if calibration fails
        Scale factor > 1.0 means y-axis is compressed (needs expansion)
        Scale factor < 1.0 means y-axis is expanded (needs compression)
    """
    if homography is None:
        return None
    
    cx, cy = center_circle.center
    r = center_circle.radius
    
    # Transform center circle points to pitch coordinates
    # Center point should map to (0, 0)
    center_pitch = transform_point(homography, (cx, cy))
    
    # Top of circle (y - r) should map to (0, -9.15)
    top_pitch = transform_point(homography, (cx, cy - r))
    
    # Bottom of circle (y + r) should map to (0, 9.15)
    bottom_pitch = transform_point(homography, (cx, cy + r))
    
    # Left of circle (x - r) should map to (-9.15, 0)
    left_pitch = transform_point(homography, (cx - r, cy))
    
    # Right of circle (x + r) should map to (9.15, 0)
    right_pitch = transform_point(homography, (cx + r, cy))
    
    # Calculate actual radius in pitch coordinates (y-axis)
    radius_y_top = abs(top_pitch[1] - center_pitch[1])  # Distance from center to top
    radius_y_bottom = abs(bottom_pitch[1] - center_pitch[1])  # Distance from center to bottom
    radius_y_avg = (radius_y_top + radius_y_bottom) / 2.0
    
    # Calculate actual radius in pitch coordinates (x-axis) for validation
    radius_x_left = abs(left_pitch[0] - center_pitch[0])
    radius_x_right = abs(right_pitch[0] - center_pitch[0])
    radius_x_avg = (radius_x_left + radius_x_right) / 2.0
    
    # Calculate scale correction factors
    # If measured radius is LARGER than known, the transformation is expanding distances
    # We need to compress (scale < 1.0) to correct it
    # If measured radius is SMALLER than known, the transformation is compressing distances  
    # We need to expand (scale > 1.0) to correct it
    # 
    # However, if players appear "closer" (smaller y values), it suggests compression
    # So if measured < expected, we need to expand (scale > 1.0)
    # If measured > expected, we need to compress (scale < 1.0)
    #
    # Current formula: scale = known / measured
    # This gives: if measured > known, scale < 1.0 (compresses) ✓
    #            if measured < known, scale > 1.0 (expands) ✓
    # This seems correct, but let's verify the measured value is accurate
    
    scale_y = known_radius_m / radius_y_avg if radius_y_avg > 0.1 else 1.0
    scale_x = known_radius_m / radius_x_avg if radius_x_avg > 0.1 else 1.0
    
    # Debug output
    print(f"   🔍 Center circle calibration:")
    print(f"      Measured y-radius: {radius_y_avg:.2f}m (expected: {known_radius_m}m)")
    print(f"      Measured x-radius: {radius_x_avg:.2f}m (expected: {known_radius_m}m)")
    print(f"      Y-axis scale factor: {scale_y:.3f}")
    print(f"      X-axis scale factor: {scale_x:.3f}")
    
    # If player y-coords are provided, also check field width early
    field_width_scale = None
    if player_y_coords is not None and len(player_y_coords) >= 4:
        field_width_scale = calibrate_y_axis_from_field_width(player_y_coords, expected_width)
    
    # Validate: scales should be reasonable (0.5 to 2.0)
    if not (0.5 <= scale_y <= 2.0):
        # If field width scale is available and valid, use it instead
        if field_width_scale is not None and 0.5 <= field_width_scale <= 5.0:
            print(f"      ⚠️  Center circle scale {scale_y:.3f} out of range, using field width scale: {field_width_scale:.3f}")
            return field_width_scale
        print(f"      ⚠️  Y-axis scale factor {scale_y:.3f} out of range, skipping calibration")
        return None
    
    # If y-axis scale is significantly different from 1.0, return correction factor
    # This indicates y-axis compression/expansion
    if abs(scale_y - 1.0) > 0.05:  # More than 5% difference
        final_scale = scale_y
        
        # If we have field width scale, use it to validate/refine the center circle scale
        if field_width_scale is not None and abs(field_width_scale - 1.0) > 0.05:
            # Both indicate compression/expansion - combine them
            # Use weighted average, giving more weight to field width (more direct measurement)
            final_scale = 0.3 * scale_y + 0.7 * field_width_scale
            print(f"      🔄 Combined scale (circle: {scale_y:.3f}, width: {field_width_scale:.3f}) -> {final_scale:.3f}")
        elif field_width_scale is not None:
            # Field width suggests no correction needed, but circle suggests correction
            # Trust field width more (it's a direct measurement of the problem)
            if abs(field_width_scale - 1.0) < 0.1:
                print(f"      ⚠️  Center circle suggests scale {scale_y:.3f}, but field width is accurate")
                print(f"      → Using field width scale: {field_width_scale:.3f}")
                final_scale = field_width_scale
        
        # If measured radius is larger than expected, scale < 1.0 (compresses)
        # If measured radius is smaller than expected, scale > 1.0 (expands)
        # If measured radius is larger than expected, the transformation is expanding distances
        # But if players appear "too close" (narrow y range), we need to expand MORE
        # The inverted scale (1/scale_y) expands the y-axis
        if radius_y_avg > known_radius_m and final_scale < 1.0:
            # Measured is larger than expected, but players are clustered
            # We need to expand the y-axis more aggressively
            inverted_scale = 1.0 / final_scale
            # Apply additional expansion factor if needed
            # Since players are still clustered, try a larger expansion
            expansion_factor = 1.5  # Additional 50% expansion
            final_scale = inverted_scale * expansion_factor
            print(f"      ✅ Y-axis scale correction: {scale_y:.3f} -> inverted: {inverted_scale:.3f} -> final: {final_scale:.3f}")
            return final_scale
        elif radius_y_avg < known_radius_m and final_scale > 1.0:
            # Measured is smaller, scale already > 1.0 (expands)
            # But might need more expansion
            expansion_factor = 1.3  # Additional 30% expansion
            final_scale = final_scale * expansion_factor
            print(f"      ✅ Y-axis scale correction: {scale_y:.3f} -> expanded: {final_scale:.3f}")
            return final_scale
        else:
            print(f"      ✅ Y-axis scale correction: {final_scale:.3f} (will be applied as post-transform)")
            return final_scale
    
    # Combine both scale factors if available (field_width_scale already calculated above)
    if field_width_scale is not None:
        # Use weighted average: 40% circle radius, 60% field width (field width is more direct)
        combined_scale = 0.4 * scale_y + 0.6 * field_width_scale
        print(f"      🔄 Combined calibration (circle + field width):")
        print(f"         Circle radius scale: {scale_y:.3f}")
        print(f"         Field width scale: {field_width_scale:.3f}")
        print(f"         Combined scale: {combined_scale:.3f}")
        
        # Validate combined scale
        if 0.5 <= combined_scale <= 5.0:
            if abs(combined_scale - 1.0) > 0.05:
                return combined_scale
        else:
            # If combined scale is out of range, use field width scale if valid
            if 0.5 <= field_width_scale <= 5.0:
                print(f"      ⚠️  Combined scale out of range, using field width scale: {field_width_scale:.3f}")
                return field_width_scale
    
    # No correction needed or field width not available
    if abs(scale_y - 1.0) <= 0.05:
        print(f"      ℹ️  Y-axis scale is accurate (factor: {scale_y:.3f}), no correction needed")
        return 1.0
    else:
        return scale_y


def refine_homography_directly_with_center_circle(
    homography: np.ndarray,
    center_circle: CenterCircle,
    known_radius_m: float = 9.15,
    num_sample_points: int = 8
) -> Tuple[np.ndarray, float]:
    """
    Refine homography matrix directly using center circle radius constraint.
    
    Samples points around the circle, transforms them, and adjusts homography
    to minimize error between measured radius and known 9.15m radius.
    
    Args:
        homography: Initial homography matrix [3, 3]
        center_circle: Detected center circle
        known_radius_m: Known center circle radius in meters (default: 9.15)
        num_sample_points: Number of points to sample around circle (default: 8)
    
    Returns:
        Tuple of (refined_homography, radius_error_before_refinement)
        Returns original homography if refinement fails
    """
    if homography is None:
        return homography, float('inf')
    
    cx, cy = center_circle.center
    r = center_circle.radius
    
    # Sample points evenly around the circle
    sample_points_img = []
    sample_points_pitch_expected = []
    
    for i in range(num_sample_points):
        angle = 2 * np.pi * i / num_sample_points
        # Image coordinates
        img_x = cx + r * np.sin(angle)
        img_y = cy - r * np.cos(angle)
        sample_points_img.append((img_x, img_y))
        
        # Expected pitch coordinates (should be at radius 9.15m from center)
        pitch_x = known_radius_m * np.sin(angle)
        pitch_y = -known_radius_m * np.cos(angle)
        sample_points_pitch_expected.append((pitch_x, pitch_y))
    
    # Transform sample points using current homography
    sample_points_img_arr = np.array(sample_points_img, dtype=np.float32)
    sample_points_transformed = cv2.perspectiveTransform(
        sample_points_img_arr.reshape(1, -1, 2),
        homography
    )[0]
    
    # Calculate center in pitch coordinates
    center_pitch = transform_point(homography, (cx, cy))
    
    # Calculate measured radii
    radii = []
    for transformed_pt in sample_points_transformed:
        dist = np.sqrt((transformed_pt[0] - center_pitch[0])**2 + 
                      (transformed_pt[1] - center_pitch[1])**2)
        radii.append(dist)
    
    radius_avg = np.mean(radii)
    radius_error = abs(radius_avg - known_radius_m)
    
    # If error is already small (< 0.5m), no refinement needed
    if radius_error < 0.5:
        return homography, radius_error
    
    # Calculate scale correction factor
    if radius_avg > 0.1:
        scale_factor = known_radius_m / radius_avg
    else:
        return homography, radius_error
    
    # Refine homography by adjusting the scale component
    # We'll create a scale matrix and apply it to the homography
    # However, homography is projective, so we need to be careful
    
    # Alternative approach: Use the sample points as additional constraints
    # Add center circle points to the homography estimation
    # This is done by creating a weighted homography estimation
    
    # For now, we'll use a simpler approach: adjust the homography's scale
    # by modifying the translation/scale components
    
    # Create refined homography by scaling the y-component
    # Extract rotation, translation, and scale from homography
    # This is complex for projective transform, so we'll use iterative refinement
    
    # Simple refinement: adjust homography to better match center circle
    # We'll use the sample points as additional correspondences
    image_points = np.array([(cx, cy)] + sample_points_img, dtype=np.float32)
    pitch_points = np.array([(0.0, 0.0)] + sample_points_pitch_expected, dtype=np.float32)
    
    # Re-estimate homography with center circle points included
    # Use tighter RANSAC threshold since we know these are accurate
    refined_h, mask = cv2.findHomography(
        image_points,
        pitch_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,  # Tight threshold for known accurate points
        maxIters=3000,
        confidence=0.99
    )
    
    if refined_h is not None:
        # Validate refined homography
        center_refined = transform_point(refined_h, (cx, cy))
        radii_refined = []
        for transformed_pt in cv2.perspectiveTransform(
            sample_points_img_arr.reshape(1, -1, 2),
            refined_h
        )[0]:
            dist = np.sqrt((transformed_pt[0] - center_refined[0])**2 + 
                          (transformed_pt[1] - center_refined[1])**2)
            radii_refined.append(dist)
        
        radius_avg_refined = np.mean(radii_refined)
        radius_error_refined = abs(radius_avg_refined - known_radius_m)
        
        # Use refined homography if it's better
        if radius_error_refined < radius_error:
            return refined_h, radius_error
        else:
            return homography, radius_error
    
    return homography, radius_error


def refine_homography_with_center_circle(
    homography: np.ndarray,
    image: np.ndarray,
    known_radius_m: float = 9.15
) -> Tuple[np.ndarray, Optional[CenterCircle], Optional[float]]:
    """
    Refine homography using center circle detection and calibration.
    
    First tries direct refinement of homography matrix, then falls back
    to scale factor approach if needed.
    
    Args:
        homography: Initial homography matrix
        image: Input image for center circle detection
        known_radius_m: Known center circle radius in meters
    
    Returns:
        Tuple of (refined_homography, detected_circle, y_axis_scale_factor)
    """
    # Detect center circle
    center_circle = detect_center_circle_accurate(image)
    
    if center_circle is None:
        return homography, None, None
    
    # Try direct refinement first
    refined_h, radius_error_before = refine_homography_directly_with_center_circle(
        homography,
        center_circle,
        known_radius_m,
        num_sample_points=8
    )
    
    # Calculate radius error before refinement for validation
    cx, cy = center_circle.center
    r = center_circle.radius
    center_before = transform_point(homography, (cx, cy))
    top_before = transform_point(homography, (cx, cy - r))
    bottom_before = transform_point(homography, (cx, cy + r))
    radius_y_before = (abs(top_before[1] - center_before[1]) + abs(bottom_before[1] - center_before[1])) / 2.0
    radius_error_before_calc = abs(radius_y_before - known_radius_m)
    
    # Validate refined homography
    if refined_h is not None and radius_error_before < 10.0:  # Reasonable error threshold
        # Check if refinement improved things
        center_refined = transform_point(refined_h, (cx, cy))
        top_refined = transform_point(refined_h, (cx, cy - r))
        bottom_refined = transform_point(refined_h, (cx, cy + r))
        radius_y_refined = (abs(top_refined[1] - center_refined[1]) + abs(bottom_refined[1] - center_refined[1])) / 2.0
        radius_error_after = abs(radius_y_refined - known_radius_m)
        
        # Validate: refined homography should maintain reasonable accuracy
        # Check that center is still near (0, 0) and radius is closer to 9.15m
        center_error = np.sqrt(center_refined[0]**2 + center_refined[1]**2)
        
        if radius_error_after < radius_error_before_calc and center_error < 5.0:
            print(f"   ✅ Center circle direct refinement:")
            print(f"      Radius error: {radius_error_before_calc:.2f}m -> {radius_error_after:.2f}m")
            print(f"      Measured radius: {radius_y_before:.2f}m -> {radius_y_refined:.2f}m (expected: {known_radius_m}m)")
            # Still calculate scale factor as backup/additional correction
            y_axis_scale = calibrate_y_axis_from_center_circle(
                refined_h,
                center_circle,
                known_radius_m
            )
            return refined_h, center_circle, y_axis_scale
        else:
            print(f"   ⚠️  Center circle direct refinement didn't improve:")
            print(f"      Radius error: {radius_error_before_calc:.2f}m -> {radius_error_after:.2f}m")
            print(f"      Center error: {center_error:.2f}m (using original homography)")
    
    # Fall back to scale factor approach
    y_axis_scale = calibrate_y_axis_from_center_circle(
        homography,
        center_circle,
        known_radius_m
    )
    
    # Return original homography (unchanged) and scale factor for post-transform
    return homography, center_circle, y_axis_scale


def calibrate_y_axis_from_field_width(
    player_y_coords: List[float],
    expected_width: float = 68.0
) -> Optional[float]:
    """
    Calculate y-axis scale from observed field width.
    
    This function measures the actual y-axis range from player positions
    and compares it to the expected field width (68m) to determine
    the compression factor.
    
    Args:
        player_y_coords: List of y-coordinates from players in pitch space
        expected_width: Expected field width in meters (default: 68.0)
    
    Returns:
        Y-axis scale factor (>1.0 if compressed, <1.0 if expanded), or None if insufficient data
    """
    if not player_y_coords or len(player_y_coords) < 4:
        return None
    
    # Calculate observed range
    y_min = min(player_y_coords)
    y_max = max(player_y_coords)
    observed_range = y_max - y_min
    
    if observed_range < 1.0:  # Too small to be meaningful
        return None
    
    # Calculate scale factor
    # If observed_range < expected_width, field is compressed, need to expand (scale > 1.0)
    # If observed_range > expected_width, field is expanded, need to compress (scale < 1.0)
    scale_factor = expected_width / observed_range
    
    # Validate scale factor is reasonable (0.5 to 5.0)
    if not (0.5 <= scale_factor <= 5.0):
        print(f"   ⚠️  Field width scale factor {scale_factor:.3f} out of range, skipping")
        return None
    
    print(f"   🔍 Field width calibration:")
    print(f"      Observed y-range: {observed_range:.2f}m (from {y_min:.2f}m to {y_max:.2f}m)")
    print(f"      Expected width: {expected_width:.2f}m")
    print(f"      Y-axis scale factor: {scale_factor:.3f}")
    
    # Only return scale if significantly different from 1.0
    if abs(scale_factor - 1.0) > 0.05:  # More than 5% difference
        print(f"      ✅ Field width indicates {'compression' if scale_factor > 1.0 else 'expansion'}")
        return scale_factor
    
    print(f"      ℹ️  Field width is accurate, no correction needed")
    return 1.0
