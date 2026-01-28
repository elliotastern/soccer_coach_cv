#!/usr/bin/env python3
"""
Test semantic segmentation on a real soccer field image.
Creates detailed visualization showing what was detected.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto
from src.analysis.pitch_line_segmentation import PitchLineSegmenter


def create_realistic_field():
    """Create a more realistic synthetic soccer field for testing."""
    # Create larger, more realistic field
    h, w = 1080, 1920
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add gradient green field (more realistic)
    for y in range(h):
        green_intensity = 34 + int((y / h) * 20)  # Slight gradient
        image[y, :] = [green_intensity, 139, 34]
    
    # Draw field lines (white, varying thickness for perspective)
    line_thickness = 6
    
    # Center line (horizontal)
    cv2.line(image, (0, h//2), (w, h//2), (255, 255, 255), line_thickness)
    
    # Touchlines (vertical sidelines) - perspective effect
    cv2.line(image, (w//20, 0), (w//20, h), (255, 255, 255), line_thickness)
    cv2.line(image, (w - w//20, 0), (w - w//20, h), (255, 255, 255), line_thickness)
    
    # Goal lines (vertical, at ends)
    goal_line_y_start = int(h * 0.15)
    goal_line_y_end = int(h * 0.85)
    cv2.line(image, (0, goal_line_y_start), (0, goal_line_y_end), (255, 255, 255), line_thickness)
    cv2.line(image, (w-1, goal_line_y_start), (w-1, goal_line_y_end), (255, 255, 255), line_thickness)
    
    # Center circle
    cv2.circle(image, (w//2, h//2), int(min(w, h) * 0.12), (255, 255, 255), line_thickness)
    # Center spot
    cv2.circle(image, (w//2, h//2), 5, (255, 255, 255), -1)
    
    # Left penalty box
    penalty_box_width = int(w * 0.15)
    penalty_box_top = int(h * 0.25)
    penalty_box_bottom = int(h * 0.75)
    cv2.rectangle(image, (0, penalty_box_top), (penalty_box_width, penalty_box_bottom), 
                  (255, 255, 255), line_thickness)
    
    # Left goal area (6-yard box)
    goal_area_width = int(w * 0.06)
    goal_area_top = int(h * 0.35)
    goal_area_bottom = int(h * 0.65)
    cv2.rectangle(image, (0, goal_area_top), (goal_area_width, goal_area_bottom), 
                  (255, 255, 255), line_thickness)
    
    # Right penalty box
    cv2.rectangle(image, (w - penalty_box_width, penalty_box_top), 
                  (w, penalty_box_bottom), (255, 255, 255), line_thickness)
    
    # Right goal area
    cv2.rectangle(image, (w - goal_area_width, goal_area_top), 
                  (w, goal_area_bottom), (255, 255, 255), line_thickness)
    
    # Penalty spots
    penalty_spot_x = int(w * 0.12)
    cv2.circle(image, (penalty_spot_x, h//2), 5, (255, 255, 255), -1)
    cv2.circle(image, (w - penalty_spot_x, h//2), 5, (255, 255, 255), -1)
    
    # Corner arcs
    corner_radius = 30
    cv2.ellipse(image, (0, goal_line_y_start), (corner_radius, corner_radius), 0, 0, 90, (255, 255, 255), line_thickness)
    cv2.ellipse(image, (0, goal_line_y_end), (corner_radius, corner_radius), 270, 0, 90, (255, 255, 255), line_thickness)
    cv2.ellipse(image, (w-1, goal_line_y_start), (corner_radius, corner_radius), 90, 0, 90, (255, 255, 255), line_thickness)
    cv2.ellipse(image, (w-1, goal_line_y_end), (corner_radius, corner_radius), 180, 0, 90, (255, 255, 255), line_thickness)
    
    # Add some noise/texture to make it more realistic
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def create_detailed_visualization(image, mask, lines, keypoints, output_path):
    """Create a detailed visualization showing all detection results."""
    h, w = image.shape[:2]
    
    # Create a large canvas for detailed view
    canvas_h = h * 2
    canvas_w = w * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas.fill(40)  # Dark gray background
    
    # Panel 1: Original image (top left)
    canvas[0:h, 0:w] = image
    cv2.putText(canvas, "1. Original Image", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 2: Segmentation mask (top right)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    canvas[0:h, w:2*w] = mask_colored
    cv2.putText(canvas, "2. Segmentation Mask", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 3: Detected lines (bottom left)
    lines_vis = image.copy()
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    canvas[h:2*h, 0:w] = lines_vis
    num_lines = len(lines) if lines is not None and len(lines) > 0 else 0
    cv2.putText(canvas, f"3. Detected Lines ({num_lines})", (10, h + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 4: Keypoints overlay (bottom right)
    keypoints_vis = image.copy()
    
    # Draw lines first
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(keypoints_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    if keypoints:
        from collections import defaultdict
        type_counts = defaultdict(int)
        
        for kp in keypoints:
            x, y = int(kp.image_point[0]), int(kp.image_point[1])
            landmark_type = kp.landmark_type
            confidence = kp.confidence
            type_counts[landmark_type] += 1
            
            # Color by type
            if landmark_type == "goal":
                color = (0, 0, 255)  # Red
            elif landmark_type == "corner":
                color = (255, 0, 0)  # Blue
            elif landmark_type == "penalty_box":
                color = (255, 255, 0)  # Cyan
            elif landmark_type == "center_line":
                color = (0, 255, 255)  # Yellow
            elif landmark_type == "touchline":
                color = (255, 0, 255)  # Magenta
            elif landmark_type == "center_circle":
                color = (128, 255, 128)  # Light green
            else:
                color = (255, 255, 255)  # White
            
            # Draw keypoint
            cv2.circle(keypoints_vis, (x, y), 8, color, -1)
            cv2.circle(keypoints_vis, (x, y), 12, color, 2)
            
            # Label
            label = f"{landmark_type[:6]}:{confidence:.2f}"
            cv2.putText(keypoints_vis, label, (x + 15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add legend
        y_offset = 40
        for kp_type, count in sorted(type_counts.items()):
            if kp_type == "goal":
                color = (0, 0, 255)
            elif kp_type == "corner":
                color = (255, 0, 0)
            elif kp_type == "penalty_box":
                color = (255, 255, 0)
            elif kp_type == "center_line":
                color = (0, 255, 255)
            elif kp_type == "touchline":
                color = (255, 0, 255)
            elif kp_type == "center_circle":
                color = (128, 255, 128)
            else:
                color = (255, 255, 255)
            
            cv2.circle(keypoints_vis, (w - 200, y_offset), 6, color, -1)
            cv2.putText(keypoints_vis, f"{kp_type}: {count}", (w - 180, y_offset + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    canvas[h:2*h, w:2*w] = keypoints_vis
    num_keypoints = len(keypoints) if keypoints else 0
    cv2.putText(canvas, f"4. Lines + Keypoints ({num_keypoints})", 
               (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved detailed visualization to {output_path}")
    
    return canvas


def test_on_image(image_path=None, use_segmentation=False):
    """Test segmentation on an image."""
    print("=" * 70)
    print("Semantic Segmentation Test on Real Image")
    print("=" * 70)
    
    # Load or create image
    if image_path and Path(image_path).exists():
        print(f"\nLoading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        image_name = Path(image_path).stem
    else:
        print("\nNo image provided, creating realistic synthetic field...")
        image = create_realistic_field()
        image_name = "synthetic_field"
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h} pixels")
    
    # Test segmentation
    print("\n" + "-" * 70)
    print("Step 1: Segmentation")
    print("-" * 70)
    segmenter = PitchLineSegmenter(model_path=None)  # Use color-based fallback
    mask = segmenter.segment_pitch_lines(image)
    line_pixels = np.sum(mask > 0)
    print(f"  Segmentation mask generated: {mask.shape}")
    print(f"  Line pixels detected: {line_pixels:,} ({100*line_pixels/(h*w):.2f}% of image)")
    
    # Test line detection
    print("\n" + "-" * 70)
    print("Step 2: Line Detection (Hough Transform on Mask)")
    print("-" * 70)
    detector = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=use_segmentation
    )
    lines = detector._detect_field_lines(image)
    print(f"  Lines detected: {len(lines)}")
    
    # Test keypoint detection
    print("\n" + "-" * 70)
    print("Step 3: Keypoint Detection")
    print("-" * 70)
    detector.enable_zero_shot = False  # Disable for faster testing
    keypoints = detector.detect_all_keypoints(image)
    print(f"  Total keypoints detected: {len(keypoints)}")
    
    if keypoints:
        from collections import Counter
        type_counts = Counter(kp.landmark_type for kp in keypoints)
        print("\n  Keypoint breakdown:")
        for kp_type, count in sorted(type_counts.items()):
            avg_conf = np.mean([kp.confidence for kp in keypoints if kp.landmark_type == kp_type])
            print(f"    {kp_type:20s}: {count:3d} (avg confidence: {avg_conf:.2f})")
    
    # Create visualization
    print("\n" + "-" * 70)
    print("Step 4: Creating Visualization")
    print("-" * 70)
    output_path = f"{image_name}_segmentation_detailed.jpg"
    vis = create_detailed_visualization(image, mask, lines, keypoints, output_path)
    print(f"  Visualization saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Image: {w}x{h} pixels")
    print(f"  Line pixels: {line_pixels:,}")
    print(f"  Lines detected: {len(lines)}")
    print(f"  Keypoints: {len(keypoints)}")
    print(f"  Output: {output_path}")
    print("\n✓ Test complete!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test segmentation on real soccer field image")
    parser.add_argument("--image", type=str, help="Path to soccer field image (optional)")
    parser.add_argument("--use_segmentation", action="store_true", 
                       help="Use semantic segmentation model (requires trained model)")
    
    args = parser.parse_args()
    
    test_on_image(args.image, args.use_segmentation)


if __name__ == "__main__":
    main()
