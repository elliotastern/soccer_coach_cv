#!/usr/bin/env python3
"""
Quick test script that creates a synthetic soccer field image for testing.
No external images needed - generates test image automatically.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector
from src.analysis.pitch_line_segmentation import PitchLineSegmenter


def create_test_field_image():
    """Create a synthetic soccer field image with lines."""
    # Create green field
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = [34, 139, 34]  # Green color
    
    # Draw field lines (white)
    # Center line
    cv2.line(image, (0, 360), (1280, 360), (255, 255, 255), 8)
    
    # Touchlines (sidelines)
    cv2.line(image, (50, 0), (50, 720), (255, 255, 255), 6)
    cv2.line(image, (1230, 0), (1230, 720), (255, 255, 255), 6)
    
    # Goal lines
    cv2.line(image, (0, 100), (0, 620), (255, 255, 255), 6)
    cv2.line(image, (1280, 100), (1280, 620), (255, 255, 255), 6)
    
    # Center circle
    cv2.circle(image, (640, 360), 100, (255, 255, 255), 4)
    
    # Penalty boxes (left)
    cv2.rectangle(image, (0, 200), (200, 520), (255, 255, 255), 4)
    cv2.rectangle(image, (0, 280), (80, 440), (255, 255, 255), 4)  # Goal area
    
    # Penalty boxes (right)
    cv2.rectangle(image, (1080, 200), (1280, 520), (255, 255, 255), 4)
    cv2.rectangle(image, (1200, 280), (1280, 440), (255, 255, 255), 4)  # Goal area
    
    # Corner arcs
    cv2.ellipse(image, (0, 100), (30, 30), 0, 0, 90, (255, 255, 255), 3)
    cv2.ellipse(image, (0, 620), (30, 30), 270, 0, 90, (255, 255, 255), 3)
    cv2.ellipse(image, (1280, 100), (30, 30), 90, 0, 90, (255, 255, 255), 3)
    cv2.ellipse(image, (1280, 620), (30, 30), 180, 0, 90, (255, 255, 255), 3)
    
    return image


def main():
    print("=" * 60)
    print("Quick Segmentation Test (Synthetic Image)")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating synthetic soccer field image...")
    image = create_test_field_image()
    print(f"   Created image: {image.shape}")
    
    # Test segmentation module
    print("\n2. Testing Segmentation Module (color-based fallback)...")
    segmenter = PitchLineSegmenter(model_path=None)
    mask = segmenter.segment_pitch_lines(image)
    print(f"   Generated mask: {mask.shape}")
    print(f"   Line pixels detected: {np.sum(mask > 0)}")
    
    # Test keypoint detector
    print("\n3. Testing Keypoint Detector...")
    detector = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=False  # Use color-based for quick test
    )
    
    lines = detector._detect_field_lines(image)
    print(f"   Detected {len(lines)} lines")
    
    # Detect keypoints (without zero-shot to speed up)
    detector.enable_zero_shot = False
    keypoints = detector.detect_all_keypoints(image)
    print(f"   Detected {len(keypoints)} keypoints")
    
    if keypoints:
        from collections import Counter
        type_counts = Counter(kp.landmark_type for kp in keypoints)
        print("   Keypoint breakdown:")
        for kp_type, count in type_counts.items():
            print(f"     {kp_type}: {count}")
    
    # Visualize
    print("\n4. Creating visualization...")
    vis = image.copy()
    
    # Overlay mask
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Draw lines
    for line in lines[:20]:  # Limit to first 20 for clarity
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in keypoints:
        x, y = int(kp.image_point[0]), int(kp.image_point[1])
        color = (0, 0, 255) if kp.landmark_type == "goal" else (255, 0, 0)
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.putText(vis, kp.landmark_type[:4], (x + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save
    output_path = "test_segmentation_output.jpg"
    cv2.imwrite(output_path, vis)
    print(f"   Saved visualization to {output_path}")
    
    # Create comparison image
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([image, mask_3ch, vis])
    cv2.putText(comparison, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask", (image.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Result", (2*image.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    comparison_path = "test_segmentation_comparison.jpg"
    cv2.imwrite(comparison_path, comparison)
    print(f"   Saved comparison to {comparison_path}")
    
    print("\n✓ Quick test complete!")
    print("\nTo test with a real image:")
    print("  python scripts/test_segmentation_visual.py --image path/to/image.jpg")
    print("\nTo test with segmentation enabled (requires trained model):")
    print("  python scripts/test_segmentation_visual.py --image path/to/image.jpg --use_segmentation")


if __name__ == "__main__":
    main()
