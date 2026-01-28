#!/usr/bin/env python3
"""
Visual test script for semantic segmentation-based 2D mapping.
Tests segmentation on real images and visualizes results.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import argparse
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto
from src.analysis.pitch_line_segmentation import PitchLineSegmenter


def visualize_segmentation(image, mask, lines, keypoints=None, output_path=None):
    """
    Visualize segmentation results.
    
    Args:
        image: Original image
        mask: Segmentation mask
        lines: Detected lines
        keypoints: Optional keypoints to visualize
        output_path: Optional path to save visualization
    """
    h, w = image.shape[:2]
    
    # Create visualization
    vis = image.copy()
    
    # Overlay mask (semi-transparent)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Draw detected lines
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints if provided
    if keypoints:
        for kp in keypoints:
            x, y = int(kp.image_point[0]), int(kp.image_point[1])
            landmark_type = kp.landmark_type
            confidence = kp.confidence
            
            # Color by type
            if landmark_type == "goal":
                color = (0, 0, 255)  # Red
            elif landmark_type == "corner":
                color = (255, 0, 0)  # Blue
            elif landmark_type == "penalty_box":
                color = (255, 255, 0)  # Cyan
            elif landmark_type == "center_line":
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.circle(vis, (x, y), 8, color, 2)
            
            # Label
            label = f"{landmark_type[:4]}:{confidence:.2f}"
            cv2.putText(vis, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Create side-by-side comparison
    # Resize mask to match image
    mask_resized = cv2.resize(mask, (w, h))
    mask_3channel = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    
    # Combine: original | mask | visualization
    combined = np.hstack([image, mask_3channel, vis])
    
    # Add labels
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Segmentation Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Lines + Keypoints", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(str(output_path), combined)
        print(f"Saved visualization to {output_path}")
    
    return combined


def test_with_image(image_path: str, use_segmentation: bool = False, config_path: str = None):
    """Test segmentation on a single image."""
    print("=" * 60)
    print("Testing Semantic Segmentation on Image")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Load config if provided
    segmentation_config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'line_segmentation' in config:
                segmentation_config = config['line_segmentation']
                use_segmentation = segmentation_config.get('enabled', False)
                print(f"Loaded config: segmentation enabled = {use_segmentation}")
    
    # Test 1: Segmentation module directly
    print("\n1. Testing Segmentation Module...")
    segmenter = PitchLineSegmenter(
        model_path=segmentation_config.get('model_path', None) if segmentation_config else None,
        model_type=segmentation_config.get('model_type', 'deeplabv3') if segmentation_config else 'deeplabv3',
        threshold=segmentation_config.get('threshold', 0.5) if segmentation_config else 0.5,
        use_pretrained=segmentation_config.get('use_pretrained', True) if segmentation_config else True
    )
    
    mask = segmenter.segment_pitch_lines(image)
    print(f"   Generated mask: {mask.shape}, {np.sum(mask > 0)} line pixels detected")
    
    # Test 2: Keypoint detector with segmentation
    print("\n2. Testing Keypoint Detector with Segmentation...")
    detector = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=use_segmentation,
        segmentation_config=segmentation_config
    )
    
    # Detect lines
    lines = detector._detect_field_lines(image)
    print(f"   Detected {len(lines)} lines")
    
    # Detect keypoints
    keypoints = detector.detect_all_keypoints(image)
    print(f"   Detected {len(keypoints)} keypoints")
    
    if keypoints:
        print("   Keypoint types:")
        from collections import Counter
        type_counts = Counter(kp.landmark_type for kp in keypoints)
        for kp_type, count in type_counts.items():
            print(f"     {kp_type}: {count}")
    
    # Test 3: Convenience function
    print("\n3. Testing Convenience Function...")
    result = detect_pitch_keypoints_auto(
        image,
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=use_segmentation,
        segmentation_config=segmentation_config
    )
    
    if result:
        print(f"   Selected {len(result['image_points'])} keypoints for homography")
    else:
        print("   No keypoints selected (may need more detections)")
    
    # Visualize
    print("\n4. Creating Visualization...")
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_segmentation_test.jpg"
    vis = visualize_segmentation(image, mask, lines, keypoints, output_path)
    
    # Display (if possible)
    try:
        # Resize if too large
        h, w = vis.shape[:2]
        if w > 1920:
            scale = 1920 / w
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        
        cv2.imshow("Segmentation Test Results", vis)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("   (Display not available, saved to file)")
    
    print("\n✓ Test complete!")


def test_with_video(video_path: str, frame_number: int = 0, use_segmentation: bool = False, config_path: str = None):
    """Test segmentation on a video frame."""
    print("=" * 60)
    print("Testing Semantic Segmentation on Video Frame")
    print("=" * 60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    print(f"Loaded frame {frame_number}: {frame.shape}")
    
    # Test with image
    temp_image_path = f"/tmp/test_frame_{frame_number}.jpg"
    cv2.imwrite(temp_image_path, frame)
    
    test_with_image(temp_image_path, use_segmentation, config_path)
    
    # Cleanup
    Path(temp_image_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Test semantic segmentation for 2D mapping")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (for video)")
    parser.add_argument("--use_segmentation", action="store_true", 
                        help="Enable semantic segmentation (requires trained model)")
    parser.add_argument("--config", type=str, default="configs/goal_detection.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    if not args.image and not args.video:
        print("Error: Must provide either --image or --video")
        parser.print_help()
        return
    
    if args.image:
        test_with_image(args.image, args.use_segmentation, args.config)
    elif args.video:
        test_with_video(args.video, args.frame, args.use_segmentation, args.config)


if __name__ == "__main__":
    main()
