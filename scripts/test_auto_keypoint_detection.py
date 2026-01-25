#!/usr/bin/env python3
"""
Test script for automatic pitch keypoint detection.
Demonstrates automatic detection of goals, center line, center circle, etc.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto
from src.analysis.homography import HomographyEstimator
from src.analysis.pitch_landmarks import LandmarkDatabase


def visualize_keypoints(image: np.ndarray, keypoints, output_path: str = None):
    """
    Visualize detected keypoints on image
    
    Args:
        image: Input image
        keypoints: List of PitchKeypoint objects
        output_path: Optional path to save visualization
    """
    vis_image = image.copy()
    
    # Color map for different landmark types
    colors = {
        "goal": (0, 255, 0),  # Green
        "center_line": (255, 0, 0),  # Blue
        "center_circle": (255, 255, 0),  # Cyan
        "penalty_box": (0, 255, 255),  # Yellow
        "penalty_spot": (255, 165, 0),  # Orange
        "goal_area": (0, 128, 255),  # Light Blue
        "corner": (255, 0, 255),  # Magenta
        "corner_arc": (128, 0, 128),  # Purple
        "touchline": (192, 192, 192),  # Silver
    }
    
    for kp in keypoints:
        x, y = int(kp.image_point[0]), int(kp.image_point[1])
        color = colors.get(kp.landmark_type, (255, 255, 255))
        
        # Draw point
        cv2.circle(vis_image, (x, y), 8, color, -1)
        cv2.circle(vis_image, (x, y), 12, (255, 255, 255), 2)
        
        # Draw label
        label = f"{kp.landmark_type} ({kp.confidence:.2f})"
        cv2.putText(vis_image, label, (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pitch coordinates
        pitch_label = f"({kp.pitch_point[0]:.1f}, {kp.pitch_point[1]:.1f})m"
        cv2.putText(vis_image, pitch_label, (x + 15, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def test_keypoint_detection(image_path: str):
    """Test automatic keypoint detection on an image"""
    print("="*70)
    print("AUTOMATIC PITCH KEYPOINT DETECTION TEST")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    print(f"\n📸 Loaded image: {image_path}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Show landmark database info
    print("\n📋 Landmark Database:")
    landmark_db = LandmarkDatabase()
    counts = landmark_db.count_landmarks()
    print(f"   Total landmarks: {counts['total']}")
    print(f"   Primary (Priority 1): {counts['by_priority'][1]}")
    print(f"   Secondary (Priority 2): {counts['by_priority'][2]}")
    print(f"   Tertiary (Priority 3): {counts['by_priority'][3]}")
    
    # Detect keypoints
    print("\n🔍 Detecting pitch keypoints...")
    detector = PitchKeypointDetector()
    all_keypoints = detector.detect_all_keypoints(image)
    
    print(f"\n✅ Detected {len(all_keypoints)} keypoints:")
    print("-" * 70)
    
    # Group by type
    by_type = {}
    for kp in all_keypoints:
        if kp.landmark_type not in by_type:
            by_type[kp.landmark_type] = []
        by_type[kp.landmark_type].append(kp)
    
    for landmark_type, kps in by_type.items():
        print(f"  {landmark_type}: {len(kps)} points")
        for kp in kps:
            print(f"    - Image: ({kp.image_point[0]:.1f}, {kp.image_point[1]:.1f}) | "
                  f"Pitch: ({kp.pitch_point[0]:.1f}, {kp.pitch_point[1]:.1f})m | "
                  f"Confidence: {kp.confidence:.2f}")
    
    # Select best keypoints
    print("\n🎯 Selecting best keypoints for homography...")
    selected = detector.select_best_keypoints(all_keypoints, min_points=4, max_points=25)
    print(f"   Selected {len(selected)} keypoints for homography estimation (comprehensive system)")
    
    # Visualize
    print("\n📊 Creating visualization...")
    vis_image = visualize_keypoints(image, selected)
    
    output_path = "output/keypoint_detection_test.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"   ✅ Saved visualization to: {output_path}")
    
    # Test homography estimation
    print("\n🔧 Testing homography estimation with detected keypoints...")
    estimator = HomographyEstimator()
    
    # Use automatic detection
    success = estimator.estimate(image, use_auto_detection=True)
    
    if success:
        print("   ✅ Homography estimated successfully!")
        
        # Test some point transformations
        print("\n🧪 Testing point transformations:")
        print("-" * 70)
        
        test_points = [
            (image.shape[1] / 2, image.shape[0] / 2, "Center"),
            (image.shape[1] * 0.1, image.shape[0] / 2, "Left edge"),
            (image.shape[1] * 0.9, image.shape[0] / 2, "Right edge"),
        ]
        
        for px, py, label in test_points:
            pitch_pos = estimator.transform((px, py))
            if pitch_pos:
                print(f"   {label:15s}: Pixel ({px:.0f}, {py:.0f}) -> "
                      f"Pitch ({pitch_pos[0]:6.2f}, {pitch_pos[1]:6.2f}) m")
    else:
        print("   ⚠️  Homography estimation failed")
        print("   Try using manual calibration or check if pitch is clearly visible")
    
    print("\n" + "="*70)
    return success


def test_with_video_frame(video_path: str, frame_number: int = 0):
    """Test keypoint detection on a video frame"""
    print("="*70)
    print(f"TESTING KEYPOINT DETECTION ON VIDEO FRAME {frame_number}")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_number}")
        return False
    
    return test_keypoint_detection_frame(frame, frame_number)


def test_keypoint_detection_frame(frame: np.ndarray, frame_id: int = 0):
    """Test keypoint detection on a frame array"""
    print(f"\n📸 Processing frame {frame_id}")
    print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Detect keypoints using convenience function
    print("\n🔍 Detecting keypoints...")
    keypoint_data = detect_pitch_keypoints_auto(frame, min_points=4)
    
    if keypoint_data is None:
        print("   ⚠️  Could not detect enough keypoints")
        return False
    
    keypoints = keypoint_data['keypoints']
    print(f"   ✅ Detected {len(keypoints)} keypoints")
    
    # Visualize
    vis_image = visualize_keypoints(frame, keypoints)
    
    output_path = f"output/keypoint_detection_frame_{frame_id:03d}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"   ✅ Saved to: {output_path}")
    
    # Test homography
    estimator = HomographyEstimator()
    success = estimator.estimate(frame, use_auto_detection=True)
    
    if success:
        print("   ✅ Homography estimated successfully")
    else:
        print("   ⚠️  Homography estimation failed")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test automatic pitch keypoint detection")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (for video)")
    
    args = parser.parse_args()
    
    if args.image:
        test_keypoint_detection(args.image)
    elif args.video:
        test_with_video_frame(args.video, args.frame)
    else:
        print("Please provide either --image or --video argument")
        print("\nExample usage:")
        print("  python scripts/test_auto_keypoint_detection.py --image path/to/image.jpg")
        print("  python scripts/test_auto_keypoint_detection.py --video path/to/video.mp4 --frame 100")
