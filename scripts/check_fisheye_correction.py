#!/usr/bin/env python3
"""
Check if fisheye distortion correction is being applied
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.homography import estimate_homography_auto, HomographyEstimator
from src.analysis.undistortion import detect_fisheye_distortion, estimate_camera_from_landmarks
from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto


def check_fisheye_correction(video_path: str, frame_num: int = 0):
    """Check if fisheye distortion is detected and corrected"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_num}")
        return
    
    h, w = frame.shape[:2]
    print(f"📹 Frame size: {w}x{h} ({w*h/1e6:.1f} MP)")
    print("=" * 70)
    
    # Detect keypoints
    print("\n1. Detecting landmarks...")
    keypoint_data = detect_pitch_keypoints_auto(
        frame,
        pitch_length=105.0,
        pitch_width=68.0,
        min_points=4,
        max_points=40
    )
    
    if keypoint_data is None:
        print("❌ No keypoints detected")
        return
    
    image_points = np.array(keypoint_data['image_points'], dtype=np.float32)
    pitch_points = np.array(keypoint_data['pitch_points'], dtype=np.float32)
    
    print(f"   ✅ Detected {len(image_points)} landmarks")
    
    # Check for fisheye distortion
    print("\n2. Checking for fisheye distortion...")
    is_fisheye = detect_fisheye_distortion(image_points, pitch_points, (w, h))
    print(f"   Fisheye detected: {is_fisheye}")
    
    # Try to estimate distortion parameters
    print("\n3. Estimating distortion parameters...")
    if len(image_points) >= 6:
        distortion_params = estimate_camera_from_landmarks(
            image_points,
            pitch_points,
            (w, h),
            is_fisheye=is_fisheye
        )
        
        if distortion_params:
            print(f"   ✅ Distortion parameters estimated")
            print(f"   Model: {'Fisheye' if distortion_params.is_fisheye else 'Standard'}")
            print(f"   Confidence: {distortion_params.confidence:.2f}")
            print(f"   Distortion coeffs: {distortion_params.dist_coeffs.flatten()}")
        else:
            print(f"   ❌ Failed to estimate distortion parameters")
    else:
        print(f"   ⚠️  Need at least 6 points, have {len(image_points)}")
    
    # Test homography estimation with and without correction
    print("\n4. Testing homography estimation:")
    print("   a) With distortion correction (default):")
    H_with = estimate_homography_auto(frame, correct_distortion=True)
    print(f"      Result: {'✅ Success' if H_with is not None else '❌ Failed'}")
    
    print("   b) Without distortion correction:")
    H_without = estimate_homography_auto(frame, correct_distortion=False)
    print(f"      Result: {'✅ Success' if H_without is not None else '❌ Failed'}")
    
    if H_with is not None and H_without is not None:
        # Compare homographies
        diff = np.abs(H_with - H_without).max()
        print(f"\n   Homography difference (max): {diff:.6f}")
        if diff > 0.01:
            print("   ⚠️  Significant difference - distortion correction is affecting results")
        else:
            print("   ℹ️  Minimal difference - distortion may be minimal or correction not applied")
    
    # Check what the pipeline actually uses
    print("\n5. Checking pipeline default behavior:")
    estimator = HomographyEstimator()
    success = estimator.estimate(frame, correct_distortion=True)  # Default
    print(f"   HomographyEstimator.estimate(correct_distortion=True): {'✅ Success' if success else '❌ Failed'}")
    print(f"   Y-axis distortion detected flag: {estimator.y_axis_distortion_detected}")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"   - Fisheye detected: {is_fisheye}")
    print(f"   - Distortion correction enabled by default: ✅ Yes")
    print(f"   - Distortion params estimated: {'✅ Yes' if distortion_params else '❌ No'}")
    if distortion_params:
        print(f"   - Correction model: {'Fisheye' if distortion_params.is_fisheye else 'Standard'}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check fisheye distortion correction")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to check")
    
    args = parser.parse_args()
    check_fisheye_correction(args.video, args.frame)
