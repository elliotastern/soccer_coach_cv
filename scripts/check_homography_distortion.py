#!/usr/bin/env python3
"""
Check if existing homography was created with distortion correction
"""
import json
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.homography import HomographyEstimator, estimate_homography_auto
from src.analysis.undistortion import detect_fisheye_distortion
from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto


def check_homography_file(homography_path: str, video_path: str, frame_num: int = 0):
    """Check if homography file was created with distortion correction"""
    
    if not Path(homography_path).exists():
        print(f"❌ Homography file not found: {homography_path}")
        return
    
    # Load homography
    with open(homography_path, 'r') as f:
        data = json.load(f)
    
    H_loaded = np.array(data['homography'], dtype=np.float32)
    
    print(f"📄 Homography file: {homography_path}")
    print("=" * 70)
    
    # Load frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_num}")
        return
    
    # Detect landmarks
    print("\n1. Detecting landmarks on original frame...")
    keypoint_data = detect_pitch_keypoints_auto(
        frame,
        pitch_length=105.0,
        pitch_width=68.0,
        min_points=4,
        max_points=40
    )
    
    if keypoint_data is None:
        print("❌ No landmarks detected")
        return
    
    image_points = np.array(keypoint_data['image_points'], dtype=np.float32)
    pitch_points = np.array(keypoint_data['pitch_points'], dtype=np.float32)
    
    print(f"   ✅ Detected {len(image_points)} landmarks")
    
    # Check for fisheye
    h, w = frame.shape[:2]
    is_fisheye = detect_fisheye_distortion(image_points, pitch_points, (w, h))
    print(f"   Fisheye detected: {is_fisheye}")
    
    # Create new homography with distortion correction
    print("\n2. Creating homography WITH distortion correction...")
    H_with_correction = estimate_homography_auto(frame, correct_distortion=True)
    
    # Create new homography WITHOUT distortion correction
    print("3. Creating homography WITHOUT distortion correction...")
    H_without_correction = estimate_homography_auto(frame, correct_distortion=False)
    
    if H_with_correction is not None and H_without_correction is not None:
        # Compare
        diff_with = np.abs(H_loaded - H_with_correction).max()
        diff_without = np.abs(H_loaded - H_without_correction).max()
        
        print("\n4. Comparing homographies:")
        print(f"   Loaded vs WITH correction: max diff = {diff_with:.6f}")
        print(f"   Loaded vs WITHOUT correction: max diff = {diff_without:.6f}")
        
        if diff_with < diff_without:
            print("\n   ✅ Homography was likely created WITH distortion correction")
            print(f"      (closer match: {diff_with:.6f} vs {diff_without:.6f})")
        else:
            print("\n   ⚠️  Homography was likely created WITHOUT distortion correction")
            print(f"      (closer match: {diff_without:.6f} vs {diff_with:.6f})")
    
    # Test mapping accuracy
    print("\n5. Testing mapping accuracy:")
    from src.analysis.mapping import PitchMapper
    mapper_loaded = PitchMapper()
    mapper_loaded.set_homography(H_loaded)
    
    # Test center circle mapping
    center_pixel = (w/2, h/2)
    center_pitch = mapper_loaded.pixel_to_pitch(center_pixel[0], center_pixel[1])
    print(f"   Center pixel ({center_pixel[0]:.0f}, {center_pixel[1]:.0f}) -> ({center_pitch.x:.2f}, {center_pitch.y:.2f})m")
    print(f"   Expected center: (0.0, 0.0)m")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check if homography was created with distortion correction")
    parser.add_argument("homography", type=str, help="Path to homography JSON file")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to test")
    
    args = parser.parse_args()
    check_homography_file(args.homography, args.video, args.frame)
