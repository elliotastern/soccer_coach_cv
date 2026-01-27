#!/usr/bin/env python3
"""
Analyze landmark detection system - count landmarks detected per frame/video
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto


def analyze_landmark_detection(video_path: str, max_frames: int = 50, output_dir: str = None):
    """
    Analyze landmark detection across video frames
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to analyze
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir) if output_dir else Path("output/landmark_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    detector = PitchKeypointDetector(pitch_length=105.0, pitch_width=68.0)
    
    frame_stats = []
    landmark_type_counts = defaultdict(list)
    total_landmarks_by_type = defaultdict(int)
    
    print("🔍 Analyzing landmark detection across frames...")
    print("=" * 70)
    print(f"Processing {max_frames} frames (this may take a few minutes)...")
    print()
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect all keypoints
        try:
            all_keypoints = detector.detect_all_keypoints(frame)
        except Exception as e:
            print(f"   ⚠️  Frame {frame_idx}: Error detecting keypoints: {e}")
            all_keypoints = []
        
        # Count by type
        type_counts = defaultdict(int)
        for kp in all_keypoints:
            landmark_type = kp.landmark_type if hasattr(kp, 'landmark_type') else 'unknown'
            type_counts[landmark_type] += 1
            total_landmarks_by_type[landmark_type] += 1
        
        # Select best keypoints (as done in pipeline)
        try:
            selected_keypoints = detector.select_best_keypoints(
                all_keypoints,
                min_points=4,
                max_points=40,
                image=frame,
                prioritize_y_axis=True
            )
        except Exception as e:
            print(f"   ⚠️  Error selecting keypoints: {e}")
            selected_keypoints = all_keypoints[:40]  # Fallback: just take first 40
        
        # Count selected by type
        selected_type_counts = defaultdict(int)
        for kp in selected_keypoints:
            landmark_type = kp.landmark_type if hasattr(kp, 'landmark_type') else 'unknown'
            selected_type_counts[landmark_type] += 1
        
        frame_stat = {
            'frame_id': frame_idx,
            'total_detected': len(all_keypoints),
            'total_selected': len(selected_keypoints),
            'by_type_detected': dict(type_counts),
            'by_type_selected': dict(selected_type_counts)
        }
        frame_stats.append(frame_stat)
        
        # Store for statistics
        for landmark_type, count in type_counts.items():
            landmark_type_counts[landmark_type].append(count)
        
        # Print progress
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"Frame {frame_idx:3d}: {len(all_keypoints):2d} detected, {len(selected_keypoints):2d} selected | "
                  f"Types: {', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()) if v > 0)}")
        
        frame_idx += 1
    
    cap.release()
    
    # Calculate statistics
    total_detected = [s['total_detected'] for s in frame_stats]
    total_selected = [s['total_selected'] for s in frame_stats]
    
    stats = {
        'total_frames': len(frame_stats),
        'landmarks_detected': {
            'mean': float(np.mean(total_detected)),
            'median': float(np.median(total_detected)),
            'min': int(np.min(total_detected)),
            'max': int(np.max(total_detected)),
            'std': float(np.std(total_detected))
        },
        'landmarks_selected': {
            'mean': float(np.mean(total_selected)),
            'median': float(np.median(total_selected)),
            'min': int(np.min(total_selected)),
            'max': int(np.max(total_selected)),
            'std': float(np.std(total_selected))
        },
        'by_type_average': {
            landmark_type: {
                'mean': float(np.mean(counts)),
                'median': float(np.median(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
                'total': int(sum(counts))
            }
            for landmark_type, counts in landmark_type_counts.items()
        },
        'by_type_total': dict(total_landmarks_by_type),
        'frame_details': frame_stats
    }
    
    # Save results
    stats_path = output_dir / "landmark_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("LANDMARK DETECTION SUMMARY")
    print("=" * 70)
    print(f"Frames analyzed: {stats['total_frames']}")
    print()
    print("📊 Landmarks Detected (all):")
    print(f"   Mean:   {stats['landmarks_detected']['mean']:.1f}")
    print(f"   Median: {stats['landmarks_detected']['median']:.1f}")
    print(f"   Range:  {stats['landmarks_detected']['min']} - {stats['landmarks_detected']['max']}")
    print(f"   Std:    {stats['landmarks_detected']['std']:.1f}")
    print()
    print("📊 Landmarks Selected (for homography):")
    print(f"   Mean:   {stats['landmarks_selected']['mean']:.1f}")
    print(f"   Median: {stats['landmarks_selected']['median']:.1f}")
    print(f"   Range:  {stats['landmarks_selected']['min']} - {stats['landmarks_selected']['max']}")
    print(f"   Std:    {stats['landmarks_selected']['std']:.1f}")
    print()
    print("📊 By Landmark Type (average per frame):")
    for landmark_type in sorted(stats['by_type_average'].keys()):
        type_stat = stats['by_type_average'][landmark_type]
        print(f"   {landmark_type:20s}: mean={type_stat['mean']:5.1f}, "
              f"range={type_stat['min']}-{type_stat['max']}, total={type_stat['total']}")
    print()
    print(f"✅ Results saved to: {stats_path}")
    print("=" * 70)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze landmark detection per frame")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--max-frames", "-n", type=int, default=50,
                       help="Maximum number of frames to analyze")
    parser.add_argument("--output", "-o", type=str, default="output/landmark_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    analyze_landmark_detection(args.video, args.max_frames, args.output)
