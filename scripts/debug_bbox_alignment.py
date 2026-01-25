#!/usr/bin/env python3
"""
Diagnostic script to test bounding box alignment.
Draws bboxes directly from results.json on video frames to verify coordinate system.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_bbox_drawing(results_path: str, video_path: str, output_dir: str, num_frames: int = 5):
    """
    Test bbox drawing directly on frames from results.json
    
    Args:
        results_path: Path to results.json
        video_path: Path to source video
        output_dir: Output directory for test images
        num_frames: Number of frames to test
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("="*70)
    print("BOUNDING BOX ALIGNMENT DIAGNOSTIC")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Video dimensions: {width}x{height}")
    print(f"Results: {results_path}")
    print(f"Testing {num_frames} frames")
    print()
    
    frame_idx = 0
    
    while frame_idx < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Verify frame dimensions
        frame_h, frame_w = frame.shape[:2]
        if frame_w != width or frame_h != height:
            print(f"⚠️  Frame {frame_idx}: Size mismatch! Video={width}x{height}, Frame={frame_w}x{frame_h}")
        
        # Find corresponding result
        result = None
        for r in results:
            if r.get('frame_id') == frame_idx:
                result = r
                break
        
        if not result or 'players' not in result:
            print(f"Frame {frame_idx}: No results found")
            frame_idx += 1
            continue
        
        # Create annotated frame
        frame_annotated = frame.copy()
        
        print(f"\nFrame {frame_idx}: {len(result['players'])} players")
        print("-" * 70)
        
        for i, player in enumerate(result['players']):
            bbox = player['bbox']
            if len(bbox) != 4:
                continue
            
            x, y, w, h = [float(v) for v in bbox]
            
            # Calculate center
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Get stored pixel center if available
            pixel_center = player.get('pixel_center', [])
            if pixel_center:
                stored_cx, stored_cy = pixel_center[0], pixel_center[1]
                center_match = abs(center_x - stored_cx) < 0.1 and abs(center_y - stored_cy) < 0.1
            else:
                stored_cx, stored_cy = None, None
                center_match = None
            
            # Check bounds
            in_bounds = (0 <= x < frame_w and 0 <= y < frame_h and 
                        x + w <= frame_w and y + h <= frame_h)
            
            # Print diagnostic info
            print(f"  Player {i+1}:")
            print(f"    Bbox: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
            print(f"    Calculated center: ({center_x:.1f}, {center_y:.1f})")
            if stored_cx is not None:
                print(f"    Stored center: ({stored_cx:.1f}, {stored_cy:.1f})")
                print(f"    Center match: {center_match}")
            print(f"    In bounds: {in_bounds}")
            print(f"    Bbox right: {x+w:.1f} (frame width: {frame_w})")
            print(f"    Bbox bottom: {y+h:.1f} (frame height: {frame_h})")
            
            # Draw bounding box
            x_int = int(x)
            y_int = int(y)
            x2_int = int(x + w)
            y2_int = int(y + h)
            
            # Color based on team
            team_id = player.get('team_id')
            if team_id == 0:
                color = (0, 0, 255)  # Red
            elif team_id == 1:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw box
            cv2.rectangle(frame_annotated, (x_int, y_int), (x2_int, y2_int), color, 3)
            
            # Draw center point
            cv2.circle(frame_annotated, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"P{i+1}"
            if team_id is not None:
                label = f"T{team_id} {label}"
            cv2.putText(frame_annotated, label, (x_int, y_int - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save test image
        output_path = output_dir / f"frame_{frame_idx:03d}_bbox_test.jpg"
        cv2.imwrite(str(output_path), frame_annotated)
        print(f"\n✅ Saved test image: {output_path}")
        
        frame_idx += 1
    
    cap.release()
    
    print()
    print("="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"Test images saved to: {output_dir}")
    print("\nReview the test images to verify bbox alignment with players.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test bounding box alignment")
    parser.add_argument("results", help="Path to results.json")
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("--output", default="output/bbox_debug", help="Output directory")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to test")
    
    args = parser.parse_args()
    
    test_bbox_drawing(args.results, args.video, args.output, args.num_frames)
