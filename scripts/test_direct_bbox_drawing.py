#!/usr/bin/env python3
"""
Test drawing bboxes directly on frame without canvas to verify coordinate system.
This isolates bbox drawing from canvas placement issues.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_direct_drawing(results_path: str, video_path: str, output_dir: str, num_frames: int = 3):
    """
    Draw bboxes directly on frame (no canvas) to test coordinate system
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
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("="*70)
    print("DIRECT BBOX DRAWING TEST")
    print("="*70)
    print(f"Video: {width}x{height}")
    print(f"Results: {len(results)} frames")
    print()
    
    frame_idx = 0
    
    while frame_idx < num_frames and cap.isOpened():
        # Read exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find corresponding result
        result = None
        for r in results:
            if r.get('frame_id') == frame_idx:
                result = r
                break
        
        if not result or 'players' not in result:
            frame_idx += 1
            continue
        
        print(f"Frame {frame_idx}: {len(result['players'])} players")
        print("-" * 70)
        
        # Draw directly on frame (no canvas, no offset)
        frame_annotated = frame.copy()
        
        # Draw reference markers at corners and center
        cv2.circle(frame_annotated, (0, 0), 10, (0, 255, 0), -1)  # Top-left
        cv2.circle(frame_annotated, (width-1, 0), 10, (0, 255, 0), -1)  # Top-right
        cv2.circle(frame_annotated, (width-1, height-1), 10, (0, 255, 0), -1)  # Bottom-right
        cv2.circle(frame_annotated, (0, height-1), 10, (0, 255, 0), -1)  # Bottom-left
        cv2.circle(frame_annotated, (width//2, height//2), 10, (255, 0, 255), -1)  # Center
        
        # Draw grid lines every 500 pixels
        for x in range(0, width, 500):
            cv2.line(frame_annotated, (x, 0), (x, height), (128, 128, 128), 1)
        for y in range(0, height, 200):
            cv2.line(frame_annotated, (0, y), (width, y), (128, 128, 128), 1)
        
        # Draw bboxes using EXACT same logic as debug_bbox_alignment.py
        for i, player in enumerate(result['players']):
            bbox = player['bbox']
            if len(bbox) != 4:
                continue
            
            x, y, w, h = [float(v) for v in bbox]
            
            # Convert to integers (same as debug script)
            x_int = int(x)
            y_int = int(y)
            x2_int = int(x + w)
            y2_int = int(y + h)
            
            # Color by team
            team_id = player.get('team_id')
            if team_id == 0:
                color = (0, 0, 255)  # Red
            elif team_id == 1:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw box (EXACT same as debug_bbox_alignment.py)
            cv2.rectangle(frame_annotated, (x_int, y_int), (x2_int, y2_int), color, 3)
            
            # Draw center point
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(frame_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"P{i+1}"
            if team_id is not None:
                label = f"T{team_id} {label}"
            cv2.putText(frame_annotated, label, (x_int, y_int - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Print first few for debugging
            if i < 3:
                print(f"  Player {i+1}: bbox=({x_int}, {y_int}, {x2_int}, {y2_int}), center=({center_x}, {center_y})")
        
        # Save test image
        output_path = output_dir / f"direct_drawing_frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(output_path), frame_annotated)
        print(f"✅ Saved: {output_path}")
        print()
        
        frame_idx += 1
    
    cap.release()
    print("="*70)
    print("DIRECT DRAWING TEST COMPLETE")
    print("="*70)
    print(f"Test images saved to: {output_dir}")
    print("Review images to verify bbox alignment with players.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test direct bbox drawing")
    parser.add_argument("results", help="Path to results.json")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="output/direct_drawing_test",
                       help="Output directory")
    parser.add_argument("--num-frames", "-n", type=int, default=3,
                       help="Number of frames to test")
    
    args = parser.parse_args()
    
    test_direct_drawing(args.results, args.video, args.output, args.num_frames)
