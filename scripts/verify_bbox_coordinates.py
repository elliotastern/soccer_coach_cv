#!/usr/bin/env python3
"""
Verify that bbox coordinates in results.json match what would be generated
by RF-DETR on the same frames. This helps identify if there's a model mismatch
or coordinate system issue.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_bboxes(results_path: str, video_path: str, model_path: str = None, num_frames: int = 5):
    """
    Verify bbox coordinates by optionally re-running detection
    
    Args:
        results_path: Path to results.json
        video_path: Path to source video
        model_path: Optional path to RF-DETR model (if None, just verify format)
        num_frames: Number of frames to test
    """
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
    print("BBOX COORDINATE VERIFICATION")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Video dimensions: {width}x{height}")
    print(f"Results: {results_path}")
    
    if model_path:
        print(f"Model: {model_path}")
        print("Will re-run detection and compare...")
        try:
            from rfdetr import RFDETRMedium
            detector = RFDETRMedium(pretrain_weights=model_path)
            detector.eval()
            print("✅ Model loaded")
            re_run = True
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("   Will only verify bbox format")
            re_run = False
    else:
        print("No model provided - will only verify bbox format")
        re_run = False
    
    print()
    
    frame_idx = 0
    mismatches = []
    
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
            frame_idx += 1
            continue
        
        print(f"\nFrame {frame_idx}: {len(result['players'])} players")
        print("-" * 70)
        
        # Optionally re-run detection
        fresh_detections = None
        if re_run:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections_raw = detector.predict(frame_rgb, threshold=0.3)
            
            fresh_detections = []
            if hasattr(detections_raw, 'class_id'):
                for i in range(len(detections_raw.class_id)):
                    if detections_raw.class_id[i] == 1:  # Person class
                        bbox_xyxy = detections_raw.xyxy[i]
                        x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
                        width_fresh = x_max - x_min
                        height_fresh = y_max - y_min
                        fresh_detections.append({
                            'bbox': [x_min, y_min, width_fresh, height_fresh],
                            'confidence': float(detections_raw.confidence[i])
                        })
        
        # Compare stored vs fresh
        for i, player in enumerate(result['players']):
            bbox = player['bbox']
            if len(bbox) != 4:
                continue
            
            x, y, w, h = [float(v) for v in bbox]
            stored_center = (x + w/2, y + h/2)
            
            # Check bounds
            in_bounds = (0 <= x < frame_w and 0 <= y < frame_h and 
                        x + w <= frame_w and y + h <= frame_h)
            
            if not in_bounds:
                print(f"  ⚠️  Player {i+1}: Bbox out of bounds!")
                print(f"      Bbox: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
                print(f"      Frame: {frame_w}x{frame_h}")
            
            # Compare with fresh detection if available
            if re_run and fresh_detections:
                # Find closest match
                min_dist = float('inf')
                closest_fresh = None
                
                for fresh in fresh_detections:
                    fx, fy, fw, fh = fresh['bbox']
                    fresh_center = (fx + fw/2, fy + fh/2)
                    dist = np.sqrt((stored_center[0] - fresh_center[0])**2 + 
                                 (stored_center[1] - fresh_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_fresh = fresh
                
                if closest_fresh:
                    fx, fy, fw, fh = closest_fresh['bbox']
                    fresh_center = (fx + fw/2, fy + fh/2)
                    
                    if min_dist > 20:  # Significant mismatch
                        mismatches.append({
                            'frame': frame_idx,
                            'player': i+1,
                            'stored': stored_center,
                            'fresh': fresh_center,
                            'distance': min_dist
                        })
                        print(f"  ⚠️  Player {i+1}: MISMATCH!")
                        print(f"      Stored center: ({stored_center[0]:.1f}, {stored_center[1]:.1f})")
                        print(f"      Fresh center: ({fresh_center[0]:.1f}, {fresh_center[1]:.1f})")
                        print(f"      Distance: {min_dist:.1f}px")
                        print(f"      Stored bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
                        print(f"      Fresh bbox: [{fx:.1f}, {fy:.1f}, {fw:.1f}, {fh:.1f}]")
        
        frame_idx += 1
    
    cap.release()
    
    print()
    print("="*70)
    if re_run:
        if mismatches:
            print(f"⚠️  FOUND {len(mismatches)} MISMATCHES")
            print("   Stored bboxes do NOT match fresh RF-DETR detections!")
            print("   This suggests the results.json was generated with a different model or preprocessing.")
        else:
            print("✅ NO MISMATCHES FOUND")
            print("   Stored bboxes match fresh RF-DETR detections.")
    else:
        print("✅ BBOX FORMAT VERIFICATION COMPLETE")
        print("   To compare with fresh detections, provide --model path")
    
    return mismatches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify bbox coordinates")
    parser.add_argument("results", help="Path to results.json")
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to RF-DETR model to re-run detection")
    parser.add_argument("--num-frames", type=int, default=5,
                       help="Number of frames to test")
    
    args = parser.parse_args()
    
    verify_bboxes(args.results, args.video, args.model, args.num_frames)
