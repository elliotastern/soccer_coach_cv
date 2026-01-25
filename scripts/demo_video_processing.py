#!/usr/bin/env python3
"""
Demonstration script for processing a video with R-002 and R-003.
Shows team ID assignment and pitch mapping on real video data.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr import RFDETRMedium
from src.types import Detection
from src.logic.team_id import TeamClusterer
from src.perception.team import extract_player_crops, extract_player_crop
from src.analysis.homography import HomographyEstimator
from src.analysis.mapping import PitchMapper


def process_video_demo(video_path: str, model_path: str, output_dir: str = "output/demo", max_frames: int = 50):
    """Process video and show results"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VIDEO PROCESSING DEMONSTRATION")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load model
    print("Loading RF-DETR model...")
    try:
        model = RFDETRMedium(pretrain_weights=model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print()
    
    # Initialize components
    team_clusterer = TeamClusterer()
    homography_estimator = HomographyEstimator()
    pitch_mapper = PitchMapper()
    
    # Accumulate crops for Golden Batch
    golden_batch_crops = []
    frame_count = 0
    results = []
    
    print("Processing frames...")
    print("-" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        timestamp = frame_count / fps
        
        # Detect players
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections_raw = model.predict(frame_rgb, threshold=0.3)
        
        # Convert to Detection objects
        detections = []
        if hasattr(detections_raw, 'class_id'):
            for i in range(len(detections_raw.class_id)):
                class_id = int(detections_raw.class_id[i])
                if class_id == 1:  # COCO person class
                    confidence = float(detections_raw.confidence[i])
                    bbox_xyxy = detections_raw.xyxy[i]
                    x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
                    
                    detections.append(Detection(
                        class_id=0,
                        confidence=confidence,
                        bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                        class_name="player"
                    ))
        
        if len(detections) == 0:
            frame_count += 1
            continue
        
        # Extract crops for Golden Batch (first 20 frames)
        if frame_count < 20:
            crops = extract_player_crops(frame, detections)
            for crop, _ in crops:
                if crop is not None:
                    golden_batch_crops.append(crop)
        
        # Train team clusterer after collecting enough crops
        if frame_count == 20 and len(golden_batch_crops) >= 10 and not team_clusterer.is_trained():
            print(f"\n🎯 Training team clusterer with {len(golden_batch_crops)} crops...")
            success = team_clusterer.fit(golden_batch_crops, min_crops=10)
            if success:
                print("✅ Team clusterer trained")
                team_colors = team_clusterer.get_team_colors()
                if team_colors:
                    print(f"   Team 0 HSV: {team_colors[0]}")
                    print(f"   Team 1 HSV: {team_colors[1]}")
            print()
        
        # Initialize homography on first frame (simple calibration)
        if frame_count == 0:
            # Use a simple calibration based on frame dimensions
            # In production, this would use the calibration tool
            manual_points = {
                'image_points': [
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ],
                'pitch_points': [
                    [-52.5, -34.0],
                    [52.5, -34.0],
                    [52.5, 34.0],
                    [-52.5, 34.0]
                ]
            }
            homography_estimator.estimate(frame, manual_points)
            if homography_estimator.homography is not None:
                pitch_mapper.homography = homography_estimator.homography
                print("✅ Homography initialized")
                print()
        
        # Process detections
        frame_result = {
            'frame_id': frame_count,
            'timestamp': timestamp,
            'players': []
        }
        
        for det in detections:
            # Extract crop
            crop = extract_player_crop(frame, det.bbox)
            
            # Get pitch position
            x, y, w, h = det.bbox
            center_x = x + w / 2
            center_y = y + h / 2
            pitch_pos = pitch_mapper.pixel_to_pitch(center_x, center_y)
            
            # Assign team
            team_id = None
            role = "PLAYER"
            confidence = 0.0
            
            if team_clusterer.is_trained() and crop is not None:
                assignment = team_clusterer.predict(crop, (pitch_pos.x, pitch_pos.y))
                team_id = assignment.team_id
                role = assignment.role
                confidence = assignment.confidence
            
            frame_result['players'].append({
                'bbox': [float(x) for x in det.bbox],
                'pixel_center': [float(center_x), float(center_y)],
                'pitch_position': [float(pitch_pos.x), float(pitch_pos.y)],
                'team_id': int(team_id) if team_id is not None else None,
                'role': str(role),
                'confidence': float(confidence),
                'detection_confidence': float(det.confidence)
            })
        
        results.append(frame_result)
        
        # Progress update
        if (frame_count + 1) % 10 == 0:
            num_players = sum(len(r['players']) for r in results[-10:])
            print(f"  Frame {frame_count + 1:3d}: {len(detections)} detections, "
                  f"{num_players} players processed (last 10 frames)")
        
        frame_count += 1
    
    cap.release()
    
    # Save results
    output_json = output_dir / "results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {sum(len(r['players']) for r in results)}")
    print(f"Team clusterer trained: {team_clusterer.is_trained()}")
    print(f"Homography initialized: {homography_estimator.homography is not None}")
    print()
    print("Sample results (first 5 frames with detections):")
    print("-" * 70)
    
    for i, result in enumerate(results[:5]):
        if len(result['players']) > 0:
            print(f"\nFrame {result['frame_id']} (t={result['timestamp']:.2f}s):")
            for j, player in enumerate(result['players'][:3]):  # Show first 3 players
                team_str = f"Team {player['team_id']}" if player['team_id'] is not None else "Unassigned"
                print(f"  Player {j+1}: {team_str} | Role: {player['role']} | "
                      f"Pitch: ({player['pitch_position'][0]:.2f}, {player['pitch_position'][1]:.2f}) m | "
                      f"Conf: {player['confidence']:.2f}")
    
    print()
    print(f"✅ Results saved to: {output_json}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate video processing with R-002 and R-003")
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("--model", "-m", type=str, 
                       default="models/rf_detr_soccertrack/checkpoint_best_total.pth",
                       help="Path to RF-DETR checkpoint")
    parser.add_argument("--output", "-o", type=str, default="output/demo",
                       help="Output directory")
    parser.add_argument("--max-frames", type=int, default=50,
                       help="Maximum frames to process")
    
    args = parser.parse_args()
    
    process_video_demo(args.video, args.model, args.output, args.max_frames)
