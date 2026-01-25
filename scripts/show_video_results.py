#!/usr/bin/env python3
"""
Show results from processed video or demonstrate on a sample frame.
If video is corrupted, extracts a frame and processes it.
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


def process_single_frame_demo(frame_path: str, model_path: str, output_dir: str = "output/demo"):
    """Process a single frame and show results"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FRAME PROCESSING DEMONSTRATION")
    print("="*70)
    print(f"Frame: {frame_path}")
    print(f"Model: {model_path}")
    print()
    
    # Load frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"❌ Could not load frame: {frame_path}")
        return
    
    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    print()
    
    # Load model
    print("Loading RF-DETR model...")
    try:
        model = RFDETRMedium(pretrain_weights=model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Detect players
    print("\n🔍 Detecting players...")
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
    
    print(f"✅ Detected {len(detections)} players")
    print()
    
    if len(detections) == 0:
        print("No players detected in frame")
        return
    
    # Initialize homography (simple calibration)
    print("📐 Initializing homography...")
    homography_estimator = HomographyEstimator()
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
    pitch_mapper = PitchMapper(homography_matrix=homography_estimator.homography)
    print("✅ Homography initialized")
    print()
    
    # Train team clusterer
    print("🎯 Training team clusterer...")
    crops = extract_player_crops(frame, detections)
    crop_images = [crop for crop, _ in crops if crop is not None]
    
    if len(crop_images) >= 3:
        team_clusterer = TeamClusterer()
        success = team_clusterer.fit(crop_images, min_crops=3)
        if success:
            print("✅ Team clusterer trained")
            team_colors = team_clusterer.get_team_colors()
            if team_colors:
                print(f"   Team 0 HSV: {team_colors[0]}")
                print(f"   Team 1 HSV: {team_colors[1]}")
        else:
            print("⚠️  Team clusterer training failed")
            team_clusterer = None
    else:
        print(f"⚠️  Not enough crops ({len(crop_images)}) for team clustering")
        team_clusterer = None
    
    print()
    
    # Process each detection
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Player':<8} {'Team':<8} {'Role':<8} {'Pixel Center':<20} {'Pitch Position':<25} {'Confidence':<10}")
    print("-" * 70)
    
    results = []
    for i, det in enumerate(detections):
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
        conf = 0.0
        
        if team_clusterer and crop is not None:
            assignment = team_clusterer.predict(crop, (pitch_pos.x, pitch_pos.y))
            team_id = assignment.team_id
            role = assignment.role
            conf = assignment.confidence
        
        team_str = f"Team {team_id}" if team_id is not None else "Unassigned"
        
        print(f"{i+1:<8} {team_str:<8} {role:<8} "
              f"({center_x:6.0f},{center_y:6.0f}){'':<4} "
              f"({pitch_pos.x:6.2f}, {pitch_pos.y:6.2f}) m{'':<4} "
              f"{conf:.3f}")
        
        results.append({
            'player_id': i+1,
            'bbox': det.bbox,
            'pixel_center': (float(center_x), float(center_y)),
            'pitch_position': (float(pitch_pos.x), float(pitch_pos.y)),
            'team_id': team_id,
            'role': role,
            'team_confidence': float(conf),
            'detection_confidence': float(det.confidence)
        })
    
    print("="*70)
    print()
    
    # Save results
    output_json = output_dir / "frame_results.json"
    with open(output_json, 'w') as f:
        json.dump({
            'frame_path': frame_path,
            'frame_size': (width, height),
            'num_detections': len(detections),
            'team_clusterer_trained': team_clusterer is not None,
            'players': results
        }, f, indent=2)
    
    print(f"✅ Results saved to: {output_json}")
    print()
    
    # Create visualization
    vis_frame = frame.copy()
    team_colors_bgr = {
        0: (0, 0, 255),  # Red
        1: (255, 0, 0),  # Blue
        None: (0, 255, 255)  # Yellow for unassigned
    }
    
    for result in results:
        x, y, w, h = result['bbox']
        team_id = result['team_id']
        color = team_colors_bgr.get(team_id, (0, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # Draw label
        label = f"{result['role']}"
        if team_id is not None:
            label = f"T{team_id} {label}"
        cv2.putText(vis_frame, label, (int(x), int(y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pitch position
        px, py = result['pixel_center']
        pitch_x, pitch_y = result['pitch_position']
        info = f"({pitch_x:.1f}, {pitch_y:.1f})m"
        cv2.putText(vis_frame, info, (int(px-50), int(py)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    output_image = output_dir / "visualization.jpg"
    cv2.imwrite(str(output_image), vis_frame)
    print(f"✅ Visualization saved to: {output_image}")
    print()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a frame and show R-002 and R-003 results")
    parser.add_argument("frame", type=str, help="Input frame image file")
    parser.add_argument("--model", "-m", type=str, 
                       default="models/rf_detr_soccertrack/checkpoint_best_total.pth",
                       help="Path to RF-DETR checkpoint")
    parser.add_argument("--output", "-o", type=str, default="output/demo",
                       help="Output directory")
    
    args = parser.parse_args()
    
    process_single_frame_demo(args.frame, args.model, args.output)
