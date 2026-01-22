#!/usr/bin/env python3
"""
Analyze first 100 frames using trained ball detection model (20 epochs, SoccerSynth_sub) WITH SAHI
Same structure as v2_20_epochs_sub_synth_plus_sahi but uses different video: 37CAE053-841F-4851-956E-CBF17A51C506.mp4
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.perception.tracker import Tracker
from src.types import Detection
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs


def detect_with_trained_model_sahi(frame: np.ndarray, model=None) -> list:
    """
    Use trained ball detection model (20 epochs on SoccerSynth_sub) WITH SAHI
    Model only detects balls (single class), not players
    SAHI splits image into overlapping tiles for better tiny object detection
    """
    from scripts.evaluate_ball_model import load_model, detect_balls
    
    # Load model if not provided (lazy loading)
    if model is None:
        checkpoint_path = "models/ball_detection/checkpoint.pth"
        model = load_model(checkpoint_path)
    
    # Detect balls using trained model WITH SAHI
    # SAHI settings: slice_size=1288 (matches training resolution), overlap=20%
    ball_detections = detect_balls(
        model, 
        frame, 
        confidence_threshold=0.3,
        use_sahi=True,
        sahi_slice_size=1288,
        sahi_overlap_ratio=0.2
    )
    
    # Convert to Detection format (ball-only)
    detections = []
    for ball_det in ball_detections:
        detections.append(Detection(
            class_id=0,  # Ball class ID
            confidence=ball_det.confidence,
            bbox=ball_det.bbox,  # (x, y, width, height)
            class_name='ball'
        ))
    
    # Note: This model doesn't detect players, only balls
    # So we return only ball detections
    return detections


def analyze_frames(video_path: str, num_frames: int = 100):
    """
    Analyze first N frames and report metrics
    """
    print(f"📊 Analyzing first {num_frames} frames with TRAINED MODEL + SAHI (20 epochs, SoccerSynth_sub)")
    print("=" * 60)
    
    # Load trained model once
    print("Loading trained ball detection model...")
    checkpoint_path = "models/ball_detection/checkpoint.pth"
    from scripts.evaluate_ball_model import load_model
    trained_model = load_model(checkpoint_path)
    print("✅ Model loaded")
    print("✅ SAHI enabled: Slice size=1288, Overlap=20%\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS")
    print(f"SAHI will slice {width}x{height} into {1288}x{1288} overlapping tiles\n")
    
    # Initialize tracker
    tracker = Tracker(
        track_thresh=0.3,
        high_thresh=0.5,
        track_buffer=30,
        match_thresh=0.7,
        frame_rate=int(fps)
    )
    
    # Metrics storage
    frame_metrics = []
    all_detections = []
    all_tracked = []
    
    print("Processing frames with SAHI...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects using trained model WITH SAHI
        detections = detect_with_trained_model_sahi(frame, model=trained_model)
        
        # Count by class (only balls for this model)
        players = [d for d in detections if d.class_name == 'player']
        balls = [d for d in detections if d.class_name == 'ball']
        
        # Track objects
        try:
            tracked_objects = tracker.update(detections, frame)
            tracked_players = [obj for obj in tracked_objects if obj.class_name == 'player']
            tracked_balls = [obj for obj in tracked_objects if obj.class_name == 'ball']
        except Exception:
            tracked_objects = []
            tracked_players = []
            tracked_balls = []
        
        # Store metrics
        frame_metrics.append({
            'frame': frame_num,
            'detections_players': len(players),
            'detections_balls': len(balls),
            'detections_total': len(detections),
            'tracked_players': len(tracked_players),
            'tracked_balls': len(tracked_balls),
            'tracked_total': len(tracked_objects),
            'player_confidences': [p.confidence for p in players],
            'ball_confidences': [b.confidence for b in balls]
        })
        
        all_detections.append(len(detections))
        all_tracked.append(len(tracked_objects))
        
        if (frame_num + 1) % 10 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("📈 DETECTION METRICS (First 100 Frames) - TRAINED MODEL + SAHI")
    print("=" * 60)
    
    # Players (will be 0 for this model)
    player_detections = [m['detections_players'] for m in frame_metrics]
    player_tracked = [m['tracked_players'] for m in frame_metrics]
    
    print(f"\n👥 PLAYERS:")
    print(f"  Detections per frame:")
    print(f"    Average: {np.mean(player_detections):.2f}")
    print(f"    Median:  {np.median(player_detections):.2f}")
    print(f"    Min:     {np.min(player_detections)}")
    print(f"    Max:     {np.max(player_detections)}")
    print(f"    Std Dev: {np.std(player_detections):.2f}")
    print(f"  Note: Trained model only detects balls, not players")
    
    print(f"\n  Tracked per frame:")
    print(f"    Average: {np.mean(player_tracked):.2f}")
    print(f"    Median:  {np.median(player_tracked):.2f}")
    print(f"    Min:     {np.min(player_tracked)}")
    print(f"    Max:     {np.max(player_tracked)}")
    
    # Confidence scores for players
    all_player_confs = []
    for m in frame_metrics:
        all_player_confs.extend(m['player_confidences'])
    
    if all_player_confs:
        print(f"\n  Player Confidence Scores:")
        print(f"    Average: {np.mean(all_player_confs):.3f}")
        print(f"    Median:  {np.median(all_player_confs):.3f}")
        print(f"    Min:     {np.min(all_player_confs):.3f}")
        print(f"    Max:     {np.max(all_player_confs):.3f}")
    
    # Balls
    ball_detections = [m['detections_balls'] for m in frame_metrics]
    ball_tracked = [m['tracked_balls'] for m in frame_metrics]
    
    print(f"\n⚽ BALLS (with SAHI):")
    print(f"  Detections per frame:")
    print(f"    Average: {np.mean(ball_detections):.2f}")
    print(f"    Median:  {np.median(ball_detections):.2f}")
    print(f"    Min:     {np.min(ball_detections)}")
    print(f"    Max:     {np.max(ball_detections)}")
    print(f"    Std Dev: {np.std(ball_detections):.2f}")
    
    print(f"\n  Tracked per frame:")
    print(f"    Average: {np.mean(ball_tracked):.2f}")
    print(f"    Median:  {np.median(ball_tracked):.2f}")
    print(f"    Min:     {np.min(ball_tracked)}")
    print(f"    Max:     {np.max(ball_tracked)}")
    
    # Confidence scores for balls
    all_ball_confs = []
    for m in frame_metrics:
        all_ball_confs.extend(m['ball_confidences'])
    
    if all_ball_confs:
        print(f"\n  Ball Confidence Scores:")
        print(f"    Average: {np.mean(all_ball_confs):.3f}")
        print(f"    Median:  {np.median(all_ball_confs):.3f}")
        print(f"    Min:     {np.min(all_ball_confs):.3f}")
        print(f"    Max:     {np.max(all_ball_confs):.3f}")
    else:
        print(f"\n  Ball Confidence Scores: No ball detections")
    
    # Overall
    total_detections = [m['detections_total'] for m in frame_metrics]
    total_tracked = [m['tracked_total'] for m in frame_metrics]
    
    print(f"\n📊 OVERALL:")
    print(f"  Total Detections per frame:")
    print(f"    Average: {np.mean(total_detections):.2f}")
    print(f"    Median:  {np.median(total_detections):.2f}")
    print(f"    Min:     {np.min(total_detections)}")
    print(f"    Max:     {np.max(total_detections)}")
    
    print(f"\n  Total Tracked per frame:")
    print(f"    Average: {np.mean(total_tracked):.2f}")
    print(f"    Median:  {np.median(total_tracked):.2f}")
    print(f"    Min:     {np.min(total_tracked)}")
    print(f"    Max:     {np.max(total_tracked)}")
    
    # Frame-by-frame breakdown
    frames_with_players = sum(1 for m in frame_metrics if m['detections_players'] > 0)
    frames_with_balls = sum(1 for m in frame_metrics if m['detections_balls'] > 0)
    frames_with_any = sum(1 for m in frame_metrics if m['detections_total'] > 0)
    
    print(f"\n📋 FRAME COVERAGE:")
    print(f"  Frames with player detections: {frames_with_players}/{num_frames} ({frames_with_players/num_frames*100:.1f}%)")
    print(f"  Frames with ball detections:   {frames_with_balls}/{num_frames} ({frames_with_balls/num_frames*100:.1f}%)")
    print(f"  Frames with any detections:    {frames_with_any}/{num_frames} ({frames_with_any/num_frames*100:.1f}%)")
    
    # Distribution
    print(f"\n📊 DISTRIBUTION:")
    player_dist = defaultdict(int)
    for count in player_detections:
        player_dist[count] += 1
    
    print(f"  Player detection counts:")
    for count in sorted(player_dist.keys()):
        print(f"    {count} players: {player_dist[count]} frames")
    
    ball_dist = defaultdict(int)
    for count in ball_detections:
        ball_dist[count] += 1
    
    print(f"  Ball detection counts:")
    for count in sorted(ball_dist.keys()):
        print(f"    {count} balls: {ball_dist[count]} frames")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    video_path = "data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
    analyze_frames(video_path, num_frames=100)
