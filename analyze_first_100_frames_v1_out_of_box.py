#!/usr/bin/env python3
"""
Analyze first 100 frames and report detection metrics
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from prelabel_video import detect_with_rfdetr
from src.perception.tracker import Tracker
from src.types import Detection
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs

def analyze_frames(video_path: str, num_frames: int = 100):
    """
    Analyze first N frames and report metrics
    """
    print(f"📊 Analyzing first {num_frames} frames of video...")
    print("=" * 60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS\n")
    
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
    
    print("Processing frames...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detect_with_rfdetr(frame)
        
        # Count by class
        players = [d for d in detections if d.class_name == 'player']
        balls = [d for d in detections if d.class_name == 'ball']
        
        # Track objects
        tracked_objects = tracker.update(detections, frame)
        tracked_players = [obj for obj in tracked_objects if obj.class_name == 'player']
        tracked_balls = [obj for obj in tracked_objects if obj.class_name == 'ball']
        
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
    print("📈 DETECTION METRICS (First 100 Frames)")
    print("=" * 60)
    
    # Players
    player_detections = [m['detections_players'] for m in frame_metrics]
    player_tracked = [m['tracked_players'] for m in frame_metrics]
    
    print(f"\n👥 PLAYERS:")
    print(f"  Detections per frame:")
    print(f"    Average: {np.mean(player_detections):.2f}")
    print(f"    Median:  {np.median(player_detections):.2f}")
    print(f"    Min:     {np.min(player_detections)}")
    print(f"    Max:     {np.max(player_detections)}")
    print(f"    Std Dev: {np.std(player_detections):.2f}")
    
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
    
    print(f"\n⚽ BALLS:")
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


if __name__ == "__main__":
    video_path = "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4"
    analyze_frames(video_path, num_frames=100)
