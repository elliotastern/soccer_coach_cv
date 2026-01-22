#!/usr/bin/env python3
"""
Visualize first 100 frames with trained ball detection model (20 epochs, SoccerSynth_sub) WITH SAHI
Optimized for fast review of detection results with SAHI (Slicing Aided Hyper Inference)
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
import os

sys.path.insert(0, str(Path(__file__).parent))

from src.perception.tracker import Tracker
from src.types import Detection
import logging

logging.basicConfig(level=logging.WARNING)


def detect_with_trained_model_sahi(frame: np.ndarray, model=None, threshold: float = 0.3) -> list:
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
        confidence_threshold=threshold,
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
    return detections


def draw_detections(frame, detections, tracked_objects=None):
    """Draw detections on frame with color coding - using transparent overlays"""
    vis_frame = frame.copy()
    
    # Get frame dimensions for scaling
    frame_h, frame_w = frame.shape[:2]
    scale_factor = max(frame_w, frame_h) / 1000.0  # Scale based on frame size
    
    # Colors: Blue for players, Red for balls (BGR format)
    player_color = (255, 0, 0)  # BGR: Blue
    ball_color = (0, 0, 255)     # BGR: Red
    
    # Transparency settings
    box_alpha = 0.3  # 30% opacity for bounding box fill
    label_bg_alpha = 0.6  # 60% opacity for label background
    border_thickness = max(2, int(3 * scale_factor))  # Thinner border for transparency
    
    # Draw detection boxes with transparency
    for det in detections:
        x, y, w, h = det.bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Skip if box is invalid
        if w <= 0 or h <= 0:
            continue
        
        color = player_color if det.class_name == 'player' else ball_color
        
        # Create overlay for transparent bounding box
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # Filled rectangle
        cv2.addWeighted(overlay, box_alpha, vis_frame, 1 - box_alpha, 0, vis_frame)
        
        # Draw border (more visible)
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, border_thickness)
        
        # Draw label with confidence - semi-transparent background
        label = f"{det.class_name[:1].upper()}:{det.confidence:.2f}"
        font_scale = max(0.8, 1.2 * scale_factor)
        font_thickness = max(2, int(2 * scale_factor))
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Position label above box
        label_y = max(y - 10, label_size[1] + 10)
        label_x = x
        
        # Draw semi-transparent label background
        label_overlay = vis_frame.copy()
        cv2.rectangle(label_overlay, 
                     (label_x - 2, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 2, label_y + baseline + 2), 
                     color, -1)
        cv2.addWeighted(label_overlay, label_bg_alpha, vis_frame, 1 - label_bg_alpha, 0, vis_frame)
        
        # Draw label text (solid for readability)
        cv2.putText(vis_frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Draw tracked objects with IDs (if provided) - also transparent
    if tracked_objects:
        for obj in tracked_objects:
            x, y, w, h = obj.detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            if w <= 0 or h <= 0:
                continue
            
            color = player_color if obj.detection.class_name == 'player' else ball_color
            thickness = max(3, int(4 * scale_factor))  # Thicker for tracked objects
            
            # Create overlay for transparent bounding box
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, box_alpha, vis_frame, 1 - box_alpha, 0, vis_frame)
            
            # Draw border
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw track ID with semi-transparent background
            track_label = f"ID:{obj.object_id}"
            font_scale = max(0.9, 1.3 * scale_factor)
            font_thickness = max(2, int(3 * scale_factor))
            track_label_size, baseline = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            track_label_y = y - 15
            track_label_x = x
            
            # Semi-transparent background for track ID
            track_overlay = vis_frame.copy()
            cv2.rectangle(track_overlay,
                         (track_label_x - 2, track_label_y - track_label_size[1] - 5),
                         (track_label_x + track_label_size[0] + 2, track_label_y + baseline + 2),
                         color, -1)
            cv2.addWeighted(track_overlay, label_bg_alpha, vis_frame, 1 - label_bg_alpha, 0, vis_frame)
            
            # Draw track ID text (solid)
            cv2.putText(vis_frame, track_label, (track_label_x, track_label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return vis_frame


def create_frame_grid(frames_list, cols=10, max_frames=100):
    """Create a grid visualization of frames"""
    if not frames_list:
        return None
    
    num_frames = min(len(frames_list), max_frames)
    rows = (num_frames + cols - 1) // cols
    
    # Get frame dimensions
    h, w = frames_list[0].shape[:2]
    
    # Resize frames for grid (make them smaller)
    grid_h, grid_w = 180, 320  # Smaller size for grid
    resized_frames = [cv2.resize(f, (grid_w, grid_h)) for f in frames_list[:num_frames]]
    
    # Create grid
    grid = np.zeros((rows * grid_h, cols * grid_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized_frames):
        row = idx // cols
        col = idx % cols
        y_start = row * grid_h
        x_start = col * grid_w
        grid[y_start:y_start + grid_h, x_start:x_start + grid_w] = frame
        
        # Add frame number
        frame_num = idx
        cv2.putText(grid, f"#{frame_num}", (x_start + 5, y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return grid


def main():
    video_path = "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4"
    threshold = 0.3
    num_frames = 100
    output_dir = Path("data/output/visualizations_v2_20_epochs_sub_synth_plus_sahi")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Visualizing First {num_frames} Frames with TRAINED MODEL + SAHI (20 epochs, SoccerSynth_sub)")
    print("=" * 60)
    
    # Load trained model once
    print("Loading trained ball detection model...")
    checkpoint_path = "models/ball_detection/checkpoint.pth"
    from scripts.evaluate_ball_model import load_model
    trained_model = load_model(checkpoint_path)
    print("✅ Model loaded")
    print("✅ SAHI enabled: Slice size=1288, Overlap=20%\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS")
    print(f"Confidence threshold: {threshold}")
    print(f"Output directory: {output_dir}\n")
    
    tracker = Tracker(track_thresh=0.3, high_thresh=0.5, track_buffer=30, match_thresh=0.7, frame_rate=int(fps))
    
    frame_metrics = []
    visualized_frames = []
    frames_with_detections = []
    
    print("Processing and visualizing frames with SAHI...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects using trained model WITH SAHI
        detections = detect_with_trained_model_sahi(frame, model=trained_model, threshold=threshold)
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
        
        # Draw detections
        vis_frame = draw_detections(frame, detections, tracked_objects)
        
        # Add frame info overlay
        info_text = f"Frame {frame_num} | P:{len(players)} B:{len(balls)} | Tracked: P:{len(tracked_players)} B:{len(tracked_balls)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save individual frame with detections
        frame_path = output_dir / f"frame_{frame_num:03d}.jpg"
        cv2.imwrite(str(frame_path), vis_frame)
        
        # Save original frame without detections (for toggle)
        original_path = output_dir / f"frame_{frame_num:03d}_original.jpg"
        cv2.imwrite(str(original_path), frame)
        
        visualized_frames.append(vis_frame)
        
        # Store frames with detections for summary
        if len(detections) > 0:
            frames_with_detections.append((frame_num, vis_frame, len(players), len(balls)))
        
        # Store metrics
        frame_metrics.append({
            'frame': frame_num,
            'detections_players': len(players),
            'detections_balls': len(balls),
            'detections_total': len(detections),
            'tracked_players': len(tracked_players),
            'tracked_balls': len(tracked_balls),
        })
        
        if (frame_num + 1) % 20 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames...")
    
    cap.release()
    
    print("\nCreating visualizations...")
    
    # Create grid of all frames
    print("  Creating full grid (all frames)...")
    full_grid = create_frame_grid(visualized_frames, cols=10, max_frames=100)
    if full_grid is not None:
        grid_path = output_dir / "grid_all_frames.jpg"
        cv2.imwrite(str(grid_path), full_grid)
        print(f"    Saved: {grid_path}")
    
    # Create grid of frames with detections only
    if frames_with_detections:
        print("  Creating grid (frames with detections only)...")
        detection_frames = [f[1] for f in frames_with_detections]
        detection_grid = create_frame_grid(detection_frames, cols=10, max_frames=len(detection_frames))
        if detection_grid is not None:
            grid_path = output_dir / "grid_with_detections.jpg"
            cv2.imwrite(str(grid_path), detection_grid)
            print(f"    Saved: {grid_path} ({len(frames_with_detections)} frames)")
    
    # Create summary HTML
    print("  Creating summary HTML...")
    html_path = output_dir / "summary_v2_20_epochs_sub_synth_plus_sahi.html"
    create_summary_html(html_path, frame_metrics, frames_with_detections, threshold, output_dir)
    print(f"    Saved: {html_path}")
    
    # Print quick stats
    print("\n" + "=" * 60)
    print("📈 QUICK STATS")
    print("=" * 60)
    
    player_detections = [m['detections_players'] for m in frame_metrics]
    ball_detections = [m['detections_balls'] for m in frame_metrics]
    
    frames_with_players = sum(1 for m in frame_metrics if m['detections_players'] > 0)
    frames_with_balls = sum(1 for m in frame_metrics if m['detections_balls'] > 0)
    
    print(f"\nFrames with detections:")
    print(f"  Players: {frames_with_players}/{num_frames} ({frames_with_players/num_frames*100:.1f}%)")
    print(f"  Balls:   {frames_with_balls}/{num_frames} ({frames_with_balls/num_frames*100:.1f}%)")
    print(f"  Any:     {len(frames_with_detections)}/{num_frames} ({len(frames_with_detections)/num_frames*100:.1f}%)")
    
    print(f"\nAverage detections per frame:")
    print(f"  Players: {np.mean(player_detections):.2f} (max: {np.max(player_detections)})")
    print(f"  Balls:   {np.mean(ball_detections):.2f} (max: {np.max(ball_detections)})")
    
    print(f"\n✅ Visualization complete!")
    print(f"   Individual frames: {output_dir}/frame_*.jpg")
    print(f"   Grid visualizations: {output_dir}/grid_*.jpg")
    print(f"   Summary HTML: {html_path}")
    print("=" * 60)


def create_summary_html(html_path, frame_metrics, frames_with_detections, threshold, output_dir):
    """Create HTML summary page for quick review"""
    # Calculate statistics
    player_detections = [m['detections_players'] for m in frame_metrics]
    ball_detections = [m['detections_balls'] for m in frame_metrics]
    total_detections = [m['detections_total'] for m in frame_metrics]
    
    frames_with_players = sum(1 for m in frame_metrics if m['detections_players'] > 0)
    frames_with_balls = sum(1 for m in frame_metrics if m['detections_balls'] > 0)
    
    avg_players = np.mean(player_detections) if player_detections else 0
    avg_balls = np.mean(ball_detections) if ball_detections else 0
    max_players = np.max(player_detections) if player_detections else 0
    max_balls = np.max(ball_detections) if ball_detections else 0
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trained Model Detection Results - 20 Epochs SoccerSynth_sub + SAHI</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .instructions {{ background: #4CAF50; color: white; padding: 15px; margin: 20px 0; border-radius: 5px; font-weight: bold; text-align: center; }}
        .stats {{ background: white; padding: 15px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .stat-item {{ background: #f9f9f9; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .frame-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
        .frame-item {{ background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
        .frame-item img {{ width: 100%; border: 2px solid #ddd; border-radius: 3px; }}
        .frame-item img.original {{ display: none; }}
        .frame-item.show-original img.with-boxes {{ display: none; }}
        .frame-item.show-original img.original {{ display: block; }}
        .frame-info {{ margin-top: 5px; font-size: 12px; color: #666; }}
        .note {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }}
        .sahi-badge {{ background: #2196F3; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; display: inline-block; margin-left: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Detection Results - Trained Model (20 Epochs, SoccerSynth_sub) <span class="sahi-badge">+ SAHI</span></h1>
        <p>Model: Ball-only detection trained on SoccerSynth_sub dataset</p>
        <p>SAHI: Slicing Aided Hyper Inference enabled for better tiny object detection</p>
        <p>Confidence Threshold: {threshold} | Total Frames Analyzed: {len(frame_metrics)}</p>
    </div>
    
    <div class="instructions">
        ⌨️ Press <strong>H</strong> to toggle detection boxes on/off
    </div>
    
    <div class="stats">
        <h2>Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Frames with Detections</div>
                <div class="stat-value">{len(frames_with_detections)}/{len(frame_metrics)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Frames with Players</div>
                <div class="stat-value">{frames_with_players}/{len(frame_metrics)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Frames with Balls</div>
                <div class="stat-value">{frames_with_balls}/{len(frame_metrics)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Players/Frame</div>
                <div class="stat-value">{avg_players:.2f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Balls/Frame</div>
                <div class="stat-value">{avg_balls:.2f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Max Balls/Frame</div>
                <div class="stat-value">{max_balls}</div>
            </div>
        </div>
    </div>
    
    <div class="note">
        <strong>Note:</strong> This model was trained only on ball detection (single class). 
        It does not detect players. Model trained for 20 epochs on SoccerSynth_sub synthetic dataset.
        <br><strong>SAHI Enabled:</strong> Slicing Aided Hyper Inference improves detection of tiny objects by processing image tiles.
    </div>
    
    <div class="stats">
        <h2>Frames with Detections ({len(frames_with_detections)} frames)</h2>
    </div>
    
    <div class="frame-grid">
"""
    
    for frame_num, vis_frame, num_players, num_balls in frames_with_detections:
        frame_filename = f"frame_{frame_num:03d}.jpg"
        original_filename = f"frame_{frame_num:03d}_original.jpg"
        html_content += f"""
        <div class="frame-item" data-frame="{frame_num}">
            <img src="{frame_filename}" alt="Frame {frame_num}" class="with-boxes">
            <img src="{original_filename}" alt="Frame {frame_num} (original)" class="original">
            <div class="frame-info">
                Frame {frame_num} | Players: {num_players} | Balls: {num_balls}
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        let boxesVisible = true;
        
        document.addEventListener('keydown', function(event) {
            // Check if 'h' or 'H' is pressed
            if (event.key === 'h' || event.key === 'H') {
                boxesVisible = !boxesVisible;
                
                // Toggle all frame items
                const frameItems = document.querySelectorAll('.frame-item');
                frameItems.forEach(item => {
                    if (boxesVisible) {
                        item.classList.remove('show-original');
                    } else {
                        item.classList.add('show-original');
                    }
                });
                
                // Update instructions
                const instructions = document.querySelector('.instructions');
                if (boxesVisible) {
                    instructions.textContent = '⌨️ Press H to toggle detection boxes on/off (Boxes: ON)';
                    instructions.style.background = '#4CAF50';
                } else {
                    instructions.textContent = '⌨️ Press H to toggle detection boxes on/off (Boxes: OFF)';
                    instructions.style.background = '#ff9800';
                }
            }
        });
    </script>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
