#!/usr/bin/env python3
"""
Visualize first 100 frames with 0.1 confidence threshold detections
Optimized for fast review of detection results
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
import os

sys.path.insert(0, str(Path(__file__).parent))

from rfdetr import RFDETRBase
from PIL import Image
from src.perception.tracker import Tracker
from src.types import Detection
import logging

logging.basicConfig(level=logging.WARNING)

def detect_with_rfdetr_custom(frame: np.ndarray, threshold: float) -> list:
    """Detect with custom threshold"""
    model = RFDETRBase()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    detections_raw = model.predict(pil_image, threshold=threshold)
    
    detections = []
    coco_person_id = 0
    coco_sports_ball_id = 37
    
    if hasattr(detections_raw, 'class_id'):
        num_detections = len(detections_raw.class_id)
        for i in range(num_detections):
            class_id_coco = int(detections_raw.class_id[i])
            confidence = float(detections_raw.confidence[i])
            bbox = detections_raw.xyxy[i]
            
            if class_id_coco == coco_person_id:
                class_id = 0
                class_name = 'player'
            elif class_id_coco == coco_sports_ball_id:
                class_id = 1
                class_name = 'ball'
            else:
                continue
            
            x_min, y_min, x_max, y_max = map(float, bbox)
            width = x_max - x_min
            height = y_max - y_min
            
            if width <= 0 or height <= 0:
                continue
            
            detections.append(Detection(
                class_id=class_id,
                confidence=confidence,
                bbox=(x_min, y_min, width, height),
                class_name=class_name
            ))
    
    return detections


def draw_detections(frame, detections, tracked_objects=None):
    """Draw detections on frame with color coding"""
    vis_frame = frame.copy()
    
    # Get frame dimensions for scaling
    frame_h, frame_w = frame.shape[:2]
    scale_factor = max(frame_w, frame_h) / 1000.0  # Scale based on frame size
    
    # Colors: Blue for players, Red for balls
    player_color = (255, 0, 0)  # BGR: Blue
    ball_color = (0, 0, 255)     # BGR: Red
    
    # Draw detection boxes
    for det in detections:
        x, y, w, h = det.bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Skip if box is invalid
        if w <= 0 or h <= 0:
            continue
        
        color = player_color if det.class_name == 'player' else ball_color
        thickness = max(3, int(4 * scale_factor))  # Thicker, scaled lines
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label with confidence - make it more visible
        label = f"{det.class_name[:1].upper()}:{det.confidence:.2f}"
        font_scale = max(0.8, 1.2 * scale_factor)
        font_thickness = max(2, int(2 * scale_factor))
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Position label above box
        label_y = max(y - 10, label_size[1] + 10)
        label_x = x
        
        # Draw label background
        cv2.rectangle(vis_frame, 
                     (label_x - 2, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 2, label_y + baseline + 2), 
                     color, -1)
        
        # Draw label text
        cv2.putText(vis_frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Draw tracked objects with IDs (if provided)
    if tracked_objects:
        for obj in tracked_objects:
            x, y, w, h = obj.detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            if w <= 0 or h <= 0:
                continue
            
            color = player_color if obj.detection.class_name == 'player' else ball_color
            thickness = max(4, int(5 * scale_factor))  # Thicker for tracked objects
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw track ID
            track_label = f"ID:{obj.object_id}"
            font_scale = max(0.9, 1.3 * scale_factor)
            font_thickness = max(2, int(3 * scale_factor))
            cv2.putText(vis_frame, track_label, (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
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
    threshold = 0.1
    num_frames = 100
    output_dir = Path("data/output/visualizations_0.1_confidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Visualizing First {num_frames} Frames (Confidence={threshold})")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS")
    print(f"Output directory: {output_dir}\n")
    
    tracker = Tracker(track_thresh=0.3, high_thresh=0.5, track_buffer=30, match_thresh=0.7, frame_rate=int(fps))
    
    frame_metrics = []
    visualized_frames = []
    frames_with_detections = []
    
    print("Processing and visualizing frames...")
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detect_with_rfdetr_custom(frame, threshold)
        players = [d for d in detections if d.class_name == 'player']
        balls = [d for d in detections if d.class_name == 'ball']
        
        # Track objects
        tracked_objects = tracker.update(detections, frame)
        tracked_players = [obj for obj in tracked_objects if obj.class_name == 'player']
        tracked_balls = [obj for obj in tracked_objects if obj.class_name == 'ball']
        
        # Draw detections
        vis_frame = draw_detections(frame, detections, tracked_objects)
        
        # Add frame info overlay
        info_text = f"Frame {frame_num} | P:{len(players)} B:{len(balls)} | Tracked: P:{len(tracked_players)} B:{len(tracked_balls)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save individual frame
        frame_path = output_dir / f"frame_{frame_num:03d}.jpg"
        cv2.imwrite(str(frame_path), vis_frame)
        
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
    html_path = output_dir / "summary.html"
    create_summary_html(html_path, frame_metrics, frames_with_detections, threshold)
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


def create_summary_html(html_path, frame_metrics, frames_with_detections, threshold):
    """Create HTML summary page for quick review"""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Detection Results - Confidence {threshold}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .stats {{ background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .frame-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
        .frame-item {{ background: white; padding: 10px; border-radius: 5px; }}
        .frame-item img {{ width: 100%; border: 2px solid #ddd; }}
        .frame-info {{ margin-top: 5px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Detection Results - Confidence Threshold: {threshold}</h1>
        <p>Total Frames Analyzed: {len(frame_metrics)}</p>
    </div>
    
    <div class="stats">
        <h2>Statistics</h2>
        <p>Frames with detections: {len(frames_with_detections)}/{len(frame_metrics)}</p>
    </div>
    
    <div class="frame-grid">
"""
    
    for frame_num, vis_frame, num_players, num_balls in frames_with_detections:
        frame_filename = f"frame_{frame_num:03d}.jpg"
        html_content += f"""
        <div class="frame-item">
            <img src="{frame_filename}" alt="Frame {frame_num}">
            <div class="frame-info">
                Frame {frame_num} | Players: {num_players} | Balls: {num_balls}
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
