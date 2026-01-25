#!/usr/bin/env python3
"""
Generate random sample frames with ground truth bounding boxes from CSV
"""
import cv2
import numpy as np
from pathlib import Path
import json
import random
import argparse
from collections import defaultdict


def parse_csv_annotations(csv_path: str):
    """Parse CSV file and extract ground truth bounding boxes"""
    with open(csv_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse headers
    team_row = lines[0].split(',')
    player_row = lines[1].split(',')
    attr_row = lines[2].split(',')
    
    # Build column mapping
    # Each player/ball has 4 attributes: bb_height, bb_left, bb_top, bb_width
    annotations = {}  # frame_id -> list of detections
    
    # Parse data rows (starting from row 4, index 4)
    for line in lines[4:]:
        parts = line.split(',')
        if not parts[0] or not parts[0].isdigit():
            continue
        
        frame_id = int(parts[0])
        detections = []
        
        # Parse each player/ball (4 columns per entity)
        i = 1
        while i < len(parts) - 3:
            team_id = team_row[i] if i < len(team_row) else None
            player_id = player_row[i] if i < len(player_row) else None
            attr = attr_row[i] if i < len(attr_row) else None
            
            if not team_id or not player_id or not attr:
                break
            
            # Get the 4 values for this entity
            if i + 3 >= len(parts):
                break
            
            try:
                bb_height = float(parts[i]) if parts[i] else 0
                bb_left = float(parts[i+1]) if parts[i+1] else 0
                bb_top = float(parts[i+2]) if parts[i+2] else 0
                bb_width = float(parts[i+3]) if parts[i+3] else 0
            except (ValueError, IndexError):
                i += 4
                continue
            
            # Skip if all zeros (no detection)
            if bb_left == 0 and bb_top == 0 and bb_width == 0 and bb_height == 0:
                i += 4
                continue
            
            # Convert from (bb_left, bb_top, bb_width, bb_height) to [x_min, y_min, x_max, y_max]
            x_min = bb_left
            y_min = bb_top
            x_max = bb_left + bb_width
            y_max = bb_top + bb_height
            
            # Determine class
            if team_id == 'BALL':
                class_name = 'ball'
                class_id = 2
            else:
                class_name = f'player_team{team_id}_p{player_id}'
                class_id = 0 if team_id == '0' else 1
            
            detections.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'class_name': class_name,
                'team_id': team_id,
                'player_id': player_id,
                'score': 1.0  # Ground truth has 100% confidence
            })
            
            i += 4
        
        if detections:
            annotations[frame_id] = detections
    
    return annotations


def draw_detections(frame, detections, draw_players=True, draw_ball=True):
    """Draw ground truth bounding boxes on frame"""
    frame_height, frame_width = frame.shape[:2]
    
    for det in detections:
        class_name = det['class_name']
        
        # Filter by type
        if 'ball' in class_name.lower() and not draw_ball:
            continue
        if 'player' in class_name.lower() and not draw_players:
            continue
        
        bbox = det['bbox']
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        
        # Clamp coordinates to image bounds
        x_min = max(0, min(x_min, frame_width - 1))
        y_min = max(0, min(y_min, frame_height - 1))
        x_max = max(0, min(x_max, frame_width - 1))
        y_max = max(0, min(y_max, frame_height - 1))
        
        # Skip if box is invalid after clamping
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Choose color based on class
        if 'ball' in class_name.lower():
            color = (0, 0, 255)  # Red for ball
        elif det.get('team_id') == '0':
            color = (255, 0, 0)  # Blue for team 0
        else:
            color = (0, 255, 0)  # Green for team 1
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = class_name
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Calculate label position - place above box, or below if box is at top of image
        if y_min >= label_size[1] + 10:
            label_y = y_min - 5
            label_x = x_min
        else:
            label_y = y_max + label_size[1] + 5
            label_x = x_min
        
        # Ensure label stays within image bounds
        label_y = max(label_size[1] + 5, min(label_y, frame_height - 5))
        label_x = max(0, min(label_x, frame_width - label_size[0] - 5))
        
        # Draw label background rectangle
        label_bg_y1 = label_y - label_size[1] - baseline
        label_bg_y2 = label_y + baseline
        label_bg_x1 = label_x
        label_bg_x2 = label_x + label_size[0]
        
        # Ensure background rectangle is within bounds
        label_bg_y1 = max(0, label_bg_y1)
        label_bg_y2 = min(frame_height, label_bg_y2)
        label_bg_x2 = min(frame_width, label_bg_x2)
        
        cv2.rectangle(frame, 
                     (label_bg_x1, label_bg_y1), 
                     (label_bg_x2, label_bg_y2), 
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def generate_ground_truth_samples(
    video_path: str,
    csv_path: str,
    output_dir: str,
    num_samples: int = 100
):
    """Generate random sample frames with ground truth bounding boxes"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "random_samples"
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("GENERATING RANDOM SAMPLES WITH GROUND TRUTH ANNOTATIONS")
    print("=" * 70)
    print()
    
    # Parse CSV annotations
    print(f"Loading ground truth from: {csv_path}")
    annotations = parse_csv_annotations(csv_path)
    print(f"Loaded {len(annotations)} frames with annotations")
    print()
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print()
    
    # Get available frames (frames that have annotations)
    available_frames = sorted([f for f in annotations.keys() if f <= total_frames])
    print(f"Available frames in annotations: {min(available_frames)} to {max(available_frames)}")
    
    # Select random frames
    if len(available_frames) < num_samples:
        selected_frames = available_frames
        print(f"Only {len(available_frames)} frames available, using all")
    else:
        selected_frames = sorted(random.sample(available_frames, num_samples))
    
    print(f"Selected {len(selected_frames)} random frames")
    print()
    
    # Process frames
    print(f"Processing {len(selected_frames)} frames...")
    total_detections = 0
    
    for idx, frame_num in enumerate(selected_frames):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-indexed
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Get annotations for this frame
        frame_detections = annotations.get(frame_num, [])
        total_detections += len(frame_detections)
        
        # Draw detections
        frame_with_boxes = draw_detections(frame.copy(), frame_detections)
        
        # Save frame
        output_path = samples_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_path), frame_with_boxes)
        
        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{len(selected_frames)} frames ({len(frame_detections)} detections on frame {frame_num})")
    
    cap.release()
    
    print()
    print(f"✅ Saved {len(selected_frames)} frames to {samples_dir}")
    print(f"   Total detections across all frames: {total_detections}")
    print()
    
    # Create HTML viewer
    print("Creating viewer for {} images...".format(len(selected_frames)))
    html_path = output_dir / "random_samples_viewer.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ground Truth Annotations ({len(selected_frames)} frames)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .controls button {{
            margin: 0 10px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .sample {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sample img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .sample-label {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Ground Truth Annotations</h1>
        <p>{len(selected_frames)} randomly selected frames with ground truth bounding boxes</p>
        <div class="controls">
            <button onclick="shuffleGrid()">Shuffle</button>
        </div>
    </div>
    <div class="grid" id="grid">
"""
    
    for frame_num in selected_frames:
        img_path = f"random_samples/frame_{frame_num:06d}.jpg"
        html_content += f"""        <div class="sample">
            <img src="{img_path}" alt="Frame {frame_num:06d}" loading="lazy">
            <div class="sample-label">Frame {frame_num}</div>
        </div>
"""
    
    html_content += """    </div>
    
    <script>
        function shuffleGrid() {
            const grid = document.getElementById('grid');
            const samples = Array.from(grid.children);
            for (let i = samples.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                grid.appendChild(samples[j]);
            }
        }
    </script>
</body>
</html>"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created viewer: {html_path}")
    print()
    print("✅ Complete!")
    print(f"   Samples: {samples_dir}")
    print(f"   Viewer: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random sample frames with ground truth annotations")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with ground truth")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of random samples")
    
    args = parser.parse_args()
    
    generate_ground_truth_samples(
        video_path=args.video,
        csv_path=args.csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
