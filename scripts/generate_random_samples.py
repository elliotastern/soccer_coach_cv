#!/usr/bin/env python3
"""
Generate 100 random frames with bounding boxes from video and predictions
Ensures correct frame-to-prediction matching
"""
import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    frame_height, frame_width = frame.shape[:2]
    
    for det in detections:
        bbox = det['bbox']
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        score = det['score']
        class_name = det.get('class_name', 'player')
        
        # Clamp coordinates to image bounds
        x_min = max(0, min(x_min, frame_width - 1))
        y_min = max(0, min(y_min, frame_height - 1))
        x_max = max(0, min(x_max, frame_width - 1))
        y_max = max(0, min(y_max, frame_height - 1))
        
        # Skip if box is invalid after clamping
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for players
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label above the bounding box
        label = f"{class_name}: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Calculate label position - place above box, or below if box is at top of image
        if y_min >= label_size[1] + 10:
            # Place label above the box
            label_y = y_min - 5
            label_x = x_min
        else:
            # Place label below the box if there's no room above
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
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def generate_random_samples(
    video_path: str,
    predictions_path: str,
    output_dir: str,
    num_samples: int = 100
):
    """Generate random sample frames with bounding boxes"""
    print("=" * 70)
    print("GENERATING RANDOM SAMPLES WITH BOUNDING BOXES")
    print("=" * 70)
    
    # Load predictions
    print(f"\nLoading predictions from: {predictions_path}")
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
    
    print(f"Loaded {len(predictions_data)} frame predictions")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "random_samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Open video
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Total frames: {total_video_frames}")
    print(f"  Resolution: {video_width}x{video_height}")
    
    # Get available frame indices from predictions
    available_frames = [pred['frame'] for pred in predictions_data]
    print(f"\nAvailable frames in predictions: {min(available_frames)} to {max(available_frames)}")
    
    # Select random frames (up to num_samples)
    num_available = len(available_frames)
    if num_available < num_samples:
        print(f"Warning: Only {num_available} frames available, using all of them")
        selected_frames = available_frames
    else:
        selected_frames = random.sample(available_frames, num_samples)
    
    selected_frames.sort()  # Sort for easier debugging
    print(f"Selected {len(selected_frames)} random frames")
    
    # Create mapping from frame number to predictions
    predictions_by_frame = {pred['frame']: pred['detections'] for pred in predictions_data}
    
    # Process each selected frame
    print(f"\nProcessing {len(selected_frames)} frames...")
    saved_count = 0
    
    for idx, frame_num in enumerate(selected_frames):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  ⚠️  Could not read frame {frame_num}")
            continue
        
        # Verify frame dimensions match
        frame_height, frame_width = frame.shape[:2]
        if frame_width != video_width or frame_height != video_height:
            print(f"  ⚠️  Frame {frame_num} size mismatch: {frame_width}x{frame_height} vs {video_width}x{video_height}")
        
        # Get predictions for this frame
        detections = predictions_by_frame.get(frame_num, [])
        
        # Draw bounding boxes
        frame_with_boxes = draw_detections(frame.copy(), detections)
        
        # Save frame
        output_path = samples_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_path), frame_with_boxes)
        saved_count += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{len(selected_frames)} frames ({len(detections)} detections on frame {frame_num})")
    
    cap.release()
    
    print(f"\n✅ Saved {saved_count} frames to {samples_dir}")
    print(f"   Total detections across all frames: {sum(len(predictions_by_frame.get(f, [])) for f in selected_frames)}")
    
    return samples_dir, selected_frames


def create_viewer_html(output_dir: str, selected_frames: List[int]):
    """Create HTML viewer for random samples"""
    output_dir = Path(output_dir)
    samples_dir = output_dir / "random_samples"
    
    # Count actual images
    image_files = sorted(samples_dir.glob("frame_*.jpg"))
    num_images = len(image_files)
    
    print(f"\nCreating viewer for {num_images} images...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Sample Predictions ({num_images} frames)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .sample-item {{
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sample-image {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
        }}
        .sample-info {{
            margin-top: 8px;
            font-size: 12px;
            color: #666;
            text-align: center;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Random Sample Predictions</h1>
        <p>{num_images} randomly selected frames with player detections</p>
        <div class="controls">
            <button onclick="toggleLabels()">Toggle Labels</button>
            <button onclick="shuffleGrid()">Shuffle</button>
        </div>
    </div>
    
    <div class="grid-container" id="grid-container">
"""
    
    # Add image entries
    for img_file in image_files:
        frame_num = int(img_file.stem.split('_')[1])
        relative_path = f"random_samples/{img_file.name}"
        html_content += f"""        <div class="sample-item">
            <img src="{relative_path}" alt="Frame {frame_num:06d}" class="sample-image">
            <div class="sample-info">Frame {frame_num}</div>
        </div>
"""
    
    html_content += """    </div>
    
    <script>
        let showLabels = true;
        
        function toggleLabels() {
            showLabels = !showLabels;
            const images = document.querySelectorAll('.sample-image');
            images.forEach(img => {
                img.style.opacity = showLabels ? '1' : '0.7';
            });
        }
        
        function shuffleGrid() {
            const container = document.getElementById('grid-container');
            const items = Array.from(container.children);
            
            // Fisher-Yates shuffle
            for (let i = items.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [items[i], items[j]] = [items[j], items[i]];
            }
            
            // Re-append in new order
            items.forEach(item => container.appendChild(item));
        }
    </script>
</body>
</html>
"""
    
    viewer_path = output_dir / "random_samples_viewer.html"
    with open(viewer_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created viewer: {viewer_path}")
    return viewer_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random sample frames with bounding boxes")
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/soccerchallenge2025/117092.mp4",
        help="Path to video file"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/test_output_training/predictions.json",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_output_training",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to generate"
    )
    
    args = parser.parse_args()
    
    # Generate random samples
    samples_dir, selected_frames = generate_random_samples(
        args.video,
        args.predictions,
        args.output_dir,
        args.num_samples
    )
    
    # Create viewer
    viewer_path = create_viewer_html(args.output_dir, selected_frames)
    
    print(f"\n✅ Complete!")
    print(f"   Samples: {samples_dir}")
    print(f"   Viewer: {viewer_path}")
