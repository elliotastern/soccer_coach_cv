#!/usr/bin/env python3
"""
Monitor training progress by generating predictions on 10 fixed frames after each epoch
Creates/updates an HTML viewer showing predictions across epochs
"""
import json
import cv2
import numpy as np
from pathlib import Path
import random
import argparse
from rfdetr import RFDETRMedium
from PIL import Image
import torch


def select_fixed_frames(annotations_path, images_dir, num_frames=10, seed=42):
    """Select fixed random frames that will be used across all epochs"""
    random.seed(seed)
    
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Select random frames
    selected_ids = random.sample(image_ids, min(num_frames, len(image_ids)))
    
    # Get image info for selected frames
    selected_frames = []
    for img_id in selected_ids:
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        selected_frames.append({
            'id': img_id,
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        })
    
    return selected_frames


def load_model_from_checkpoint(checkpoint_path):
    """Load RF-DETR model from checkpoint"""
    model = RFDETRMedium(pretrain_weights=checkpoint_path)
    return model


def predict_on_frame(model, image_path, threshold=0.3):
    """Run inference on a single frame"""
    image = Image.open(image_path).convert('RGB')
    detections = model.predict(image, threshold=threshold)
    
    # Convert to list format
    predictions = []
    for i in range(len(detections.xyxy)):
        predictions.append({
            'bbox': detections.xyxy[i].tolist(),
            'score': float(detections.confidence[i]),
            'class_id': int(detections.class_id[i])
        })
    
    return predictions


def draw_predictions(frame, predictions, color=(0, 255, 0)):
    """Draw bounding boxes on frame"""
    frame_height, frame_width = frame.shape[:2]
    
    for pred in predictions:
        bbox = pred['bbox']  # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        score = pred['score']
        
        # Clamp coordinates
        x_min = max(0, min(x_min, frame_width - 1))
        y_min = max(0, min(y_min, frame_height - 1))
        x_max = max(0, min(x_max, frame_width - 1))
        y_max = max(0, min(y_max, frame_height - 1))
        
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"person: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Position label above box
        if y_min >= label_size[1] + 10:
            label_y = y_min - 5
            label_x = x_min
        else:
            label_y = y_max + label_size[1] + 5
            label_x = x_min
        
        # Clamp label position
        label_y = max(label_size[1] + 5, min(label_y, frame_height - 5))
        label_x = max(0, min(label_x, frame_width - label_size[0] - 5))
        
        # Draw label background
        label_bg_y1 = label_y - label_size[1] - baseline
        label_bg_y2 = label_y + baseline
        label_bg_x1 = label_x
        label_bg_x2 = label_x + label_size[0]
        
        label_bg_y1 = max(0, label_bg_y1)
        label_bg_y2 = min(frame_height, label_bg_y2)
        label_bg_x2 = min(frame_width, label_bg_x2)
        
        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def generate_epoch_predictions(
    checkpoint_path,
    dataset_dir,
    selected_frames,
    output_dir,
    epoch_num,
    threshold=0.3
):
    """Generate predictions for selected frames using a checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path)
    
    images_dir = Path(dataset_dir) / "valid" / "images"
    epoch_output_dir = Path(output_dir) / f"epoch_{epoch_num}"
    epoch_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for frame_info in selected_frames:
        # Handle file_name that may or may not include 'images/' prefix
        file_name = frame_info['file_name']
        if file_name.startswith('images/'):
            file_name = file_name.replace('images/', '')
        
        image_path = images_dir / file_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load and process image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        # Run prediction
        predictions = predict_on_frame(model, image_path, threshold=threshold)
        
        # Draw predictions
        annotated_frame = draw_predictions(frame.copy(), predictions)
        
        # Save annotated image
        output_filename = f"frame_{frame_info['id']:04d}.jpg"
        output_path = epoch_output_dir / output_filename
        cv2.imwrite(str(output_path), annotated_frame)
        
        results.append({
            'frame_id': frame_info['id'],
            'file_name': frame_info['file_name'],
            'output_path': str(output_path.relative_to(Path(output_dir))),
            'num_detections': len(predictions),
            'predictions': predictions
        })
        
        print(f"  Frame {frame_info['id']}: {len(predictions)} detections")
    
    return results


def generate_html_viewer(output_dir, all_epoch_results):
    """Generate HTML viewer showing predictions across epochs"""
    html_path = Path(output_dir) / "epoch_predictions_viewer.html"
    
    epochs = sorted(all_epoch_results.keys())
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RF-DETR Training Progress - Epoch Predictions</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .epoch-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .epoch-header {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 15px;
        }}
        .frames-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .frame-item {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background-color: #fafafa;
        }}
        .frame-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .frame-info {{
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }}
        .detection-count {{
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <h1>RF-DETR Training Progress - Epoch Predictions</h1>
    <p>Showing predictions on 10 fixed validation frames across training epochs</p>
"""
    
    for epoch in epochs:
        epoch_results = all_epoch_results[epoch]
        html_content += f"""
    <div class="epoch-section">
        <div class="epoch-header">Epoch {epoch}</div>
        <div class="frames-grid">
"""
        
        for result in epoch_results:
            html_content += f"""
            <div class="frame-item">
                <img src="{result['output_path']}" alt="Frame {result['frame_id']}">
                <div class="frame-info">
                    Frame {result['frame_id']}: <span class="detection-count">{result['num_detections']} detections</span>
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML viewer generated: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate epoch predictions on fixed frames")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epoch", type=int, required=True, help="Current epoch number")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--frames-file", type=str, default=None, help="JSON file with selected frames")
    
    args = parser.parse_args()
    
    # Load or create selected frames
    if args.frames_file and Path(args.frames_file).exists():
        with open(args.frames_file) as f:
            selected_frames = json.load(f)
        print(f"Loaded {len(selected_frames)} selected frames from {args.frames_file}")
    else:
        annotations_path = Path(args.dataset_dir) / "valid" / "_annotations.coco.json"
        selected_frames = select_fixed_frames(annotations_path, None, num_frames=10, seed=42)
        
        # Save selected frames for future use
        frames_file = Path(args.output_dir) / "selected_frames.json"
        frames_file.parent.mkdir(parents=True, exist_ok=True)
        with open(frames_file, 'w') as f:
            json.dump(selected_frames, f, indent=2)
        print(f"Selected and saved {len(selected_frames)} frames to {frames_file}")
    
    # Generate predictions
    print(f"\nGenerating predictions for epoch {args.epoch}...")
    epoch_results = generate_epoch_predictions(
        args.checkpoint_path,
        args.dataset_dir,
        selected_frames,
        args.output_dir,
        args.epoch,
        args.threshold
    )
    
    # Load existing results
    results_file = Path(args.output_dir) / "epoch_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_epoch_results = json.load(f)
    else:
        all_epoch_results = {}
    
    # Update with current epoch
    all_epoch_results[str(args.epoch)] = epoch_results
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_epoch_results, f, indent=2)
    
    # Generate HTML viewer
    generate_html_viewer(args.output_dir, all_epoch_results)
    
    print(f"\n✅ Epoch {args.epoch} predictions complete!")


if __name__ == "__main__":
    main()
