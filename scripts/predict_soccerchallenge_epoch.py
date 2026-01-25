#!/usr/bin/env python3
"""
Generate predictions on SoccerChallenge video using a specific epoch checkpoint
Creates visual output comparable to earlier epoch predictions
"""
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
# Add rf-detr to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rf-detr"))

from rfdetr import RFDETRMedium
from PIL import Image


def extract_frames_from_video(video_path: str, frame_indices: List[int], output_dir: Path) -> List[Dict]:
    """Extract specific frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Extracting {len(frame_indices)} frames")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    frame_data = []
    for idx, frame_idx in enumerate(sorted(frame_indices)):
        if frame_idx >= total_frames:
            print(f"Warning: Frame {frame_idx} exceeds total frames ({total_frames})")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame
        frame_filename = f"frame_{idx+1:02d}_idx_{frame_idx:06d}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        timestamp = frame_idx / fps if fps > 0 else 0
        
        frame_data.append({
            'frame_id': idx + 1,
            'original_frame_idx': frame_idx,
            'timestamp': timestamp,
            'image_path': str(frame_path.relative_to(output_dir)),
            'width': width,
            'height': height
        })
        
        print(f"  Extracted frame {idx+1}: {frame_idx} (t={timestamp:.2f}s)")
    
    cap.release()
    return frame_data


def load_model_from_checkpoint(checkpoint_path: str):
    """Load RF-DETR model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = RFDETRMedium(pretrain_weights=checkpoint_path)
    print("✅ Model loaded successfully")
    return model


def predict_on_image(model, image_path: Path, threshold: float = 0.3) -> List[Dict]:
    """Run inference on a single image"""
    pil_image = Image.open(image_path).convert('RGB')
    detections = model.predict(pil_image, threshold=threshold)
    
    predictions = []
    for i in range(len(detections.class_id)):
        predictions.append({
            'bbox': detections.xyxy[i].tolist(),  # [x_min, y_min, x_max, y_max]
            'score': float(detections.confidence[i]),
            'class_id': int(detections.class_id[i]),
            'class_name': 'person'  # RF-DETR detects person class
        })
    
    return predictions


def draw_predictions(frame: np.ndarray, predictions: List[Dict], color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes on frame"""
    annotated = frame.copy()
    
    for pred in predictions:
        bbox = pred['bbox']
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        score = pred['score']
        class_name = pred.get('class_name', 'person')
        
        # Clamp coordinates to image bounds
        h, w = annotated.shape[:2]
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, h - 1))
        
        # Skip if box is invalid
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Draw bounding box
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Position label above box, or below if at top
        if y_min >= label_size[1] + 10:
            label_y = y_min - 5
        else:
            label_y = y_max + label_size[1] + 5
        
        label_y = max(label_size[1] + 5, min(label_y, h - 5))
        label_x = max(0, min(x_min, w - label_size[0] - 5))
        
        # Draw label background
        label_bg_y1 = max(0, label_y - label_size[1] - baseline)
        label_bg_y2 = min(h, label_y + baseline)
        label_bg_x1 = label_x
        label_bg_x2 = min(w, label_x + label_size[0])
        
        cv2.rectangle(annotated, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
        cv2.putText(annotated, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated


def generate_predictions(
    model,
    frame_data: List[Dict],
    output_dir: Path,
    threshold: float = 0.3
) -> List[Dict]:
    """Generate predictions for all frames"""
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    print(f"\nGenerating predictions on {len(frame_data)} frames...")
    for frame_info in frame_data:
        frame_id = frame_info['frame_id']
        image_path = output_dir / frame_info['image_path']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Run prediction
        predictions = predict_on_image(model, image_path, threshold=threshold)
        
        # Load frame and draw predictions
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not load frame: {image_path}")
            continue
        
        annotated_frame = draw_predictions(frame.copy(), predictions)
        
        # Save annotated image
        prediction_filename = f"prediction_{frame_id:02d}.jpg"
        prediction_path = predictions_dir / prediction_filename
        cv2.imwrite(str(prediction_path), annotated_frame)
        
        result = {
            'frame_id': frame_id,
            'original_frame_idx': frame_info['original_frame_idx'],
            'timestamp': frame_info['timestamp'],
            'image_path': frame_info['image_path'],
            'prediction_image_path': str(prediction_path.relative_to(output_dir)),
            'num_detections': len(predictions),
            'predictions': predictions
        }
        
        all_results.append(result)
        print(f"  Frame {frame_id}: {len(predictions)} detections")
    
    return all_results


def generate_html_viewer(output_dir: Path, results: List[Dict], epoch: int, video_path: str):
    """Generate HTML viewer for predictions"""
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results) if results else 0
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RF-DETR SoccerChallenge 2025 Predictions - Epoch {epoch}</title>
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
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .frames-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .frame-item {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .frame-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .frame-info {{
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }}
        .detection-count {{
            font-weight: bold;
            color: #4CAF50;
        }}
        .stats {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>RF-DETR Predictions on SoccerChallenge 2025 Video</h1>
    
    <div class="summary">
        <h2>📊 Experiment Summary</h2>
        <p><strong>Model:</strong> RF-DETR Medium (Epoch {epoch})</p>
        <p><strong>Training Dataset:</strong> SoccerTrack wide_view</p>
        <p><strong>Test Video:</strong> SoccerChallenge 2025 (117092.mp4) - outside training set</p>
        <p><strong>Confidence Threshold:</strong> 0.3</p>
        <p><strong>Frames Tested:</strong> {len(results)} random frames</p>
        <p><strong>Total Detections:</strong> {total_detections}</p>
        <p><strong>Average per Frame:</strong> {avg_detections:.1f} detections</p>
        <p><strong>Video FPS:</strong> {fps:.1f}</p>
        <p><strong>Video Duration:</strong> {duration:.1f} seconds</p>
    </div>

    <h2>🎯 Frame-by-Frame Predictions</h2>
    <div class="frames-grid">
"""
    
    for result in results:
        minutes = int(result['timestamp'] // 60)
        seconds = int(result['timestamp'] % 60)
        time_str = f"{minutes:02d}:{seconds:02d} ({result['timestamp']:.1f}s)"
        
        html_content += f"""
        <div class="frame-item">
            <img src="{result['prediction_image_path']}" alt="Frame {result['frame_id']}">
            <div class="frame-info">
                <strong>Frame {result['frame_id']}</strong><br>
                Video Time: {time_str}<br>
                Frame Index: {result['original_frame_idx']}<br>
                <span class="detection-count">Detections: {result['num_detections']}</span>
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        // Add shuffle functionality
        function shuffleFrames() {
            const container = document.querySelector('.frames-grid');
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
    
    html_path = output_dir / "soccerchallenge_predictions_viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML viewer generated: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate predictions on SoccerChallenge video using epoch checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/rf_detr_soccertrack/checkpoint_best_total.pth",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/soccerchallenge2025/117092.mp4",
        help="Path to SoccerChallenge video"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temp_soccerchallenge_predictions_epoch99",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=99,
        help="Epoch number (for display purposes)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated frame indices to extract (default: use same as previous run)"
    )
    
    args = parser.parse_args()
    
    # Default frame indices (same as previous run based on predictions.json)
    default_frame_indices = [1673, 2004, 2036, 2246, 3860, 4526, 4537, 4839, 4930, 5073]
    
    if args.frame_indices:
        frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
    else:
        frame_indices = default_frame_indices
        print(f"Using default frame indices: {frame_indices}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RF-DETR SoccerChallenge Video Predictions")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Video: {args.video}")
    print(f"Epoch: {args.epoch}")
    print(f"Output: {output_dir}")
    print()
    
    # Extract frames
    print("Step 1: Extracting frames from video...")
    frame_data = extract_frames_from_video(args.video, frame_indices, output_dir)
    print(f"✅ Extracted {len(frame_data)} frames\n")
    
    # Load model
    print("Step 2: Loading model...")
    model = load_model_from_checkpoint(args.checkpoint)
    print()
    
    # Generate predictions
    print("Step 3: Generating predictions...")
    results = generate_predictions(model, frame_data, output_dir, threshold=args.threshold)
    print(f"✅ Generated predictions for {len(results)} frames\n")
    
    # Save predictions JSON
    predictions_json_path = output_dir / "predictions.json"
    with open(predictions_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved predictions to: {predictions_json_path}\n")
    
    # Generate HTML viewer
    print("Step 4: Generating HTML viewer...")
    html_path = generate_html_viewer(output_dir, results, args.epoch, args.video)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"Predictions: {predictions_json_path}")
    print(f"Viewer: {html_path}")
    print(f"Total detections: {sum(r['num_detections'] for r in results)}")
    print(f"Average per frame: {sum(r['num_detections'] for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
