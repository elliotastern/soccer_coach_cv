#!/usr/bin/env python3
"""
Test trained DETR model on video frames
"""
import os
# Disable CUDNN graph optimization
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.model import get_detr_model
import yaml


def load_trained_model(checkpoint_path: str, config_path: str = None):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load config
    if config_path is None:
        config_path = "configs/training_soccertrack_phase2.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = get_detr_model(config['model'], config.get('training', {}))
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found 'model_state_dict' in checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' in checkpoint")
    elif isinstance(checkpoint, dict) and any(k.startswith('detr_model.') or k.startswith('model.') for k in checkpoint.keys()):
        state_dict = checkpoint
        print("Checkpoint is state_dict directly")
    else:
        # Try to load as state_dict directly
        state_dict = checkpoint
        print("Attempting to load checkpoint as state_dict")
    
    # Load state dict with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
    
    model = model.cuda()
    model.eval()
    
    # Disable CUDNN to avoid compatibility issues
    torch.backends.cudnn.enabled = False
    print("✅ Model loaded successfully (CUDNN disabled)")
    return model, config


def preprocess_frame(frame):
    """Preprocess frame for DETR model wrapper"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Convert to tensor and normalize (same as training)
    from torchvision.transforms import ToTensor, Normalize
    img_tensor = ToTensor()(pil_image)
    img_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
    
    return img_tensor.cuda(), pil_image


def postprocess_detections(output, confidence_threshold=0.5):
    """Post-process model outputs"""
    detections = []
    
    # Model returns dict with boxes, scores, labels
    # Handle empty outputs
    if 'boxes' not in output or len(output['boxes']) == 0:
        return detections
    
    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()
    labels = output['labels'].cpu()
    
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            detections.append({
                'bbox': box.tolist(),
                'score': score.item(),
                'label': label.item(),
                'class_name': 'player'  # Only player class
            })
    
    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    for det in detections:
        x_min, y_min, x_max, y_max = [int(coord) for coord in det['bbox']]
        score = det['score']
        class_name = det['class_name']
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for players
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x_min, y_min - label_size[1] - 5), 
                     (x_min + label_size[0], y_min), color, -1)
        cv2.putText(frame, label, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def test_on_video(video_path: str, checkpoint_path: str, num_frames: int = 100, 
                  output_dir: str = "data/test_output", confidence_threshold: float = 0.5):
    """Test model on video frames"""
    print("=" * 70)
    print("TESTING TRAINED MODEL ON VIDEO")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Load model
    model, config = load_trained_model(checkpoint_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing: First {num_frames} frames")
    
    # Process frames
    frame_count = 0
    all_detections = []
    
    print(f"\nProcessing frames...")
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        img_tensor, pil_image = preprocess_frame(frame)
        
        # Get raw predictions directly from DETR processor (bypass wrapper filtering)
        from transformers import DetrImageProcessor
        import torchvision.transforms.functional as TF
        
        # Denormalize for processor
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device, dtype=img_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device, dtype=img_tensor.dtype).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        pil_img = TF.to_pil_image(img_denorm)
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(img_tensor.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            detr_outputs = model.detr_model(**inputs)
        
        # Post-process with threshold=0.0 to get ALL predictions
        target_sizes = torch.tensor([pil_img.size[::-1]], device=img_tensor.device, dtype=torch.float32)
        results = processor.post_process_object_detection(
            detr_outputs, target_sizes=target_sizes, threshold=0.0
        )[0]
        
        # Convert to our format (include ALL predictions, even low confidence)
        all_predictions = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            # Label 0 = background, 1 = player (for our single-class model)
            # Save all predictions, but mark background separately
            class_name = 'player' if label.item() == 1 else 'background'
            all_predictions.append({
                'bbox': box.cpu().tolist(),
                'score': score.item(),
                'label': label.item(),
                'class_name': class_name
            })
        
        # Filter for drawing
        filtered_detections = [d for d in all_predictions if d['score'] >= confidence_threshold]
        
        all_detections.append({
            'frame': frame_count,
            'detections': all_predictions  # Save all predictions, not just filtered ones
        })
        
        # Draw filtered detections
        frame_with_detections = draw_detections(frame.copy(), filtered_detections)
        
        # Save frame
        output_path = frames_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(output_path), frame_with_detections)
        
        if (frame_count + 1) % 10 == 0:
            print(f"  Processed {frame_count + 1}/{num_frames} frames ({len(all_predictions)} total, {len(filtered_detections)} above threshold)")
        
        frame_count += 1
    
    cap.release()
    
    # Save results summary
    total_detections = sum(len(d['detections']) for d in all_detections)
    avg_detections = total_detections / len(all_detections) if all_detections else 0
    
    summary = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'avg_detections_per_frame': avg_detections,
        'confidence_threshold': confidence_threshold
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save predictions for each frame
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"  - Predictions: {predictions_path}")
    
    print(f"\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_detections:.2f}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Summary: {summary_path}")
    print("=" * 70)
    
    return all_detections, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained DETR model on video")
    parser.add_argument(
        "--video",
        type=str,
        default="/workspace/soccer_coach_cv/data/raw/SoccerTrack_sub/videos/117093_panorama_1st_half-017.mp4",
        help="Path to video file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/soccer_coach_cv/models/checkpoints/latest_checkpoint.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    
    args = parser.parse_args()
    
    test_on_video(
        args.video,
        args.checkpoint,
        args.num_frames,
        args.output_dir,
        args.confidence
    )
