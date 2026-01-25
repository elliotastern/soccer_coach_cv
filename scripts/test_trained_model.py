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
        
        # Get original frame dimensions BEFORE any processing
        # OpenCV frame shape is (height, width, channels)
        original_height, original_width = frame.shape[:2]
        
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
        
        # Verify PIL image size matches original frame dimensions
        # PIL uses (width, height) format, OpenCV uses (height, width)
        pil_width, pil_height = pil_img.size
        pil_image_width, pil_image_height = pil_image.size
        
        # Verify both PIL images have correct size
        if pil_width != original_width or pil_height != original_height:
            print(f"⚠️  Warning: pil_img size ({pil_width}x{pil_height}) doesn't match original frame ({original_width}x{original_height})")
        if pil_image_width != original_width or pil_image_height != original_height:
            print(f"⚠️  Warning: pil_image size ({pil_image_width}x{pil_image_height}) doesn't match original frame ({original_width}x{original_height})")
        
        # Verify both PIL images have same size
        if pil_img.size != pil_image.size:
            print(f"⚠️  Warning: pil_img.size ({pil_img.size}) != pil_image.size ({pil_image.size})")
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(img_tensor.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            detr_outputs = model.detr_model(**inputs)
        
        # Post-process with threshold=0.0 to get ALL predictions
        # CRITICAL FIX: DETR post_process_object_detection expects target_sizes as [height, width]
        # This matches DETR's internal (C, H, W) tensor coordinate system
        # The processor resizes images to (C, H, W) format, so target_sizes must be [H, W]
        # This matches the working code in src/training/model.py line 297
        target_sizes = torch.tensor([pil_img.size[::-1]], device=img_tensor.device, dtype=torch.float32)  # [height, width]
        results = processor.post_process_object_detection(
            detr_outputs, target_sizes=target_sizes, threshold=0.0
        )[0]
        
        # BUG FIX: transformers post_process_object_detection incorrectly maps labels
        # The model predicts class 1 (player) but post-processing outputs class 0 (background)
        # We need to manually extract the correct labels from logits
        logits = detr_outputs.logits[0]  # [100, 2] for 100 queries, 2 classes
        probs = torch.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probs, dim=-1)  # [100] - actual predicted class
        predicted_scores = probs.max(dim=-1)[0]  # [100] - confidence of predicted class
        
        # Filter out background predictions and low confidence
        # Use confidence threshold from command line argument
        player_mask = (predicted_labels == 1) & (predicted_scores >= confidence_threshold)  # Only players above threshold
        
        if player_mask.sum() > 0:
            # Get player boxes and scores
            player_boxes = results['boxes'][player_mask]
            player_scores = predicted_scores[player_mask]
            
            # CRITICAL: Use ORIGINAL frame dimensions for filtering, not resized PIL image dimensions
            # The bounding boxes are already scaled to original frame size via target_sizes parameter
            # pil_img.size gives resized dimensions (e.g., 662x1333), but boxes are in original dimensions (e.g., 1906x3840)
            img_height = original_height
            img_width = original_width
            
            # Filter out very small boxes (likely false positives)
            box_areas = (player_boxes[:, 2] - player_boxes[:, 0]) * (player_boxes[:, 3] - player_boxes[:, 1])
            min_area = 2000  # Minimum box area in pixels
            max_area = 50000  # Maximum box area (filter huge boxes on lights/structures)
            size_mask = (box_areas >= min_area) & (box_areas <= max_area)
            
            # Filter out boxes in top 15% of image (likely stadium lights)
            # For panoramic videos, be less aggressive
            top_threshold = img_height * 0.10 if img_width / img_height > 3.0 else img_height * 0.15
            top_mask = player_boxes[:, 1] >= top_threshold
            
            # Filter out boxes in bottom 10% of image (likely sidelines/dugouts)
            # For very wide videos (panoramic), disable bottom filter as it's too aggressive
            if img_width / img_height > 3.0:
                bottom_mask = torch.ones(len(player_boxes), dtype=torch.bool, device=player_boxes.device)  # Don't filter bottom for panoramic
            else:
                bottom_mask = player_boxes[:, 3] <= (img_height * 0.90)
            
            # Combine all filters
            valid_mask = size_mask & top_mask & bottom_mask
            
            if valid_mask.sum() > 0:
                player_boxes = player_boxes[valid_mask]
                player_scores = player_scores[valid_mask]
                
                # Apply Non-Maximum Suppression to remove duplicate/overlapping detections
                # Use stricter IoU threshold to remove more duplicates
                from torchvision.ops import nms
                keep_indices = nms(player_boxes, player_scores, iou_threshold=0.4)
                
                player_boxes = player_boxes[keep_indices]
                player_scores = player_scores[keep_indices]
                
                # Sort by confidence and keep only top N detections
                # Limit to max 22 detections per frame (reasonable for soccer: 11 players per team)
                if len(player_scores) > 22:
                    sorted_indices = torch.argsort(player_scores, descending=True)
                    top_indices = sorted_indices[:22]
                    player_boxes = player_boxes[top_indices]
                    player_scores = player_scores[top_indices]
                
                # Convert to our format
                all_predictions = []
                for box, score in zip(player_boxes, player_scores):
                    all_predictions.append({
                        'bbox': box.cpu().tolist(),
                        'score': score.item(),
                        'label': 1,
                        'class_name': 'player'
                    })
            else:
                all_predictions = []
        else:
            all_predictions = []
        
        # Filter by confidence threshold for display
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
