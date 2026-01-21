#!/usr/bin/env python3
"""
Run inference on video frames 1+ using fine-tuned model
Skips frame 0 (manually labeled) and processes remaining frames
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.perception.local_detector import LocalDetector
from src.perception.tracker import Tracker
from src.types import TrackedObject, Detection


def infer_remaining_frames(
    video_path: str,
    model_path: str,
    confidence_threshold: float = 0.5,
    start_frame: int = 1,
    device: str = None
) -> Dict[int, List[TrackedObject]]:
    """
    Run inference on frames starting from start_frame
    
    Args:
        video_path: Path to video file
        model_path: Path to fine-tuned model checkpoint
        confidence_threshold: Minimum confidence for detections
        start_frame: First frame to process (default: 1, skipping frame 0)
        device: Device to use ('cuda' or 'cpu'), auto-detect if None
    
    Returns:
        Dictionary mapping frame_id -> List[TrackedObject]
    """
    # Initialize detector
    print(f"Loading model from: {model_path}")
    detector = LocalDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )
    
    # Initialize tracker
    tracker = Tracker(
        track_thresh=0.5,
        high_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30.0
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_count} frames at {fps} fps")
    print(f"Processing frames {start_frame} to {frame_count - 1}")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    tracked_objects_by_frame = {}
    frame_id = start_frame
    
    print("\nRunning inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame)
        
        if detections:
            # Track objects
            tracked_objects = tracker.update(detections, frame)
            
            if tracked_objects:
                tracked_objects_by_frame[frame_id] = tracked_objects
        
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{frame_count} frames ({len(tracked_objects_by_frame)} frames with detections)")
    
    cap.release()
    
    print(f"\nInference complete: {len(tracked_objects_by_frame)} frames with detections")
    
    return tracked_objects_by_frame


def main():
    parser = argparse.ArgumentParser(description="Run inference on remaining frames")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First frame to process (default: 1, skipping frame 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'), auto-detect if None"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save tracked objects to file (for debugging)"
    )
    
    args = parser.parse_args()
    
    tracked_objects_by_frame = infer_remaining_frames(
        video_path=args.video,
        model_path=args.model,
        confidence_threshold=args.confidence,
        start_frame=args.start_frame,
        device=args.device
    )
    
    # Save to file if requested
    if args.output:
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump(tracked_objects_by_frame, f)
        print(f"\nSaved tracked objects to: {args.output}")


if __name__ == "__main__":
    main()
