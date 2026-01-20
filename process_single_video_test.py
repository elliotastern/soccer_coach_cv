#!/usr/bin/env python3
"""
Test version that processes video and generates XML structure
Works without a trained model (uses mock detections for structure testing)
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from cvat_xml_generator import create_cvat_xml
from src.types import TrackedObject, Detection, Event, EventType, Location


def create_mock_detections(frame_id: int, num_players: int = 4) -> List[TrackedObject]:
    """Create mock detections for testing"""
    tracked_objects = []
    
    # Mock players
    for i in range(num_players):
        x = 100 + i * 200
        y = 100 + (i % 2) * 300
        w, h = 50, 100
        
        det = Detection(
            class_id=0,
            confidence=0.8 + np.random.random() * 0.2,
            bbox=(x, y, w, h),
            class_name="player"
        )
        
        tracked_objects.append(TrackedObject(
            object_id=i + 1,
            detection=det
        ))
    
    # Mock ball (every 10 frames)
    if frame_id % 10 == 0:
        x, y = 400, 300
        w, h = 20, 20
        det = Detection(
            class_id=1,
            confidence=0.9,
            bbox=(x, y, w, h),
            class_name="ball"
        )
        tracked_objects.append(TrackedObject(
            object_id=100,
            detection=det
        ))
    
    return tracked_objects


def create_mock_events() -> List[Event]:
    """Create mock events for testing"""
    events = []
    
    # Mock shot event (single frame)
    events.append(Event(
        id="shot_500",
        type=EventType.SHOT,
        start_frame=500,
        end_frame=500,
        start_location=Location(x=50.0, y=0.0),
        end_location=Location(x=50.0, y=0.0),
        involved_players=[1],
        confidence=0.92,
        timestamp_start=16.67,
        timestamp_end=16.67
    ))
    
    # Mock pass event (duration)
    events.append(Event(
        id="pass_100",
        type=EventType.PASS,
        start_frame=100,
        end_frame=150,
        start_location=Location(x=-20.0, y=10.0),
        end_location=Location(x=20.0, y=10.0),
        involved_players=[2, 3],
        confidence=0.85,
        timestamp_start=3.33,
        timestamp_end=5.0
    ))
    
    return events


def process_video_test(video_path: str):
    """Process video with mock detections to test XML generation"""
    logger.info(f"Processing video (TEST MODE): {video_path}")
    
    # Open video to get metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {frame_count} frames, {fps} fps, {width}x{height}")
    
    tracked_objects_by_frame = {}
    frame_id = 0
    
    # Process every 10th frame for speed
    while True:
        ret = cap.grab()  # Skip frames
        if not ret:
            break
        
        if frame_id % 10 == 0:  # Process every 10th frame
            tracked_objects = create_mock_detections(frame_id)
            if tracked_objects:
                tracked_objects_by_frame[frame_id] = tracked_objects
        
        frame_id += 1
        
        if frame_id % 100 == 0:
            logger.info(f"Processed {frame_id}/{frame_count} frames")
    
    cap.release()
    
    # Create mock events
    events = create_mock_events()
    
    logger.info(f"Generated {len(tracked_objects_by_frame)} frames with detections")
    logger.info(f"Generated {len(events)} events")
    
    # Generate XML
    logger.info("Generating CVAT XML...")
    xml_content = create_cvat_xml(
        video_path=video_path,
        tracked_objects_by_frame=tracked_objects_by_frame,
        events=events,
        video_metadata={"width": width, "height": height, "fps": fps, "frame_count": frame_count}
    )
    
    # Save XML
    xml_output = video_path.replace('.mp4', '_annotations.xml')
    with open(xml_output, 'w') as f:
        f.write(xml_content)
    
    logger.info(f"✅ Saved XML to: {xml_output}")
    logger.info(f"✅ XML file size: {os.path.getsize(xml_output)} bytes")
    logger.info("\nTo use with a real model:")
    logger.info("1. Train a model or get a checkpoint")
    logger.info("2. Update configs/auto_ingest.yaml with model path")
    logger.info("3. Run: python process_single_video.py <video_path>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_single_video_test.py <video_path>")
        print("This is a TEST MODE that generates XML structure without a model")
        sys.exit(1)
    
    video_path = sys.argv[1]
    process_video_test(video_path)
