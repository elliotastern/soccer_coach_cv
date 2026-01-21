#!/usr/bin/env python3
"""
Pre-label a video using out-of-the-box RF-DETR model
Generates CVAT XML annotations for the annotation editor
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.perception.tracker import Tracker
from src.types import Detection, TrackedObject
from cvat_xml_generator import create_cvat_xml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_with_rfdetr(frame: np.ndarray) -> List[Detection]:
    """
    Use out-of-the-box RF-DETR model for detection
    RF-DETR is pre-trained on COCO, so we map:
    - COCO class 0 (person) -> player
    - COCO class 37 (sports ball) -> ball
    """
    try:
        from rfdetr import RFDETRBase
        from PIL import Image
        
        # Initialize RF-DETR Base (pre-trained on COCO)
        model = RFDETRBase()
        
        # Convert BGR to RGB and to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run inference
        detections_raw = model.predict(pil_image, threshold=0.3)
        
        # Convert to our Detection format
        detections = []
        coco_person_id = 0
        coco_sports_ball_id = 37
        
        # RF-DETR returns an object with attributes: class_id, confidence, xyxy
        # where xyxy is [x_min, y_min, x_max, y_max] for each detection
        if hasattr(detections_raw, 'class_id'):
            # Object with attributes (correct format)
            num_detections = len(detections_raw.class_id)
            for i in range(num_detections):
                class_id_coco = int(detections_raw.class_id[i])
                confidence = float(detections_raw.confidence[i])
                bbox = detections_raw.xyxy[i]  # [x_min, y_min, x_max, y_max]
                
                # Map COCO classes to our classes
                if class_id_coco == coco_person_id:
                    # Person -> player
                    class_id = 0
                    class_name = 'player'
                elif class_id_coco == coco_sports_ball_id:
                    # Sports ball -> ball
                    class_id = 1
                    class_name = 'ball'
                else:
                    # Skip other COCO classes
                    continue
                
                # Convert bbox from [x_min, y_min, x_max, y_max] to (x, y, width, height)
                x_min, y_min, x_max, y_max = map(float, bbox)
                width = x_max - x_min
                height = y_max - y_min
                
                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue
                
                detections.append(Detection(
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x_min, y_min, width, height),
                    class_name=class_name
                ))
        else:
            # Fallback: try list/dict format
            for det in detections_raw:
                if isinstance(det, dict):
                    class_id_coco = det.get('class_id', -1)
                    confidence = det.get('confidence', 0.0)
                    bbox = det.get('bbox', [])
                else:
                    continue
                
                # Map COCO classes to our classes
                if class_id_coco == coco_person_id:
                    class_id = 0
                    class_name = 'player'
                elif class_id_coco == coco_sports_ball_id:
                    class_id = 1
                    class_name = 'ball'
                else:
                    continue
                
                # Convert bbox format
                if isinstance(bbox, dict):
                    x_min = float(bbox.get('x_min', 0))
                    y_min = float(bbox.get('y_min', 0))
                    x_max = float(bbox.get('x_max', 0))
                    y_max = float(bbox.get('y_max', 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x_min, y_min, x_max, y_max = map(float, bbox[:4])
                else:
                    continue
                
                width = x_max - x_min
                height = y_max - y_min
                
                if width <= 0 or height <= 0:
                    continue
                
                detections.append(Detection(
                    class_id=class_id,
                    confidence=float(confidence),
                    bbox=(x_min, y_min, width, height),
                    class_name=class_name
                ))
        
        return detections
        
    except ImportError:
        logger.error("rfdetr library not found. Install with: pip install rfdetr")
        logger.error("Alternatively, you can use Roboflow SDK (requires API key)")
        raise
    except Exception as e:
        logger.error(f"Error in RF-DETR detection: {e}")
        raise


def process_video(video_path: str, output_xml_path: str, confidence_threshold: float = 0.3):
    """
    Process video and generate CVAT XML annotations
    
    Args:
        video_path: Path to input video
        output_xml_path: Path to output CVAT XML file
        confidence_threshold: Detection confidence threshold
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output XML: {output_xml_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize tracker
    tracker = Tracker(
        track_thresh=0.3,
        high_thresh=0.5,
        track_buffer=30,
        match_thresh=0.7,
        frame_rate=int(fps)
    )
    
    # Process frames
    frame_num = 0
    all_tracks = {}  # track_id -> list of boxes
    
    logger.info("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % 10 == 0:
            logger.info(f"  Frame {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
        
        # Detect objects
        detections = detect_with_rfdetr(frame)
        
        # Count detections by class
        players = [d for d in detections if d.class_name == 'player']
        balls = [d for d in detections if d.class_name == 'ball']
        
        if frame_num % 10 == 0:
            logger.info(f"    Detections: {len(players)} players, {len(balls)} balls")
        
        # Track objects
        tracked_objects = tracker.update(detections, frame)
        
        if frame_num % 10 == 0:
            tracked_players = [obj for obj in tracked_objects if obj.class_name == 'player']
            tracked_balls = [obj for obj in tracked_objects if obj.class_name == 'ball']
            logger.info(f"    Tracked: {len(tracked_players)} players, {len(tracked_balls)} balls")
        
        # Store tracks
        for obj in tracked_objects:
            track_id = obj.track_id
            if track_id not in all_tracks:
                all_tracks[track_id] = {
                    'label': obj.class_name,
                    'boxes': []
                }
            
            # Convert bbox to CVAT format (xtl, ytl, xbr, ybr)
            x, y, w, h = obj.bbox
            all_tracks[track_id]['boxes'].append({
                'frame': frame_num,
                'xtl': float(x),
                'ytl': float(y),
                'xbr': float(x + w),
                'ybr': float(y + h),
                'outside': 0,
                'occluded': 0,
                'keyframe': 1,
                'confidence': obj.confidence
            })
        
        frame_num += 1
    
    cap.release()
    logger.info(f"Processed {frame_num} frames, found {len(all_tracks)} tracks")
    
    # Generate CVAT XML
    logger.info("Generating CVAT XML...")
    xml_content = create_cvat_xml(
        video_path=video_path,
        width=width,
        height=height,
        tracks=all_tracks,
        events=[],  # No events for now
        fps=fps
    )
    
    # Write XML file
    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    logger.info(f"✅ CVAT XML saved to: {output_xml_path}")
    logger.info(f"   Total tracks: {len(all_tracks)}")
    logger.info(f"   Total boxes: {sum(len(t['boxes']) for t in all_tracks.values())}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python prelabel_video.py <video_path> [output_xml_path]")
        print("Example: python prelabel_video.py data/raw/real_data/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_xml_path = sys.argv[2]
    else:
        # Auto-generate output path
        video_path_obj = Path(video_path)
        output_xml_path = str(video_path_obj.parent / f"{video_path_obj.stem}_annotations.xml")
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        process_video(video_path, output_xml_path)
        logger.info("✅ Pre-labeling complete!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
