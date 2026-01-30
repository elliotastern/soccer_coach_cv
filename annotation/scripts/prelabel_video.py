#!/usr/bin/env python3
"""
Pre-label a video using out-of-the-box RF-DETR model
Generates CVAT XML annotations for the annotation editor
"""
import sys
from pathlib import Path

# Workspace root and soccer_coach_cv for src.* imports
_workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_workspace_root))
sys.path.insert(0, str(_workspace_root / "soccer_coach_cv"))

from src.perception.tracker import Tracker
from src.types import Detection, TrackedObject
from annotation.cvat_xml_generator import create_cvat_xml

import cv2
import numpy as np
from typing import List
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
        else:
            for det in detections_raw:
                if not isinstance(det, dict):
                    continue
                class_id_coco = det.get('class_id', -1)
                confidence = det.get('confidence', 0.0)
                bbox = det.get('bbox', [])
                if class_id_coco == coco_person_id:
                    class_id, class_name = 0, 'player'
                elif class_id_coco == coco_sports_ball_id:
                    class_id, class_name = 1, 'ball'
                else:
                    continue
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
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
        raise
    except Exception as e:
        logger.error(f"Error in RF-DETR detection: {e}")
        raise


def process_video(video_path: str, output_xml_path: str, confidence_threshold: float = 0.3):
    """
    Process video and generate CVAT XML annotations
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output XML: {output_xml_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    tracker = Tracker(
        track_thresh=0.3,
        high_thresh=0.5,
        track_buffer=30,
        match_thresh=0.7,
        frame_rate=int(fps)
    )

    tracked_objects_by_frame = {}
    frame_num = 0
    logger.info("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 10 == 0:
            logger.info(f"  Frame {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")

        detections = detect_with_rfdetr(frame)
        tracked_objects = tracker.update(detections, frame)

        if tracked_objects:
            tracked_objects_by_frame[frame_num] = tracked_objects

        if frame_num % 10 == 0 and tracked_objects:
            logger.info(f"    Tracked: {len([o for o in tracked_objects if o.detection.class_name == 'player'])} players, {len([o for o in tracked_objects if o.detection.class_name == 'ball'])} balls")

        frame_num += 1

    cap.release()
    logger.info(f"Processed {frame_num} frames, {len(tracked_objects_by_frame)} frames with tracks")

    logger.info("Generating CVAT XML...")
    xml_content = create_cvat_xml(
        video_path=video_path,
        tracked_objects_by_frame=tracked_objects_by_frame,
        events=[]
    )

    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    logger.info(f"CVAT XML saved to: {output_xml_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python prelabel_video.py <video_path> [output_xml_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_xml_path = sys.argv[2] if len(sys.argv) >= 3 else str(Path(video_path).parent / f"{Path(video_path).stem}_annotations.xml")

    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    try:
        process_video(video_path, output_xml_path)
        logger.info("Pre-labeling complete!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
