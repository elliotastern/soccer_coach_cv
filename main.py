# The Orchestrator
import cv2
import yaml
import os
import pandas as pd
from pathlib import Path
from typing import Optional
import numpy as np
from dotenv import load_dotenv

from src.types import FrameData, Player, Ball, Location
from src.perception.camera import is_gameplay_view, detect_scene_cut
from src.perception.detector import Detector
from src.perception.tracker import Tracker
from src.perception.track_ball import create_ball_tracker_wrapper
from src.perception.team import assign_teams
from src.analysis.mapping import PitchMapper
from src.analysis.events import EventDetector
from src.analysis.event_manager import EventManager
from src.schema import frame_data_to_csv_row, get_csv_schema


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_frame(frame: np.ndarray, frame_id: int, timestamp: float,
                  detector: Detector, tracker: Tracker, pitch_mapper: PitchMapper,
                  prev_frame: Optional[np.ndarray], config: dict) -> Optional[FrameData]:
    """
    Process a single frame through the pipeline
    
    Returns:
        FrameData if frame is valid, None if skipped
    """
    # Gatekeeper: Check if gameplay view
    if not is_gameplay_view(frame):
        return None
    
    # Scene cut detection
    scene_cut = detect_scene_cut(frame, prev_frame)
    if scene_cut:
        tracker.reset()
    
    # Detection
    detections = detector.detect(frame)
    if not detections:
        return None
    
    # Tracking
    tracked_objects = tracker.update(detections, frame)
    if not tracked_objects:
        return None
    
    # Team assignment
    tracked_objects = assign_teams(
        tracked_objects,
        frame,
        n_clusters=config['team_assignment']['kmeans_clusters']
    )
    
    # Convert to FrameData
    players = []
    ball = None
    
    for obj in tracked_objects:
        if obj.detection.class_id == config['detection']['player_class_id']:
            # Player
            location = pitch_mapper.bbox_center_to_pitch(obj.detection.bbox)
            players.append(Player(
                object_id=obj.object_id,
                team_id=obj.team_id if obj.team_id is not None else -1,
                x_pitch=location.x,
                y_pitch=location.y,
                bbox=obj.detection.bbox,
                frame_id=frame_id,
                timestamp=timestamp
            ))
        elif obj.detection.class_id == config['detection']['ball_class_id']:
            # Ball
            location = pitch_mapper.bbox_center_to_pitch(obj.detection.bbox)
            ball = Ball(
                x_pitch=location.x,
                y_pitch=location.y,
                bbox=obj.detection.bbox,
                frame_id=frame_id,
                timestamp=timestamp,
                object_id=obj.object_id
            )
    
    if not players:
        return None
    
    return FrameData(
        frame_id=frame_id,
        timestamp=timestamp,
        players=players,
        ball=ball,
        detections=detections
    )


def process_video(video_path: str, config: dict, output_dir: str = "data/output"):
    """
    Process video through the complete pipeline
    
    Args:
        video_path: Path to input video file
        config: Configuration dictionary
        output_dir: Output directory for results
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment")
    
    # Initialize components
    detector = Detector(
        model_id=config['roboflow']['model_id'],
        api_key=api_key,
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    base_tracker = Tracker(
        track_thresh=config['tracker']['track_thresh'],
        high_thresh=config['tracker']['high_thresh'],
        track_buffer=config['tracker']['track_buffer'],
        match_thresh=config['tracker']['match_thresh'],
        frame_rate=config['tracker']['frame_rate']
    )
    
    # Wrap tracker with ball validation (parabolic fit check)
    # This filters out false positives like white socks that don't follow gravity curves
    tracker = create_ball_tracker_wrapper(
        base_tracker,
        min_track_length=5,  # Check after 5 detections
        fit_threshold=0.15  # 15% max normalized residual for valid trajectory
    )
    
    pitch_mapper = PitchMapper(
        pitch_length=config['mapping']['pitch_length'],
        pitch_width=config['mapping']['pitch_width']
    )
    
    event_detector = EventDetector(
        pitch_mapper=pitch_mapper,
        pass_velocity_threshold=config['events']['pass_velocity_threshold'],
        dribble_distance_threshold=config['events']['dribble_distance_threshold'],
        shot_velocity_threshold=config['events']['shot_velocity_threshold'],
        recovery_proximity=config['events']['recovery_proximity']
    )
    
    event_manager = EventManager(
        checkpoint_interval=config['checkpoint']['interval_frames'],
        output_dir=output_dir
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {frame_count} frames at {fps} fps")
    
    # Process frames
    prev_frame = None
    prev_frame_data = None
    frame_id = 0
    csv_rows = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_id / fps if fps > 0 else frame_id * 0.033
        
        # Process frame
        frame_data = process_frame(
            frame, frame_id, timestamp,
            detector, tracker, pitch_mapper,
            prev_frame, config
        )
        
        if frame_data:
            # Detect events
            events = event_detector.detect_events(frame_data, prev_frame_data)
            if events:
                event_manager.add_events(events)
            
            # Add to CSV rows
            csv_rows.extend(frame_data_to_csv_row(frame_data))
            
            prev_frame_data = frame_data
        
        prev_frame = frame.copy()
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{frame_count} frames")
    
    cap.release()
    
    # Save outputs
    match_id = Path(video_path).stem
    event_manager.save_final_output(
        match_id=match_id,
        csv_path=config['output']['csv_path'],
        json_path=config['output']['json_path']
    )
    
    # Save CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = config['output']['frame_data_path']
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
    
    print("Processing complete!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Soccer Analysis Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="data/output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Process video
    process_video(args.video, config, args.output)


if __name__ == "__main__":
    main()
