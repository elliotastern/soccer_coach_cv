#!/usr/bin/env python3
"""
CVAT Watchdog Automation Script
Monitors incoming_videos/ for new .mp4 files, runs RF-DETR inference,
generates CVAT XML annotations with events, and uploads to CVAT.
"""
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import yaml
import cv2
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# CVAT SDK imports
try:
    from cvat_sdk import Client
    from cvat_sdk.api_client import models
except ImportError:
    print("ERROR: cvat-sdk not installed. Run: pip install cvat-sdk")
    sys.exit(1)

# Local imports
from src.perception.local_detector import LocalDetector
from src.perception.tracker import Tracker
from src.analysis.events import EventDetector
from src.analysis.mapping import PitchMapper
from src.types import TrackedObject, Event
from cvat_xml_generator import create_cvat_xml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoHandler(FileSystemEventHandler):
    """Watchdog handler for new video files"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.processed_files = set()  # Track processed files to avoid duplicates
        
        # Initialize detector
        model_path = config['model']['checkpoint_path']
        confidence_threshold = config['detection']['confidence_threshold']
        
        # Check if model exists, otherwise try Roboflow API
        if os.path.exists(model_path):
            logger.info(f"Initializing detector with model: {model_path}")
            self.detector = LocalDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device='cuda' if config['model'].get('use_cuda', True) else 'cpu'
            )
        else:
            logger.warning(f"Model checkpoint not found: {model_path}")
            logger.info("Attempting to use Roboflow API detector as fallback...")
            from src.perception.detector import Detector
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv('ROBOFLOW_API_KEY')
            if not api_key:
                raise ValueError("No model checkpoint found and ROBOFLOW_API_KEY not set. Please train a model or set ROBOFLOW_API_KEY.")
            model_id = config.get('roboflow', {}).get('model_id', '')
            if not model_id:
                raise ValueError("No model checkpoint found and roboflow.model_id not set in config.")
            self.detector = Detector(
                model_id=model_id,
                api_key=api_key,
                confidence_threshold=confidence_threshold
            )
        
        # Initialize tracker
        tracker_config = config['tracker']
        self.tracker = Tracker(
            track_thresh=tracker_config['track_thresh'],
            high_thresh=tracker_config['high_thresh'],
            track_buffer=tracker_config['track_buffer'],
            match_thresh=tracker_config['match_thresh'],
            frame_rate=tracker_config['frame_rate']
        )
        
        # Initialize pitch mapper
        mapping_config = config['mapping']
        self.pitch_mapper = PitchMapper(
            pitch_length=mapping_config['pitch_length'],
            pitch_width=mapping_config['pitch_width']
        )
        
        # Initialize event detector
        events_config = config['events']
        self.event_detector = EventDetector(
            pitch_mapper=self.pitch_mapper,
            pass_velocity_threshold=events_config['pass_velocity_threshold'],
            dribble_distance_threshold=events_config['dribble_distance_threshold'],
            shot_velocity_threshold=events_config['shot_velocity_threshold'],
            recovery_proximity=events_config['recovery_proximity']
        )
        
        # CVAT client
        self.cvat_client = None
        self._init_cvat_client()
    
    def _init_cvat_client(self):
        """Initialize CVAT client"""
        cvat_url = self.config['cvat']['url']
        cvat_user = self.config['cvat']['username']
        cvat_pass = self.config['cvat']['password']
        
        try:
            self.cvat_client = Client(url=cvat_url)
            self.cvat_client.login((cvat_user, cvat_pass))
            logger.info(f"Connected to CVAT at {cvat_url}")
        except Exception as e:
            logger.error(f"Failed to connect to CVAT: {e}")
            self.cvat_client = None
    
    def on_created(self, event):
        """Handle new file creation"""
        if event.is_directory:
            return
        
        if not event.src_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return
        
        # Avoid processing the same file twice
        if event.src_path in self.processed_files:
            return
        
        logger.info(f"New video detected: {event.src_path}")
        
        # Wait for file to be fully written
        if not self._wait_for_file_ready(event.src_path):
            logger.warning(f"File {event.src_path} not ready, skipping")
            return
        
        # Process the video
        try:
            self.process_video(event.src_path)
            self.processed_files.add(event.src_path)
        except Exception as e:
            logger.error(f"Error processing {event.src_path}: {e}", exc_info=True)
    
    def _wait_for_file_ready(self, file_path: str, max_wait: int = 30, check_interval: float = 0.5) -> bool:
        """
        Wait for file to be fully written by checking size stability
        
        Args:
            file_path: Path to file
            max_wait: Maximum seconds to wait
            check_interval: Seconds between checks
        
        Returns:
            True if file is ready, False if timeout
        """
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        while time.time() - start_time < max_wait:
            if not os.path.exists(file_path):
                time.sleep(check_interval)
                continue
            
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 3:  # Stable for 3 checks
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
            except OSError:
                pass
            
            time.sleep(check_interval)
        
        return False
    
    def process_video(self, video_path: str):
        """
        Process video: run inference, detect events, generate XML, upload to CVAT
        
        Args:
            video_path: Path to video file
        """
        video_name = Path(video_path).stem
        logger.info(f"Processing video: {video_name}")
        
        # Step 1: Run RF-DETR inference and tracking
        logger.info("Running RF-DETR inference...")
        tracked_objects_by_frame, all_events = self._run_inference_pipeline(video_path)
        
        logger.info(f"Detected {len(tracked_objects_by_frame)} frames with objects")
        logger.info(f"Detected {len(all_events)} events")
        
        # Step 2: Generate CVAT XML
        logger.info("Generating CVAT XML...")
        xml_output = video_path.replace('.mp4', '_annotations.xml').replace('.avi', '_annotations.xml')
        xml_output = xml_output.replace('.mov', '_annotations.xml').replace('.mkv', '_annotations.xml')
        
        xml_content = create_cvat_xml(
            video_path=video_path,
            tracked_objects_by_frame=tracked_objects_by_frame,
            events=all_events
        )
        
        with open(xml_output, 'w') as f:
            f.write(xml_content)
        
        logger.info(f"Saved XML to {xml_output}")
        
        # Step 3: Upload to CVAT
        if self.cvat_client:
            self._upload_to_cvat(video_path, xml_output, video_name)
        else:
            logger.warning("CVAT client not available, skipping upload")
        
        # Step 4: Move to processed folder
        processed_dir = Path(self.config['paths']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = processed_dir / Path(video_path).name
        shutil.move(video_path, dest_path)
        logger.info(f"Moved video to {dest_path}")
        
        # Also move XML if it exists
        if os.path.exists(xml_output):
            xml_dest = processed_dir / Path(xml_output).name
            shutil.move(xml_output, xml_dest)
            logger.info(f"Moved XML to {xml_dest}")
    
    def _run_inference_pipeline(self, video_path: str) -> Tuple[Dict[int, List[TrackedObject]], List[Event]]:
        """
        Run complete inference pipeline: detection, tracking, event detection
        
        Returns:
            Tuple of (tracked_objects_by_frame, all_events)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        tracked_objects_by_frame = {}
        all_events = []
        prev_frame_data = None
        frame_id = 0
        
        logger.info(f"Processing {frame_count} frames at {fps} fps")
        
        # Reset tracker for new video
        self.tracker.reset()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_id / fps if fps > 0 else frame_id * 0.033
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            if detections:
                # Track objects
                tracked_objects = self.tracker.update(detections, frame)
                
                if tracked_objects:
                    tracked_objects_by_frame[frame_id] = tracked_objects
                    
                    # Create FrameData for event detection
                    from src.types import FrameData, Player, Ball, Location
                    
                    players = []
                    ball = None
                    
                    for tracked_obj in tracked_objects:
                        det = tracked_obj.detection
                        location = self.pitch_mapper.bbox_center_to_pitch(tracked_obj.detection.bbox)
                        
                        if det.class_name == "player":
                            players.append(Player(
                                object_id=tracked_obj.object_id,
                                team_id=tracked_obj.team_id if tracked_obj.team_id else -1,
                                x_pitch=location.x,
                                y_pitch=location.y,
                                bbox=det.bbox,
                                frame_id=frame_id,
                                timestamp=timestamp
                            ))
                        elif det.class_name == "ball":
                            ball = Ball(
                                x_pitch=location.x,
                                y_pitch=location.y,
                                bbox=det.bbox,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                object_id=tracked_obj.object_id
                            )
                    
                    if players:  # Only process if we have players
                        frame_data = FrameData(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            players=players,
                            ball=ball,
                            detections=detections
                        )
                        
                        # Detect events
                        events = self.event_detector.detect_events(frame_data, prev_frame_data)
                        if events:
                            all_events.extend(events)
                        
                        prev_frame_data = frame_data
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id}/{frame_count} frames")
        
        cap.release()
        logger.info(f"Inference complete: {len(tracked_objects_by_frame)} frames, {len(all_events)} events")
        
        return tracked_objects_by_frame, all_events
    
    def _upload_to_cvat(self, video_path: str, xml_path: str, task_name: str):
        """
        Upload video and annotations to CVAT
        
        Args:
            video_path: Path to video file
            xml_path: Path to XML annotations
            task_name: Name for CVAT task
        """
        if not self.cvat_client:
            logger.error("CVAT client not initialized")
            return
        
        try:
            # Create task
            logger.info(f"Creating CVAT task: {task_name}")
            
            # Define labels
            labels = [
                models.LabelRequest(name="player"),
                models.LabelRequest(name="ball"),
                models.LabelRequest(name="pass"),
                models.LabelRequest(name="dribble"),
                models.LabelRequest(name="shot"),
                models.LabelRequest(name="recovery"),
            ]
            
            # Create task spec
            task_spec = models.TaskWriteRequest(
                name=task_name,
                labels=labels,
                segment_size=1  # Process entire video as one segment
            )
            
            # Create task
            task = self.cvat_client.tasks.create(spec=task_spec)
            logger.info(f"Created task {task.id}")
            
            # Upload video
            logger.info(f"Uploading video: {video_path}")
            # Try different CVAT SDK API methods based on version
            try:
                # CVAT SDK v2.x - create_data with file paths
                self.cvat_client.tasks.create_data(task.id, [video_path])
            except (AttributeError, TypeError):
                # Fallback: try with file objects
                with open(video_path, 'rb') as f:
                    self.cvat_client.tasks.create_data(task.id, [f])
            
            # Wait for upload to complete
            logger.info("Waiting for video upload to complete...")
            time.sleep(5)  # Give CVAT time to process
            
            # Upload annotations
            logger.info(f"Uploading annotations: {xml_path}")
            # Try different CVAT SDK API methods based on version
            try:
                # Method 1: import_annotations with filename
                self.cvat_client.tasks.import_annotations(
                    id=task.id,
                    format_name="CVAT 1.1",
                    filename=xml_path
                )
            except (AttributeError, TypeError):
                try:
                    # Method 2: import_annotations with file object
                    with open(xml_path, 'rb') as f:
                        self.cvat_client.tasks.import_annotations(
                            id=task.id,
                            format_name="CVAT 1.1",
                            file=f
                        )
                except (AttributeError, TypeError):
                    # Method 3: create_annotations
                    with open(xml_path, 'rb') as f:
                        self.cvat_client.tasks.create_annotations(
                            id=task.id,
                            format_name="CVAT 1.1",
                            file=f
                        )
            
            logger.info(f"Successfully uploaded task {task.id} to CVAT")
            
        except Exception as e:
            logger.error(f"Failed to upload to CVAT: {e}", exc_info=True)
            raise


def load_config(config_path: str = "configs/auto_ingest.yaml") -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if 'CVAT_URL' in os.environ:
        config['cvat']['url'] = os.environ['CVAT_URL']
    if 'CVAT_USER' in os.environ:
        config['cvat']['username'] = os.environ['CVAT_USER']
    if 'CVAT_PASS' in os.environ:
        config['cvat']['password'] = os.environ['CVAT_PASS']
    if 'MODEL_PATH' in os.environ:
        config['model']['checkpoint_path'] = os.environ['MODEL_PATH']
    
    return config


def get_default_config() -> dict:
    """Get default configuration"""
    return {
        'paths': {
            'watch_dir': './incoming_videos',
            'processed_dir': './processed'
        },
        'model': {
            'checkpoint_path': './models/checkpoints/latest_checkpoint.pth',
            'use_cuda': True
        },
        'detection': {
            'confidence_threshold': 0.5
        },
        'tracker': {
            'track_thresh': 0.5,
            'high_thresh': 0.6,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'frame_rate': 30
        },
        'mapping': {
            'pitch_length': 105.0,
            'pitch_width': 68.0
        },
        'events': {
            'pass_velocity_threshold': 5.0,
            'dribble_distance_threshold': 2.0,
            'shot_velocity_threshold': 15.0,
            'recovery_proximity': 1.0
        },
        'cvat': {
            'url': 'http://localhost:8080',
            'username': 'admin',
            'password': 'admin'
        }
    }


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Create watch directory if it doesn't exist
    watch_dir = Path(config['paths']['watch_dir'])
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting watchdog on directory: {watch_dir}")
    logger.info(f"Model: {config['model']['checkpoint_path']}")
    logger.info(f"CVAT: {config['cvat']['url']}")
    
    # Create event handler
    event_handler = VideoHandler(config)
    
    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    
    logger.info("Watchdog started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watchdog...")
        observer.stop()
    
    observer.join()
    logger.info("Watchdog stopped.")


if __name__ == "__main__":
    main()
