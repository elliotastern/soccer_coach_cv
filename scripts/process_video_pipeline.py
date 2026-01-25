#!/usr/bin/env python3
"""
Integrated video processing pipeline combining:
- R-001: RF-DETR player detection
- R-002: Team ID assignment (HSV clustering)
- R-003: Pixel-to-pitch coordinate mapping (homography)

This script demonstrates the "Watchdog" architecture concept where
GPU inference (detection) is decoupled from CPU logic (team ID, mapping).
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr import RFDETRMedium
from src.types import Detection, TrackedObject, Player, Ball, FrameData
from src.logic.team_id import TeamClusterer, TeamAssignment
from src.perception.team import extract_player_crops, assign_teams_to_tracked_objects
from src.analysis.homography import HomographyEstimator
from src.analysis.mapping import PitchMapper
# from src.perception.tracker import ByteTracker  # Not needed for basic pipeline


class VideoProcessingPipeline:
    """
    Complete video processing pipeline integrating detection, team ID, and mapping.
    """
    
    def __init__(self, 
                 model_path: str,
                 homography_path: Optional[str] = None,
                 pitch_length: float = 105.0,
                 pitch_width: float = 68.0,
                 golden_batch_size: int = 500,
                 confidence_threshold: float = 0.3):
        """
        Initialize pipeline
        
        Args:
            model_path: Path to RF-DETR checkpoint
            homography_path: Optional path to pre-computed homography JSON
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            golden_batch_size: Number of frames for Golden Batch initialization
            confidence_threshold: Detection confidence threshold
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.golden_batch_size = golden_batch_size
        self.confidence_threshold = confidence_threshold
        
        # Initialize RF-DETR model
        print("Loading RF-DETR model...")
        self.detector = RFDETRMedium(pretrain_weights=model_path)
        self.detector.eval()
        print("✅ Model loaded")
        
        # Initialize team clusterer
        self.team_clusterer = TeamClusterer(pitch_length, pitch_width)
        self.team_clusterer_trained = False
        
        # Initialize homography estimator
        self.homography_estimator = HomographyEstimator(pitch_length, pitch_width)
        
        # Load or initialize homography
        if homography_path and Path(homography_path).exists():
            print(f"Loading homography from: {homography_path}")
            self._load_homography(homography_path)
        else:
            print("⚠️  No homography provided. Will need manual calibration.")
            print("   Run: python scripts/calibrate_homography.py <video> --output homography.json")
        
        # Initialize pitch mapper
        self.pitch_mapper = PitchMapper(
            pitch_length, 
            pitch_width,
            homography_estimator=self.homography_estimator
        )
        
        # Frame tracking
        self.frame_count = 0
        self.golden_batch_crops = []
        self.golden_batch_positions = []
        
        # Output data
        self.frame_data_list = []
    
    def _load_homography(self, homography_path: str):
        """Load homography from JSON file"""
        with open(homography_path, 'r') as f:
            data = json.load(f)
        
        H = np.array(data['homography'], dtype=np.float32)
        self.homography_estimator.set_homography(H)
        self.pitch_mapper.set_homography(H)
        print("✅ Homography loaded")
    
    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players using RF-DETR
        
        Args:
            frame: BGR image frame
        
        Returns:
            List of Detection objects
        """
        # Convert BGR to RGB for RF-DETR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections_raw = self.detector.predict(frame_rgb, threshold=self.confidence_threshold)
        
        # Convert to Detection objects
        detections = []
        if hasattr(detections_raw, 'class_id'):
            num_detections = len(detections_raw.class_id)
            for i in range(num_detections):
                class_id = int(detections_raw.class_id[i])
                confidence = float(detections_raw.confidence[i])
                bbox_xyxy = detections_raw.xyxy[i]  # [x_min, y_min, x_max, y_max]
                
                # Filter for players only (class_id 1 = person in COCO)
                if class_id == 1:  # COCO person class
                    x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    detections.append(Detection(
                        class_id=0,  # 0 = player in our system
                        confidence=confidence,
                        bbox=(x_min, y_min, width, height),
                        class_name="player"
                    ))
        
        return detections
    
    def initialize_team_clustering(self, frame: np.ndarray, detections: List[Detection]):
        """
        Initialize team clustering using Golden Batch strategy
        
        Args:
            frame: Current frame
            detections: Player detections
        """
        # Extract high-confidence player crops
        high_conf_detections = [d for d in detections if d.confidence > 0.8]
        
        if len(high_conf_detections) == 0:
            return
        
        # Extract crops
        crops = extract_player_crops(frame, high_conf_detections)
        
        if len(crops) == 0:
            return
        
        # Get pitch positions for crops (if homography available)
        positions = None
        if self.homography_estimator.get_homography() is not None:
            positions = []
            for det in high_conf_detections:
                x, y, w, h = det.bbox
                center_x = x + w / 2
                center_y = y + h / 2
                pitch_pos = self.pitch_mapper.pixel_to_pitch(center_x, center_y)
                positions.append((pitch_pos.x, pitch_pos.y))
        
        # Accumulate crops
        crop_images = [crop for crop, _ in crops]
        self.golden_batch_crops.extend(crop_images)
        if positions:
            self.golden_batch_positions.extend(positions)
        
        # Train if we have enough samples
        if len(self.golden_batch_crops) >= 20 and not self.team_clusterer_trained:
            print(f"\n🎯 Training team clusterer with {len(self.golden_batch_crops)} crops...")
            success = self.team_clusterer.fit(
                self.golden_batch_crops,
                self.golden_batch_positions if self.golden_batch_positions else None,
                confidence_threshold=0.8,
                min_crops=20
            )
            
            if success:
                self.team_clusterer_trained = True
                print("✅ Team clusterer trained")
                
                # Get team colors for visualization
                team_colors = self.team_clusterer.get_team_colors()
                if team_colors:
                    print(f"   Team 0 HSV: {team_colors[0]}")
                    print(f"   Team 1 HSV: {team_colors[1]}")
            else:
                print("⚠️  Team clusterer training failed")
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[FrameData]:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: BGR image frame
            frame_id: Frame number
            timestamp: Timestamp in seconds
        
        Returns:
            FrameData object or None
        """
        # Step 1: Detect players
        detections = self.detect_players(frame)
        
        if len(detections) == 0:
            return None
        
        # Step 2: Golden Batch - Initialize team clustering
        if self.frame_count < self.golden_batch_size and not self.team_clusterer_trained:
            self.initialize_team_clustering(frame, detections)
        
        # Step 3: Map to pitch coordinates
        players = []
        ball = None
        
        player_bboxes = []
        for det in detections:
            if det.class_name == 'player':
                x, y, w, h = det.bbox
                player_bboxes.append((x, y, w, h))
        
        # Update homography with optical flow (if initialized)
        if self.homography_estimator.get_homography() is not None:
            self.homography_estimator.track_with_optical_flow(frame, player_bboxes)
            
            # Check for drift
            if self.homography_estimator.detect_drift(frame):
                print(f"⚠️  Drift detected at frame {frame_id}. Re-alignment needed.")
                # In production, would trigger re-calibration here
        
        # Step 4: Assign team IDs and map to pitch
        for det in detections:
            if det.class_name == 'player':
                # Extract crop
                crop = extract_player_crops(frame, [det])[0][0] if extract_player_crops(frame, [det]) else None
                
                # Get pitch position
                x, y, w, h = det.bbox
                center_x = x + w / 2
                center_y = y + h / 2
                pitch_pos = self.pitch_mapper.pixel_to_pitch(center_x, center_y)
                
                # Assign team ID
                team_id = None
                role = "PLAYER"
                
                if self.team_clusterer_trained and crop is not None:
                    assignment = self.team_clusterer.predict(
                        crop,
                        (pitch_pos.x, pitch_pos.y)
                    )
                    team_id = assignment.team_id
                    role = assignment.role
                
                # Create player object
                players.append(Player(
                    object_id=len(players),  # Simple ID assignment
                    team_id=team_id if team_id is not None else -1,
                    x_pitch=pitch_pos.x,
                    y_pitch=pitch_pos.y,
                    bbox=det.bbox,
                    frame_id=frame_id,
                    timestamp=timestamp
                ))
        
        # Step 5: Create FrameData
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            players=players,
            ball=ball,
            detections=detections
        )
        
        self.frame_count += 1
        return frame_data
    
    def process_video(self, video_path: str, output_dir: str, max_frames: Optional[int] = None):
        """
        Process entire video
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            max_frames: Maximum frames to process (None for all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"\n📹 Processing video: {total_frames} frames at {fps:.2f} fps")
        print("="*70)
        
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_id >= max_frames:
                break
            
            timestamp = frame_id / fps if fps > 0 else frame_id * 0.033
            
            # Process frame
            frame_data = self.process_frame(frame, frame_id, timestamp)
            
            if frame_data:
                self.frame_data_list.append(frame_data)
            
            # Progress update
            if (frame_id + 1) % 100 == 0:
                print(f"  Processed {frame_id + 1}/{total_frames} frames "
                      f"({len(self.frame_data_list)} with detections)")
            
            frame_id += 1
        
        cap.release()
        
        # Save results
        self._save_results(output_dir)
        
        print("\n" + "="*70)
        print("✅ Processing complete!")
        print(f"   Frames processed: {frame_id}")
        print(f"   Frames with detections: {len(self.frame_data_list)}")
        print(f"   Team clusterer trained: {self.team_clusterer_trained}")
        print("="*70)
    
    def _save_results(self, output_dir: Path):
        """Save processing results to JSON"""
        # Convert FrameData to JSON-serializable format
        results = []
        for frame_data in self.frame_data_list:
            frame_dict = {
                "frame_id": frame_data.frame_id,
                "timestamp": frame_data.timestamp,
                "players": [
                    {
                        "object_id": p.object_id,
                        "team_id": p.team_id,
                        "x_pitch": p.x_pitch,
                        "y_pitch": p.y_pitch,
                        "bbox": p.bbox,
                    }
                    for p in frame_data.players
                ],
                "num_players": len(frame_data.players)
            }
            results.append(frame_dict)
        
        # Save JSON
        output_path = output_dir / "frame_data.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Save summary
        summary = {
            "total_frames": len(self.frame_data_list),
            "team_clusterer_trained": self.team_clusterer_trained,
            "pitch_dimensions": {
                "length": self.pitch_length,
                "width": self.pitch_width
            }
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Process video with R-001, R-002, and R-003 pipeline"
    )
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to RF-DETR checkpoint")
    parser.add_argument("--homography", "-H", type=str, default=None,
                       help="Path to homography JSON file (from calibrate_homography.py)")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--golden-batch", type=int, default=500,
                       help="Number of frames for Golden Batch initialization")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoProcessingPipeline(
        model_path=args.model,
        homography_path=args.homography,
        golden_batch_size=args.golden_batch,
        confidence_threshold=args.confidence
    )
    
    # Process video
    pipeline.process_video(args.video, args.output, args.max_frames)


if __name__ == "__main__":
    main()
