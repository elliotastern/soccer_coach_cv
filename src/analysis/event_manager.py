# Event Manager - Aggregates and manages events
from typing import List, Dict
import json
import os
from src.types import Event, MatchData
from src.schema import events_to_json


class EventManager:
    """Manages event aggregation and checkpointing"""
    
    def __init__(self, checkpoint_interval: int = 300, output_dir: str = "data/output"):
        """
        Initialize event manager
        
        Args:
            checkpoint_interval: Save checkpoint every N frames
            output_dir: Output directory for checkpoints and final files
        """
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.events: List[Event] = []
        self.frame_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    def add_events(self, events: List[Event]):
        """Add new events"""
        self.events.extend(events)
        self.frame_count += 1
        
        # Checkpoint if needed
        if self.frame_count % self.checkpoint_interval == 0:
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save checkpoint to disk"""
        checkpoint_path = os.path.join(
            self.output_dir,
            "checkpoints",
            f"checkpoint_frame_{self.frame_count}.json"
        )
        
        events_json = events_to_json(self.events)
        with open(checkpoint_path, 'w') as f:
            json.dump({
                "frame_count": self.frame_count,
                "events": events_json
            }, f, indent=2)
    
    def save_final_output(self, match_id: str, csv_path: str = None, json_path: str = None):
        """
        Save final output files
        
        Args:
            match_id: Match identifier
            csv_path: Path for CSV output (uses default if None)
            json_path: Path for JSON output (uses default if None)
        """
        if csv_path is None:
            csv_path = os.path.join(self.output_dir, "events.csv")
        if json_path is None:
            json_path = os.path.join(self.output_dir, "events.json")
        
        # Save JSON
        match_data = MatchData(
            match_id=match_id,
            events=self.events,
            metadata={"total_frames": self.frame_count}
        )
        
        events_json = events_to_json(self.events)
        with open(json_path, 'w') as f:
            json.dump({
                "match_id": match_id,
                "events": events_json,
                "metadata": match_data.metadata
            }, f, indent=2)
        
        # CSV will be saved by main orchestrator using schema.py
    
    def get_events(self) -> List[Event]:
        """Get all events"""
        return self.events
