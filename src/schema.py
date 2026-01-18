# Output Schema Definitions
from typing import Dict, List
import pandas as pd
from src.types import Event, FrameData, Player, Ball


def get_csv_schema() -> List[str]:
    """Returns CSV column schema"""
    return [
        "frame_id",
        "timestamp",
        "object_id",
        "team_id",
        "x_pitch",
        "y_pitch",
        "event_type",
        "confidence"
    ]


def frame_data_to_csv_row(frame_data: FrameData, event_type: str = None, confidence: float = None) -> List[Dict]:
    """Convert FrameData to CSV rows"""
    rows = []
    
    for player in frame_data.players:
        rows.append({
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "object_id": player.object_id,
            "team_id": player.team_id,
            "x_pitch": player.x_pitch,
            "y_pitch": player.y_pitch,
            "event_type": event_type or "movement",
            "confidence": confidence or 1.0
        })
    
    if frame_data.ball:
        rows.append({
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "object_id": -1,  # Ball has special ID
            "team_id": -1,
            "x_pitch": frame_data.ball.x_pitch,
            "y_pitch": frame_data.ball.y_pitch,
            "event_type": event_type or "movement",
            "confidence": confidence or 1.0
        })
    
    return rows


def events_to_json(events: List[Event]) -> List[Dict]:
    """Convert events to JSON-serializable format"""
    return [
        {
            "id": event.id,
            "type": event.type.value,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_location": {
                "x": event.start_location.x,
                "y": event.start_location.y
            },
            "end_location": {
                "x": event.end_location.x,
                "y": event.end_location.y
            },
            "involved_players": event.involved_players,
            "confidence": event.confidence,
            "timestamp_start": event.timestamp_start,
            "timestamp_end": event.timestamp_end
        }
        for event in events
    ]
