# Output Schema Definitions (Phase 1 export)
from __future__ import annotations

from typing import Dict, List

from src.state.types import Event, FrameData


def get_csv_schema() -> List[str]:
    """Phase 1 CSV columns (plus frame_id for review)."""
    return [
        "Timestamp",
        "Team_ID",
        "Player_ID",
        "Event",
        "Location_X",
        "Location_Y",
        "frame_id",
        "confidence",
    ]


def frame_data_to_csv_row(
    frame_data: FrameData, event_type: str = None, confidence: float = None
) -> List[Dict]:
    """Convert FrameData to Phase 1 CSV rows."""
    rows = []
    evt = event_type or "movement"
    conf = 1.0 if confidence is None else float(confidence)

    for player in frame_data.players:
        rows.append(
            {
                "Timestamp": round(float(frame_data.timestamp), 3),
                "Team_ID": int(player.team_id),
                "Player_ID": int(player.object_id),
                "Event": evt,
                "Location_X": round(float(player.x_pitch), 3),
                "Location_Y": round(float(player.y_pitch), 3),
                "frame_id": int(frame_data.frame_id),
                "confidence": conf,
            }
        )

    if frame_data.ball:
        rows.append(
            {
                "Timestamp": round(float(frame_data.timestamp), 3),
                "Team_ID": -1,
                "Player_ID": -1,
                "Event": evt,
                "Location_X": round(float(frame_data.ball.x_pitch), 3),
                "Location_Y": round(float(frame_data.ball.y_pitch), 3),
                "frame_id": int(frame_data.frame_id),
                "confidence": conf,
            }
        )
    return rows


def events_to_csv_rows(events: List[Event]) -> List[Dict]:
    """One row per event for Phase 1 events.csv."""
    rows = []
    for event in events:
        team = -1
        player = -1
        if event.involved_players:
            player = int(event.involved_players[0])
        rows.append(
            {
                "Timestamp": round(float(event.timestamp_start), 3),
                "Team_ID": team,
                "Player_ID": player,
                "Event": event.type.value,
                "Location_X": round(float(event.start_location.x), 3),
                "Location_Y": round(float(event.start_location.y), 3),
                "frame_id": int(event.start_frame),
                "confidence": round(float(event.confidence), 3),
            }
        )
    return rows


def events_to_json(events: List[Event]) -> List[Dict]:
    """Convert events to JSON-serializable format."""
    return [
        {
            "id": event.id,
            "type": event.type.value,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_location": {
                "x": event.start_location.x,
                "y": event.start_location.y,
            },
            "end_location": {
                "x": event.end_location.x,
                "y": event.end_location.y,
            },
            "involved_players": event.involved_players,
            "confidence": event.confidence,
            "timestamp_start": event.timestamp_start,
            "timestamp_end": event.timestamp_end,
        }
        for event in events
    ]
