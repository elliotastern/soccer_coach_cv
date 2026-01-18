# Data Classes (Don't pass dicts everywhere!)
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class EventType(Enum):
    PASS = "pass"
    DRIBBLE = "dribble"
    MOVEMENT = "movement"
    RECOVERY = "recovery"
    SHOT = "shot"


@dataclass
class Detection:
    """Single detection from RF-DETR"""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    class_name: str = ""


@dataclass
class TrackedObject:
    """Tracked object with ID"""
    object_id: int
    detection: Detection
    team_id: Optional[int] = None


@dataclass
class Player:
    """Player with pitch coordinates"""
    object_id: int
    team_id: int
    x_pitch: float
    y_pitch: float
    bbox: Tuple[float, float, float, float]
    frame_id: int
    timestamp: float


@dataclass
class Ball:
    """Ball position"""
    x_pitch: float
    y_pitch: float
    bbox: Tuple[float, float, float, float]
    frame_id: int
    timestamp: float
    object_id: Optional[int] = None


@dataclass
class FrameData:
    """Data for a single frame"""
    frame_id: int
    timestamp: float
    players: List[Player]
    ball: Optional[Ball] = None
    detections: List[Detection] = None


@dataclass
class Location:
    """Pitch location"""
    x: float
    y: float


@dataclass
class Event:
    """Detected event"""
    id: str
    type: EventType
    start_frame: int
    end_frame: int
    start_location: Location
    end_location: Location
    involved_players: List[int]  # object_ids
    confidence: float
    timestamp_start: float
    timestamp_end: float


@dataclass
class MatchData:
    """Complete match data"""
    match_id: str
    events: List[Event]
    metadata: dict
