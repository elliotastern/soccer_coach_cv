# Data Classes for annotation package (standalone subset of src.types). See README in annotation/.
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
    role: Optional[str] = None  # "PLAYER", "GK", "REF"


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
