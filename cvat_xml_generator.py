"""
CVAT XML Generator for Video 1.1 Format
Converts detections, tracks, and events to CVAT-compatible XML
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional
from pathlib import Path
import cv2
from src.types import Detection, TrackedObject, Event, EventType


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_track_xml(track_id: int, label: str, boxes: List[Dict], source: str = "auto") -> ET.Element:
    """
    Generate a <track> element for bounding boxes
    
    Args:
        track_id: Unique track ID
        label: Class label (player, ball)
        boxes: List of dicts with keys: frame, xtl, ytl, xbr, ybr, outside, keyframe, confidence
        source: Source of annotation (auto, manual)
    
    Returns:
        ET.Element for the track
    """
    track = ET.Element("track", {
        "id": str(track_id),
        "label": label,
        "source": source
    })
    
    for box_data in boxes:
        box = ET.SubElement(track, "box", {
            "frame": str(box_data["frame"]),
            "xtl": f"{box_data['xtl']:.2f}",
            "ytl": f"{box_data['ytl']:.2f}",
            "xbr": f"{box_data['xbr']:.2f}",
            "ybr": f"{box_data['ybr']:.2f}",
            "outside": str(box_data.get("outside", 0)),
            "occluded": str(box_data.get("occluded", 0)),
            "keyframe": str(box_data.get("keyframe", 1))
        })
        
        # Add confidence as attribute if available
        if "confidence" in box_data:
            attr = ET.SubElement(box, "attribute", {"name": "confidence"})
            attr.text = f"{box_data['confidence']:.3f}"
    
    return track


def generate_event_tag_xml(event: Event, source: str = "auto") -> ET.Element:
    """
    Generate a <tag> element for single-frame events
    
    Args:
        event: Event object
        source: Source of annotation (auto, manual)
    
    Returns:
        ET.Element for the tag
    """
    # For single-frame events, use the start_frame
    # If start_frame == end_frame, it's a single-frame event
    frame = event.start_frame if event.start_frame == event.end_frame else event.start_frame
    
    tag = ET.Element("tag", {
        "label": event.type.value,
        "frame": str(frame),
        "source": source
    })
    
    # Add confidence as attribute
    attr = ET.SubElement(tag, "attribute", {"name": "confidence"})
    attr.text = f"{event.confidence:.3f}"
    
    # Add involved players as attribute
    if event.involved_players:
        players_attr = ET.SubElement(tag, "attribute", {"name": "involved_players"})
        players_attr.text = ",".join(map(str, event.involved_players))
    
    return tag


def generate_duration_event_track(event: Event, source: str = "auto") -> ET.Element:
    """
    Generate a <track> element for duration events (pass, dribble)
    Uses a special "event" track that spans the duration
    
    Args:
        event: Event object with start_frame and end_frame
        source: Source of annotation (auto, manual)
    
    Returns:
        ET.Element for the event track
    """
    # Create a track ID based on event ID hash
    track_id = hash(event.id) % 1000000
    
    track = ET.Element("track", {
        "id": str(track_id),
        "label": f"event_{event.type.value}",
        "source": source
    })
    
    # Add box at start frame
    start_box = ET.SubElement(track, "box", {
        "frame": str(event.start_frame),
        "xtl": "0",
        "ytl": "0",
        "xbr": "1",
        "ybr": "1",
        "outside": "0",
        "occluded": "0",
        "keyframe": "1"
    })
    
    # Add confidence attribute
    attr = ET.SubElement(start_box, "attribute", {"name": "confidence"})
    attr.text = f"{event.confidence:.3f}"
    
    # Add box at end frame if different
    if event.end_frame != event.start_frame:
        end_box = ET.SubElement(track, "box", {
            "frame": str(event.end_frame),
            "xtl": "0",
            "ytl": "0",
            "xbr": "1",
            "ybr": "1",
            "outside": "1",  # Mark as outside to end the track
            "occluded": "0",
            "keyframe": "1"
        })
    
    return track


def create_cvat_xml(
    video_path: str,
    tracked_objects_by_frame: Dict[int, List[TrackedObject]],
    events: List[Event],
    video_metadata: Optional[Dict] = None
) -> str:
    """
    Create complete CVAT XML document for Video 1.1 format
    
    Args:
        video_path: Path to video file
        tracked_objects_by_frame: Dict mapping frame_id -> List[TrackedObject]
        events: List of detected events
        video_metadata: Optional dict with video info (width, height, fps, etc.)
    
    Returns:
        Pretty-printed XML string
    """
    # Get video metadata if not provided
    if video_metadata is None:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_metadata = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            cap.release()
        else:
            video_metadata = {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 0}
    
    # Create root element
    root = ET.Element("annotations")
    
    # Add version
    version = ET.SubElement(root, "version")
    version.text = "1.1"
    
    # Add meta section
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    
    # Task info
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = Path(video_path).stem
    ET.SubElement(task, "size").text = str(video_metadata.get("frame_count", 0))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "created").text = ""
    ET.SubElement(task, "updated").text = ""
    ET.SubElement(task, "subset").text = ""
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(video_metadata.get("frame_count", 0) - 1)
    ET.SubElement(task, "frame_filter").text = ""
    ET.SubElement(task, "segments").text = ""
    ET.SubElement(task, "owner").text = ET.SubElement(task, "assignee").text = ""
    
    # Labels
    labels = ET.SubElement(task, "labels")
    
    # Player label
    player_label = ET.SubElement(labels, "label")
    ET.SubElement(player_label, "name").text = "player"
    ET.SubElement(player_label, "color").text = "#ff0000"
    player_attrs = ET.SubElement(player_label, "attributes")
    conf_attr = ET.SubElement(player_attrs, "attribute")
    ET.SubElement(conf_attr, "name").text = "confidence"
    ET.SubElement(conf_attr, "mutable").text = "false"
    ET.SubElement(conf_attr, "input_type").text = "number"
    ET.SubElement(conf_attr, "default_value").text = "0.0"
    ET.SubElement(conf_attr, "values").text = ""
    
    # Ball label
    ball_label = ET.SubElement(labels, "label")
    ET.SubElement(ball_label, "name").text = "ball"
    ET.SubElement(ball_label, "color").text = "#00ff00"
    ball_attrs = ET.SubElement(ball_label, "attributes")
    conf_attr = ET.SubElement(ball_attrs, "attribute")
    ET.SubElement(conf_attr, "name").text = "confidence"
    ET.SubElement(conf_attr, "mutable").text = "false"
    ET.SubElement(conf_attr, "input_type").text = "number"
    ET.SubElement(conf_attr, "default_value").text = "0.0"
    ET.SubElement(conf_attr, "values").text = ""
    
    # Event labels (for tags)
    event_types = ["pass", "dribble", "shot", "recovery", "movement"]
    for event_type in event_types:
        event_label = ET.SubElement(labels, "label")
        ET.SubElement(event_label, "name").text = event_type
        ET.SubElement(event_label, "color").text = "#0000ff"
        event_attrs = ET.SubElement(event_label, "attributes")
        conf_attr = ET.SubElement(event_attrs, "attribute")
        ET.SubElement(conf_attr, "name").text = "confidence"
        ET.SubElement(conf_attr, "mutable").text = "false"
        ET.SubElement(conf_attr, "input_type").text = "number"
        ET.SubElement(conf_attr, "default_value").text = "0.0"
        ET.SubElement(conf_attr, "values").text = ""
        players_attr = ET.SubElement(event_attrs, "attribute")
        ET.SubElement(players_attr, "name").text = "involved_players"
        ET.SubElement(players_attr, "mutable").text = "false"
        ET.SubElement(players_attr, "input_type").text = "text"
        ET.SubElement(players_attr, "default_value").text = ""
        ET.SubElement(players_attr, "values").text = ""
    
    # Group tracked objects by track_id and class
    tracks_dict = {}  # (track_id, class_name) -> List[boxes]
    
    for frame_id, tracked_objects in tracked_objects_by_frame.items():
        for tracked_obj in tracked_objects:
            det = tracked_obj.detection
            x, y, w, h = det.bbox
            key = (tracked_obj.object_id, det.class_name)
            
            if key not in tracks_dict:
                tracks_dict[key] = []
            
            tracks_dict[key].append({
                "frame": frame_id,
                "xtl": x,
                "ytl": y,
                "xbr": x + w,
                "ybr": y + h,
                "outside": 0,
                "occluded": 0,
                "keyframe": 1,
                "confidence": det.confidence
            })
    
    # Generate track elements
    track_id_counter = 0
    for (obj_id, class_name), boxes in tracks_dict.items():
        # Sort boxes by frame
        boxes.sort(key=lambda b: b["frame"])
        track = generate_track_xml(track_id_counter, class_name, boxes)
        root.append(track)
        track_id_counter += 1
    
    # Generate event tags
    for event in events:
        # Single-frame events use tags
        if event.start_frame == event.end_frame:
            tag = generate_event_tag_xml(event)
            root.append(tag)
        else:
            # Duration events can use tags at start frame or tracks
            # Using tags for simplicity - CVAT will show them on timeline
            tag = generate_event_tag_xml(event)
            root.append(tag)
    
    # Return pretty-printed XML
    return prettify_xml(root)
