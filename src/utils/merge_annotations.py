"""
Merge new predictions into existing CVAT XML, preserving frame 0 annotations
"""
import xml.etree.ElementTree as ET
from typing import Dict, List
from pathlib import Path
import cv2

from cvat_xml_generator import create_cvat_xml
from src.types import TrackedObject, Event


def parse_existing_xml(xml_path: str) -> Dict:
    """
    Parse existing CVAT XML to extract frame 0 annotations and metadata
    
    Args:
        xml_path: Path to existing CVAT XML file
    
    Returns:
        Dictionary with:
            - frame_0_tracks: Dict mapping track_id -> track element
            - video_metadata: Dict with width, height, fps, frame_count
            - events: List of event elements
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract video metadata
    meta = root.find('.//meta/task')
    if meta is not None:
        size_elem = meta.find('size')
        frame_count = int(size_elem.text) if size_elem is not None else 0
    else:
        frame_count = 0
    
    # Get video path from XML if available
    video_path = None
    source_elem = root.find('.//source')
    if source_elem is not None:
        video_path = source_elem.text
    
    # Extract video metadata from video file if available
    video_metadata = {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": frame_count}
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_metadata = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            cap.release()
    
    # Extract frame 0 annotations (preserve these)
    frame_0_tracks = {}
    all_tracks = root.findall('.//track')
    
    for track in all_tracks:
        track_id = track.get('id')
        label = track.get('label', 'player')
        source = track.get('source', 'manual')
        
        # Find boxes in frame 0
        frame_0_boxes = track.findall('.//box[@frame="0"]')
        
        if frame_0_boxes:
            # Create a copy of the track with ALL boxes (not just frame 0)
            # This preserves existing annotations beyond frame 0
            frame_0_track = ET.Element('track', {
                'id': track_id,
                'label': label,
                'source': source  # Preserve source attribute
            })
            
            # Add ALL boxes from this track (preserve existing annotations)
            for box in track.findall('.//box'):
                frame_0_track.append(box)
            
            frame_0_tracks[track_id] = frame_0_track
    
    # Extract events
    events = root.findall('.//tag')
    
    return {
        'frame_0_tracks': frame_0_tracks,
        'video_metadata': video_metadata,
        'events': events
    }


def convert_tracked_objects_to_dict(
    tracked_objects_by_frame: Dict[int, List[TrackedObject]]
) -> Dict[int, List[Dict]]:
    """
    Convert TrackedObject list to dictionary format expected by create_cvat_xml
    
    Args:
        tracked_objects_by_frame: Dict mapping frame_id -> List[TrackedObject]
    
    Returns:
        Dict mapping frame_id -> List of box dicts
    """
    result = {}
    
    for frame_id, tracked_objects in tracked_objects_by_frame.items():
        frame_boxes = []
        
        for tracked_obj in tracked_objects:
            det = tracked_obj.detection
            x, y, w, h = det.bbox
            
            frame_boxes.append({
                "frame": frame_id,
                "xtl": x,
                "ytl": y,
                "xbr": x + w,
                "ybr": y + h,
                "outside": 0,
                "occluded": 0,
                "keyframe": 1,
                "confidence": det.confidence,
                "track_id": tracked_obj.object_id,
                "label": det.class_name
            })
        
        result[frame_id] = frame_boxes
    
    return result


def merge_annotations(
    original_xml_path: str,
    video_path: str,
    new_tracked_objects_by_frame: Dict[int, List[TrackedObject]],
    output_xml_path: str
):
    """
    Merge new predictions into existing XML, preserving frame 0
    
    Args:
        original_xml_path: Path to original CVAT XML with frame 0 annotations
        video_path: Path to video file
        new_tracked_objects_by_frame: New predictions for frames 1+
        output_xml_path: Path to save merged XML
    """
    # Parse existing XML
    print(f"Parsing existing XML: {original_xml_path}")
    existing_data = parse_existing_xml(original_xml_path)
    
    frame_0_tracks = existing_data['frame_0_tracks']
    video_metadata = existing_data['video_metadata']
    
    # Convert new tracked objects to format for XML generation
    print(f"Converting {len(new_tracked_objects_by_frame)} frames of new predictions...")
    new_boxes_by_frame = convert_tracked_objects_to_dict(new_tracked_objects_by_frame)
    
    # Merge frame 0 boxes with new boxes
    # Group boxes by track_id across all frames
    all_tracks_dict = {}
    
    # Add frame 0 tracks (preserve original)
    for track_id, track_elem in frame_0_tracks.items():
        label = track_elem.get('label', 'player')
        source = track_elem.get('source', 'manual')
        
        boxes = []
        for box_elem in track_elem.findall('.//box'):
            frame = int(box_elem.get('frame'))
            boxes.append({
                "frame": frame,
                "xtl": float(box_elem.get('xtl')),
                "ytl": float(box_elem.get('ytl')),
                "xbr": float(box_elem.get('xbr')),
                "ybr": float(box_elem.get('ybr')),
                "outside": int(box_elem.get('outside', 0)),
                "occluded": int(box_elem.get('occluded', 0)),
                "keyframe": int(box_elem.get('keyframe', 1)),
                "confidence": 1.0,  # Manual annotations have full confidence
                "track_id": track_id,
                "label": label
            })
        
        all_tracks_dict[track_id] = {
            'label': label,
            'source': source,
            'boxes': boxes
        }
    
    # Add new predictions (frames 1+)
    for frame_id, frame_boxes in new_boxes_by_frame.items():
        for box in frame_boxes:
            track_id = box['track_id']
            label = box.get('label', 'player')
            
            if track_id not in all_tracks_dict:
                all_tracks_dict[track_id] = {
                    'label': label,
                    'source': 'auto',
                    'boxes': []
                }
            
            all_tracks_dict[track_id]['boxes'].append(box)
    
    # Instead of using create_cvat_xml (which reassigns track IDs),
    # we'll directly modify the original XML to preserve track IDs
    print(f"Preserving original XML structure and track IDs...")
    
    # Load original XML tree
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    
    # Remove all existing tracks and tags (we'll rebuild tracks preserving IDs)
    # But keep meta, version, and other structure
    for track in root.findall('.//track'):
        root.remove(track)
    for tag in root.findall('.//tag'):
        root.remove(tag)
    
    # Rebuild tracks preserving original track IDs
    for track_id, track_data in all_tracks_dict.items():
        # Create track element with original ID
        track_elem = ET.Element('track', {
            'id': str(track_id),
            'label': track_data['label'],
            'source': track_data.get('source', 'manual')
        })
        
        # Sort boxes by frame
        sorted_boxes = sorted(track_data['boxes'], key=lambda b: b['frame'])
        
        # Add boxes to track
        for box in sorted_boxes:
            box_elem = ET.SubElement(track_elem, 'box', {
                'frame': str(box['frame']),
                'xtl': f"{box['xtl']:.2f}",
                'ytl': f"{box['ytl']:.2f}",
                'xbr': f"{box['xbr']:.2f}",
                'ybr': f"{box['ybr']:.2f}",
                'outside': str(box.get('outside', 0)),
                'occluded': str(box.get('occluded', 0)),
                'keyframe': str(box.get('keyframe', 1))
            })
            
            # Add confidence attribute if present
            if 'confidence' in box:
                conf_attr = ET.SubElement(box_elem, 'attribute', {'name': 'confidence'})
                conf_attr.text = f"{box['confidence']:.3f}"
        
        # Append track to root
        root.append(track_elem)
    
    # Preserve events from original XML
    # (events are already in the tree, we just need to make sure they're not removed)
    
    # Generate pretty-printed XML
    from cvat_xml_generator import prettify_xml
    xml_content = prettify_xml(root)
    
    # Save merged XML
    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"✅ Merged XML saved to: {output_xml_path}")
    print(f"   - Preserved {len(frame_0_tracks)} tracks from frame 0")
    print(f"   - Added {len(new_tracked_objects_by_frame)} frames of new predictions")
