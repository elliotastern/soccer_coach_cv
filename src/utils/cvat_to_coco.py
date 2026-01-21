"""
Convert CVAT XML annotations to COCO format
Extracts annotations for a specific frame and converts to COCO JSON
"""
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_cvat_xml(xml_path: str) -> ET.ElementTree:
    """Parse CVAT XML file"""
    tree = ET.parse(xml_path)
    return tree


def extract_frame_annotations(tree: ET.ElementTree, frame_id: int = 0) -> List[Dict]:
    """
    Extract annotations for a specific frame from CVAT XML
    
    Args:
        tree: Parsed XML tree
        frame_id: Frame number to extract (default: 0)
    
    Returns:
        List of annotation dicts with keys: track_id, label, bbox (xtl, ytl, xbr, ybr)
    """
    root = tree.getroot()
    annotations = []
    
    # Find all tracks
    tracks = root.findall('.//track')
    
    for track in tracks:
        track_id = track.get('id')
        label = track.get('label', 'player')
        
        # Find boxes for this frame
        boxes = track.findall(f'.//box[@frame="{frame_id}"]')
        
        for box in boxes:
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            # Get confidence if available
            confidence = 1.0
            conf_attr = box.find('.//attribute[@name="confidence"]')
            if conf_attr is not None:
                try:
                    confidence = float(conf_attr.text)
                except (ValueError, TypeError):
                    pass
            
            annotations.append({
                'track_id': track_id,
                'label': label,
                'bbox': (xtl, ytl, xbr, ybr),
                'confidence': confidence
            })
    
    return annotations


def cvat_bbox_to_coco(xtl: float, ytl: float, xbr: float, ybr: float) -> Tuple[float, float, float, float]:
    """
    Convert CVAT bbox format (xtl, ytl, xbr, ybr) to COCO format (x, y, width, height)
    
    Args:
        xtl: Top-left x coordinate
        ytl: Top-left y coordinate
        xbr: Bottom-right x coordinate
        ybr: Bottom-right y coordinate
    
    Returns:
        Tuple of (x, y, width, height)
    """
    x = xtl
    y = ytl
    width = xbr - xtl
    height = ybr - ytl
    
    # Ensure non-negative dimensions
    width = max(0, width)
    height = max(0, height)
    
    return (x, y, width, height)


def label_to_category_id(label: str) -> int:
    """
    Map CVAT label to COCO category ID
    
    Args:
        label: Label name ("player", "ball", etc.)
    
    Returns:
        Category ID (1=player, 2=ball)
    """
    label_lower = label.lower()
    if label_lower == 'player':
        return 1
    elif label_lower == 'ball':
        return 2
    else:
        # Default to player for unknown labels
        return 1


def create_coco_json(
    image_path: str,
    image_id: int,
    width: int,
    height: int,
    annotations: List[Dict],
    output_path: Optional[str] = None
) -> Dict:
    """
    Create COCO format JSON from frame annotations
    
    Args:
        image_path: Path to image file
        image_id: Unique image ID
        width: Image width
        height: Image height
        annotations: List of annotation dicts from extract_frame_annotations()
        output_path: Optional path to save JSON file
    
    Returns:
        COCO format dictionary
    """
    # Categories
    categories = [
        {"id": 1, "name": "player", "supercategory": "object"},
        {"id": 2, "name": "ball", "supercategory": "object"}
    ]
    
    # Image entry
    image_entry = {
        "id": image_id,
        "file_name": Path(image_path).name,
        "width": width,
        "height": height
    }
    
    # Convert annotations to COCO format
    coco_annotations = []
    for ann_idx, ann in enumerate(annotations):
        xtl, ytl, xbr, ybr = ann['bbox']
        x, y, w, h = cvat_bbox_to_coco(xtl, ytl, xbr, ybr)
        
        category_id = label_to_category_id(ann['label'])
        
        coco_ann = {
            "id": ann_idx + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }
        
        coco_annotations.append(coco_ann)
    
    # Create COCO structure
    coco_data = {
        "info": {
            "description": "Single frame training dataset",
            "version": "1.0"
        },
        "licenses": [],
        "images": [image_entry],
        "annotations": coco_annotations,
        "categories": categories
    }
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    return coco_data


def convert_frame_to_coco(
    xml_path: str,
    frame_id: int,
    image_path: str,
    image_width: int,
    image_height: int,
    output_json_path: str
) -> Dict:
    """
    Main function to convert CVAT XML frame to COCO format
    
    Args:
        xml_path: Path to CVAT XML file
        frame_id: Frame number to extract (default: 0)
        image_path: Path to extracted frame image
        image_width: Image width
        image_height: Image height
        output_json_path: Path to save COCO JSON
    
    Returns:
        COCO format dictionary
    """
    # Parse XML
    tree = parse_cvat_xml(xml_path)
    
    # Extract annotations for frame
    annotations = extract_frame_annotations(tree, frame_id)
    
    # Create COCO JSON
    coco_data = create_coco_json(
        image_path=image_path,
        image_id=1,
        width=image_width,
        height=image_height,
        annotations=annotations,
        output_path=output_json_path
    )
    
    return coco_data
