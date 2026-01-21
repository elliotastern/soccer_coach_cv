"""
Extract a single frame from a video file
"""
import cv2
from pathlib import Path
from typing import Tuple, Optional


def extract_frame(
    video_path: str,
    frame_id: int,
    output_path: Optional[str] = None
) -> Tuple[str, int, int]:
    """
    Extract a single frame from video
    
    Args:
        video_path: Path to video file
        frame_id: Frame number to extract (0-indexed)
        output_path: Optional output path. If None, generates from video path
    
    Returns:
        Tuple of (output_image_path, width, height)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if frame_id >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_id} out of range (total: {total_frames})")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    # Read frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_id} from video")
    
    # Generate output path if not provided
    if output_path is None:
        video_stem = Path(video_path).stem
        output_dir = Path(video_path).parent / "temp_frames"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{video_stem}_frame_{frame_id:06d}.jpg")
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save frame
    cv2.imwrite(output_path, frame)
    
    return (output_path, width, height)
