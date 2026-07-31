"""File and RTSP video ingest adapters.

Product Phase 1 uses file/batch ingest. RTSP live ingest is Phase 2+.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def open_video_file(path: str):
    """Open a local video file for batch processing."""
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def open_rtsp_stream(url: str):
    """Open an RTSP stream (Product Phase 2+ live path)."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {url}")
    return cap


def iter_frames(cap, max_frames: Optional[int] = None) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_id, timestamp_sec, frame) from an opened capture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    while True:
        if max_frames is not None and frame_id >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_id / fps
        yield frame_id, timestamp, frame
        frame_id += 1
