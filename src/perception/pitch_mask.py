"""On-pitch gate: drop ball boxes on brown sidelines / off-turf clutter."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.state.types import Detection

BBox = Tuple[float, float, float, float]
Pred = Tuple[BBox, float, float]  # box, conf, side


def _crop_pad(frame, bbox: BBox, pad_scale: float = 1.5):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    pad = int(max(bw, bh) * pad_scale)
    x1 = int(max(0, x - pad))
    y1 = int(max(0, y - pad))
    x2 = int(min(w, x + bw + pad))
    y2 = int(min(h, y + bh + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def turf_ratios(crop_bgr) -> tuple[float, float]:
    """Return (green_frac, brown_frac) in a local crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, 1.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Match 2 indoor turf: green stripes; sideline track is reddish-brown.
    green = ((h >= 35) & (h <= 95) & (s >= 40) & (v >= 35))
    brown = ((h <= 25) | (h >= 160)) & (s >= 35) & (v >= 30) & (~green)
    n = float(h.size)
    return float(np.mean(green)), float(np.mean(brown))


def on_pitch_bbox(
    frame_bgr,
    bbox: BBox,
    min_green: float = 0.22,
    max_brown: float = 0.50,
    pad_scale: float = 0.35,
) -> bool:
    """True if the box itself sits on turf, not brown sideline track.

    Uses a tight pad so on-pitch balls near the touchline are not rejected
    just because brown track appears in a wide neighborhood.
    """
    crop = _crop_pad(frame_bgr, bbox, pad_scale=pad_scale)
    green, brown = turf_ratios(crop)
    if green < min_green:
        return False
    if brown > max_brown and brown >= green:
        return False
    return True


def on_pitch_det(frame_bgr, det: Detection, **kwargs) -> bool:
    return on_pitch_bbox(frame_bgr, tuple(det.bbox), **kwargs)


def filter_dets_on_pitch(frame_bgr, dets: Sequence[Detection], **kwargs) -> List[Detection]:
    return [d for d in dets if d.class_name != "ball" or on_pitch_det(frame_bgr, d, **kwargs)]


def filter_pred_map_on_pitch(
    frames_by_cam: Dict[str, object],
    pred_map: Dict[str, List[Pred]],
    **kwargs,
) -> Dict[str, List[Pred]]:
    """Keep only on-pitch preds. frames_by_cam[cam] = BGR frame for that cam."""
    out: Dict[str, List[Pred]] = {}
    for cam, preds in pred_map.items():
        frame = frames_by_cam.get(cam)
        if frame is None:
            out[cam] = list(preds)
            continue
        kept = [p for p in preds if on_pitch_bbox(frame, p[0], **kwargs)]
        if kept:
            out[cam] = kept
    return out
