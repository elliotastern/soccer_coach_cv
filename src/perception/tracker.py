"""ByteTrack multi-object tracker (supervision) with tracklet emit gating."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from supervision.tracker.byte_tracker.core import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    try:
        from supervision.tracker.byte_tracker import ByteTrack
        BYTETRACK_AVAILABLE = True
    except ImportError:
        BYTETRACK_AVAILABLE = False
        ByteTrack = None

from supervision.detection.core import Detections

from src.state.types import Detection, TrackedObject


def _build_byte_track(
    track_activation_threshold: float,
    lost_track_buffer: int,
    minimum_matching_threshold: float,
    frame_rate: int,
    minimum_consecutive_frames: int = 1,
):
    if not BYTETRACK_AVAILABLE:
        raise ImportError("ByteTrack not available. Install: pip install supervision")
    return ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        frame_rate=frame_rate,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )


class Tracker:
    """ByteTrack tracker: low-conf ingest, high-conf emit via tracklet EMA."""

    def __init__(
        self,
        track_thresh: float = 0.10,
        high_thresh: float = 0.35,  # kept for config compat; unused by current API
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
        emit_thresh: float = 0.80,
        ema_alpha: float = 0.3,
        apply_emit_gate: bool = True,
    ):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.emit_thresh = emit_thresh
        self.ema_alpha = ema_alpha
        self.apply_emit_gate = apply_emit_gate
        self._ema: Dict[int, float] = {}
        self.byte_tracker = _build_byte_track(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        if not detections:
            empty = Detections.empty()
            self.byte_tracker.update_with_detections(empty)
            return []

        xyxy = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] for d in detections],
            dtype=np.float32,
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=np.int32)
        supervision_detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracks = self.byte_tracker.update_with_detections(supervision_detections)
        objects = self._tracks_to_objects(tracks, detections)
        return self._apply_emit_gate(objects)

    def tracklet_ema(self, track_id: int) -> Optional[float]:
        return self._ema.get(track_id)

    def _update_ema(self, track_id: int, conf: float) -> float:
        prev = self._ema.get(track_id)
        if prev is None:
            ema = conf
        else:
            ema = self.ema_alpha * conf + (1.0 - self.ema_alpha) * prev
        self._ema[track_id] = ema
        return ema

    def _apply_emit_gate(self, objects: List[TrackedObject]) -> List[TrackedObject]:
        published = []
        live_ids = set()
        for obj in objects:
            live_ids.add(obj.object_id)
            ema = self._update_ema(obj.object_id, float(obj.detection.confidence))
            # Attach EMA on detection confidence for emit score when gating
            if not self.apply_emit_gate:
                published.append(obj)
                continue
            # Players: keep legacy behavior (publish if track exists); balls: EMA gate
            if obj.detection.class_name != "ball":
                published.append(obj)
                continue
            if ema >= self.emit_thresh or float(obj.detection.confidence) >= self.emit_thresh:
                published.append(
                    TrackedObject(
                        object_id=obj.object_id,
                        detection=Detection(
                            class_id=obj.detection.class_id,
                            confidence=max(float(obj.detection.confidence), ema),
                            bbox=obj.detection.bbox,
                            class_name=obj.detection.class_name,
                        ),
                        team_id=obj.team_id,
                        role=obj.role,
                    )
                )
        # Drop EMAs for dead tracks
        for tid in list(self._ema.keys()):
            if tid not in live_ids:
                del self._ema[tid]
        return published

    def _tracks_to_objects(self, tracks: Detections, detections: List[Detection]) -> List[TrackedObject]:
        if tracks is None or len(tracks) == 0:
            return []
        tracked_objects = []
        tracker_ids = tracks.tracker_id
        for i in range(len(tracks)):
            track_xyxy = tracks.xyxy[i]
            tid = int(tracker_ids[i]) if tracker_ids is not None else i
            det = self._nearest_detection(track_xyxy, detections)
            if det is None:
                x1, y1, x2, y2 = map(float, track_xyxy)
                conf = float(tracks.confidence[i]) if tracks.confidence is not None else 0.5
                cid = int(tracks.class_id[i]) if tracks.class_id is not None else 0
                det = Detection(
                    class_id=cid,
                    confidence=conf,
                    bbox=(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)),
                    class_name="player" if cid == 0 else "ball",
                )
            tracked_objects.append(TrackedObject(object_id=tid, detection=det))
        return tracked_objects

    def _nearest_detection(self, track_xyxy, detections: List[Detection]) -> Optional[Detection]:
        tcx = (float(track_xyxy[0]) + float(track_xyxy[2])) / 2
        tcy = (float(track_xyxy[1]) + float(track_xyxy[3])) / 2
        best, best_dist = None, 80.0
        for det in detections:
            x, y, w, h = det.bbox
            dist = ((x + w / 2 - tcx) ** 2 + (y + h / 2 - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist = det, dist
        return best

    def reset(self) -> None:
        self._ema.clear()
        self.byte_tracker = _build_byte_track(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
