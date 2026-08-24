"""Tracklet-level team assignment for batch export (Golden Batch + propagate)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.perception.team_core import (
    TEAM_MIN_TRACKLETS,
    assign_feature,
    fit_match_centroids,
    jersey_feature,
    load_centroids,
    save_centroids,
    torso_crop,
    tracklet_median_feature,
)
from src.state.types import TrackedObject


class TrackletAccumulator:
    """Collect jersey features per ByteTrack id during Golden Batch window."""

    def __init__(self):
        self._feats: dict[int, list[np.ndarray]] = defaultdict(list)
        self._positions: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def add(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        bbox,
        pitch_xy: tuple[float, float] | None = None,
        cam: str | None = None,
        frame_wh: tuple[int, int] | None = None,
    ) -> None:
        crop = torso_crop(frame_bgr, bbox, cam=cam, frame_wh=frame_wh)
        feat = jersey_feature(crop) if crop is not None else None
        if feat is None:
            return
        self._feats[int(track_id)].append(feat)
        if pitch_xy is not None:
            self._positions[int(track_id)].append(pitch_xy)

    def tracklet_medians(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for tid, feats in self._feats.items():
            med = tracklet_median_feature(feats)
            if med is not None:
                out[tid] = med
        return out

    def tracklet_positions(self) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for tid, pts in self._positions.items():
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                out[tid] = (float(np.median(xs)), float(np.median(ys)))
        return out


class TrackletTeamModel:
    """Match-level centroids + per-tracklet labels."""

    def __init__(self):
        self.centroids: np.ndarray | None = None
        self.radius: float | None = None
        self.track_labels: dict[int, tuple[int, float]] = {}
        self._live_feats: dict[int, list[np.ndarray]] = defaultdict(list)

    def fit_from_accumulator(
        self,
        acc: TrackletAccumulator,
        min_tracklets: int = TEAM_MIN_TRACKLETS,
    ) -> bool:
        medians = acc.tracklet_medians()
        positions = acc.tracklet_positions()
        if len(medians) < min_tracklets:
            return False
        fit = fit_match_centroids(list(medians.values()), min_crops=min_tracklets)
        if fit is None:
            return False
        self.centroids, self.radius = fit
        self.track_labels = {}
        for tid, feat in medians.items():
            pos = positions.get(tid)
            self.track_labels[tid] = assign_feature(
                feat, self.centroids, self.radius, position_xy=pos
            )
        return True

    def load(self, path: Path) -> bool:
        loaded = load_centroids(path)
        if loaded is None:
            return False
        self.centroids, self.radius = loaded
        labels_path = path.with_name("team_track_labels.json")
        if labels_path.is_file():
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            self.track_labels = {int(k): (int(v[0]), float(v[1])) for k, v in raw.items()}
        return True

    def save(self, path: Path) -> None:
        if self.centroids is None or self.radius is None:
            return
        save_centroids(path, self.centroids, self.radius)
        labels_path = path.with_name("team_track_labels.json")
        payload = {str(k): [int(v[0]), float(v[1])] for k, v in self.track_labels.items()}
        labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def label_track(
        self,
        track_id: int,
        frame_bgr: np.ndarray | None = None,
        bbox=None,
        pitch_xy: tuple[float, float] | None = None,
        cam: str | None = None,
        frame_wh: tuple[int, int] | None = None,
    ) -> tuple[int, float]:
        tid = int(track_id)
        if tid in self.track_labels:
            return self.track_labels[tid]
        if self.centroids is None or self.radius is None:
            return -1, 0.0
        if frame_bgr is not None and bbox is not None:
            crop = torso_crop(frame_bgr, bbox, cam=cam, frame_wh=frame_wh)
            feat = jersey_feature(crop) if crop is not None else None
            if feat is not None:
                self._live_feats[tid].append(feat)
                med = tracklet_median_feature(self._live_feats[tid])
                if med is not None:
                    lab = assign_feature(med, self.centroids, self.radius, position_xy=pitch_xy)
                    self.track_labels[tid] = lab
                    return lab
        return -1, 0.0

    def apply_to_tracked(
        self,
        tracked_objects: list[TrackedObject],
        frame_bgr: np.ndarray,
        player_class_id: int = 0,
        pitch_positions: dict[int, tuple[float, float]] | None = None,
        cam: str | None = None,
        frame_wh: tuple[int, int] | None = None,
    ) -> list[TrackedObject]:
        wh = frame_wh or (frame_bgr.shape[1], frame_bgr.shape[0])
        for obj in tracked_objects:
            det = obj.detection
            if int(det.class_id) != int(player_class_id):
                continue
            pos = pitch_positions.get(int(obj.object_id)) if pitch_positions else None
            tid, _conf = self.label_track(
                obj.object_id,
                frame_bgr=frame_bgr,
                bbox=det.bbox,
                pitch_xy=pos,
                cam=cam,
                frame_wh=wh,
            )
            obj.team_id = int(tid)
        return tracked_objects
