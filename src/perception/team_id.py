"""
Team ID Assignment (R-002) — delegates to shared team_core.

Golden Batch + tracklet path: src.perception.team_tracklet
Live review: src.review.team_live
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.perception.team_core import (
    assign_feature,
    fit_match_centroids,
    in_pitch1_goal_box,
    jersey_feature,
    torso_crop,
    which_goal_box,
)


@dataclass
class TeamAssignment:
    team_id: Optional[int]
    role: str
    confidence: float
    is_outlier: bool


class TeamClusterer:
    """Compatibility wrapper over team_core (Pitch 1 spatial heuristics)."""

    def __init__(self, pitch_length: float = 53.9, pitch_width: float = 34.84):
        self.team_centroids: np.ndarray | None = None
        self.outlier_threshold: float = 0.0
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.accumulated_crops: list = []
        self.accumulated_positions: list = []

    def fit(
        self,
        player_crops: List[np.ndarray],
        positions: Optional[List[Tuple[float, float]]] = None,
        confidence_threshold: float = 0.8,
        min_crops: int = 20,
    ) -> bool:
        del confidence_threshold
        features = []
        for i, crop in enumerate(player_crops):
            feat = jersey_feature(torso_crop(crop, (0, 0, crop.shape[1], crop.shape[0])))
            if feat is None:
                feat = jersey_feature(crop)
            if feat is not None:
                features.append(feat)
                if positions and i < len(positions):
                    self.accumulated_positions.append(positions[i])
        if len(features) < min_crops:
            return False
        fit = fit_match_centroids(features, min_crops=min_crops)
        if fit is None:
            return False
        self.team_centroids, self.outlier_threshold = fit
        self.accumulated_crops = player_crops
        return True

    def predict(
        self, crop: np.ndarray, position_xy: Optional[Tuple[float, float]] = None
    ) -> TeamAssignment:
        if self.team_centroids is None:
            return TeamAssignment(None, "PLAYER", 0.0, True)
        tcrop = torso_crop(crop, (0, 0, crop.shape[1], crop.shape[0])) if crop is not None else None
        feature = jersey_feature(tcrop) if tcrop is not None else None
        if feature is None and crop is not None:
            feature = jersey_feature(crop)
        if feature is None:
            return TeamAssignment(None, "PLAYER", 0.0, True)
        tid, conf = assign_feature(
            feature, self.team_centroids, self.outlier_threshold, position_xy
        )
        if tid < 0:
            if position_xy and which_goal_box(position_xy):
                return TeamAssignment(None, "GK", conf, True)
            return TeamAssignment(None, "REF", conf, True)
        return TeamAssignment(int(tid), "PLAYER", conf, False)

    def predict_batch(
        self,
        crops: List[np.ndarray],
        positions: Optional[List[Tuple[float, float]]] = None,
    ) -> List[TeamAssignment]:
        return [self.predict(c, positions[i] if positions and i < len(positions) else None) for i, c in enumerate(crops)]

    def _is_in_penalty_box(self, position_xy: Tuple[float, float]) -> bool:
        return in_pitch1_goal_box(position_xy)

    def get_team_colors(self) -> Optional[Dict[int, np.ndarray]]:
        if self.team_centroids is None:
            return None
        return {0: self.team_centroids[0], 1: self.team_centroids[1]}

    def is_trained(self) -> bool:
        return self.team_centroids is not None
