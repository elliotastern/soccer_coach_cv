#!/usr/bin/env python3
"""Unit tests for tracklet team model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import TEAM_MIN_CROPS, assign_feature, jersey_feature
from src.perception.team_tracklet import TrackletAccumulator, TrackletTeamModel


def _paint(bgr, h=80, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    noise = np.random.RandomState(0).randint(0, 8, img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_tracklet_fit_and_label() -> None:
    acc = TrackletAccumulator()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(TEAM_MIN_CROPS):
        crop = _paint((220, 90, 40))
        frame[30:110, 20:70] = crop
        acc.add(i, frame, (20.0, 30.0, 50.0, 80.0))
    for i in range(TEAM_MIN_CROPS):
        crop = _paint((230, 230, 230))
        frame[30:110, 100:150] = crop
        acc.add(100 + i, frame, (100.0, 30.0, 50.0, 80.0))
    model = TrackletTeamModel()
    assert model.fit_from_accumulator(acc, min_tracklets=10)
    assert model.centroids is not None
    for tid, (lab, _conf) in model.track_labels.items():
        feat = acc.tracklet_medians()[tid]
        pred, _ = assign_feature(feat, model.centroids, model.radius)
        assert pred == lab


if __name__ == "__main__":
    test_tracklet_fit_and_label()
    print("ok")
