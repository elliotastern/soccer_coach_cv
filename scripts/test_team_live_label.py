#!/usr/bin/env python3
"""Unit tests for live team labeling (synthetic blue vs white kits)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import (  # noqa: E402
    KIT_MODE_MATCH3,
    TEAM_MIN_CROPS,
    assign_from_feature,
    fit_team_centroids,
    jersey_feature,
    torso_crop,
)
from src.review.team_live import TeamSession, label_player_pts  # noqa: E402


def _paint(bgr, h=60, w=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    yy, xx = np.mgrid[0:h, 0:w]
    tex = (((xx + yy) % 7) - 3).astype(np.int16)[:, :, None]
    return np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)


def test_torso_skips_legs() -> None:
    fr = np.zeros((200, 100, 3), dtype=np.uint8)
    fr[30:90, :] = (220, 80, 40)  # blue jersey BGR
    fr[90:, :] = (0, 255, 0)
    crop = torso_crop(fr, (10, 10, 40, 160))
    assert crop is not None


def test_blue_vs_white_split() -> None:
    blues = [_paint((220, 90, 40)) for _ in range(TEAM_MIN_CROPS)]
    whites = [_paint((230, 230, 230)) for _ in range(TEAM_MIN_CROPS)]
    feats = [jersey_feature(c) for c in blues + whites]
    feats = [f for f in feats if f is not None]
    fit = fit_team_centroids(feats)
    assert fit is not None
    cents, radius = fit
    labs = [assign_from_feature(f, cents, radius)[0] for f in feats]
    assert 0 in labs and 1 in labs
    # blues should mostly be team 0 (bluer lock)
    assert sum(1 for t in labs[:TEAM_MIN_CROPS] if t == 0) >= TEAM_MIN_CROPS - 1


def test_grass_rejected() -> None:
    assert jersey_feature(_paint((40, 200, 40))) is None


def test_label_player_pts() -> None:
    fr_a = np.zeros((200, 200, 3), dtype=np.uint8)
    fr_b = np.zeros((200, 200, 3), dtype=np.uint8)
    fr_a[30:70, 25:55] = (220, 90, 40)
    fr_b[30:70, 105:135] = (230, 230, 230)
    noise = np.random.RandomState(1).randint(0, 8, fr_a.shape, dtype=np.uint8)
    fr_a = np.clip(fr_a.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fr_b = np.clip(fr_b.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    pts = []
    for i in range(4):
        pts.append(
            {
                "xy": (0.0, float(i)),
                "team": -1,
                "pid": -1,
                "conf": 0.8,
                "cam": "P10",
                "bbox": (20.0, 20.0, 40.0, 100.0),
            }
        )
    for i in range(4):
        pts.append(
            {
                "xy": (5.0, float(i)),
                "team": -1,
                "pid": -1,
                "conf": 0.8,
                "cam": "P7",
                "bbox": (100.0, 20.0, 40.0, 100.0),
            }
        )
    label_player_pts(
        pts,
        {"P10": fr_a, "P7": fr_b},
        team_session=TeamSession(kit_mode=KIT_MODE_MATCH3),
    )
    teams = [p["team"] for p in pts]
    assert 0 in teams and 1 in teams


if __name__ == "__main__":
    test_torso_skips_legs()
    test_blue_vs_white_split()
    test_grass_rejected()
    test_label_player_pts()
    print("ok")
