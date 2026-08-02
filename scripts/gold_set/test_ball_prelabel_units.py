#!/usr/bin/env python3
"""Unit tests for ball prelabel helpers (no model weights required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import (
    KalmanBallTracker,
    Detection,
    filter_ball_geometry,
    iou_xywh,
    nms_balls,
    sahi_recover_only,
    slice_grid,
    topk_balls,
)


def test_slice_grid_covers_image():
    tiles = slice_grid(1920, 1080, slice_size=640, overlap=0.2)
    assert len(tiles) >= 6
    # last tile reaches right/bottom edge
    assert any(t[2] == 1920 for t in tiles)
    assert any(t[3] == 1080 for t in tiles)


def test_nms_keeps_highest():
    a = Detection(1, 0.9, (10, 10, 20, 20), "ball")
    b = Detection(1, 0.5, (12, 12, 20, 20), "ball")
    c = Detection(1, 0.8, (200, 200, 20, 20), "ball")
    kept = nms_balls([a, b, c], iou_thr=0.3)
    assert len(kept) == 2
    assert kept[0].confidence == 0.9


def test_size_filter_and_topk():
    dets = [
        Detection(1, 0.9, (0, 0, 2, 2), "ball"),  # too small
        Detection(1, 0.8, (0, 0, 200, 200), "ball"),  # too big
        Detection(1, 0.7, (0, 0, 16, 16), "ball"),
        Detection(1, 0.6, (50, 50, 18, 18), "ball"),
    ]
    filtered = filter_ball_geometry(dets, min_side=4, max_side=80)
    assert len(filtered) == 2
    assert len(topk_balls(filtered, 1)) == 1


def test_kalman_coasts_then_updates():
    kf = KalmanBallTracker(max_coast=5, gate_px=80)
    # Need 2 hits before coast is allowed
    d1 = Detection(1, 0.7, (100, 100, 12, 12), "ball")
    d2 = Detection(1, 0.7, (104, 102, 12, 12), "ball")
    assert len(kf.step([d1])) == 1
    assert len(kf.step([d2])) == 1
    coast = kf.step([])
    assert len(coast) == 1
    assert coast[0].confidence < d2.confidence
    far = Detection(1, 0.9, (900, 900, 12, 12), "ball")
    moved = kf.step([far])
    assert len(moved) == 1


def test_iou():
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xywh((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_sahi_recover_only():
    full = [Detection(1, 0.8, (10, 10, 20, 20), "ball")]
    tiles = [
        Detection(1, 0.7, (12, 12, 20, 20), "ball"),
        Detection(1, 0.7, (300, 300, 16, 16), "ball"),
    ]
    got = sahi_recover_only(full, tiles, max_iou=0.1)
    assert len(got) == 1 and got[0].bbox[0] == 300


def main():
    test_slice_grid_covers_image()
    test_nms_keeps_highest()
    test_size_filter_and_topk()
    test_kalman_coasts_then_updates()
    test_iou()
    test_sahi_recover_only()
    print("PASS: ball_prelabel unit tests (6/6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
