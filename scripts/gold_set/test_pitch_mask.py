#!/usr/bin/env python3
"""Unit checks for on-pitch / sideline gate (no model)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.pitch_mask import on_pitch_bbox  # noqa: E402


def _green_frame(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 140, 40)  # BGR-ish green turf
    return img


def _brown_frame(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 60, 120)  # brown track
    return img


def test_green_keeps_ball():
    fr = _green_frame()
    assert on_pitch_bbox(fr, (100, 80, 20, 20)) is True


def test_brown_rejects_sideline():
    fr = _brown_frame()
    assert on_pitch_bbox(fr, (100, 80, 20, 20)) is False


def test_mixed_prefers_turf_side():
    fr = _brown_frame()
    fr[:, 150:] = (40, 140, 40)  # right half turf
    assert on_pitch_bbox(fr, (200, 80, 20, 20)) is True
    assert on_pitch_bbox(fr, (40, 80, 20, 20)) is False


if __name__ == "__main__":
    test_green_keeps_ball()
    test_brown_rejects_sideline()
    test_mixed_prefers_turf_side()
    print("ok")
