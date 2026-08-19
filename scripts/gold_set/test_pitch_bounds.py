#!/usr/bin/env python3
"""Unit tests for in_pitch_bounds (Pitch 1 measured size)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.pitch_bounds import in_pitch_bounds  # noqa: E402


def test_center_in():
    assert in_pitch_bounds(0.0, 0.0) is True


def test_corner_in():
    assert in_pitch_bounds(26.95, 17.42) is True
    assert in_pitch_bounds(-26.95, -17.42) is True


def test_just_outside():
    assert in_pitch_bounds(27.0, 0.0) is False
    assert in_pitch_bounds(0.0, 17.5) is False


def test_margin_allows_sideline():
    assert in_pitch_bounds(0.0, 18.0, margin_m=1.0) is True
    assert in_pitch_bounds(0.0, 19.0, margin_m=1.0) is False


def test_track_ball_rejected():
    assert in_pitch_bounds(10.0, 40.0, margin_m=0.5) is False


if __name__ == "__main__":
    test_center_in()
    test_corner_in()
    test_just_outside()
    test_margin_allows_sideline()
    test_track_ball_rejected()
    print("ok")
