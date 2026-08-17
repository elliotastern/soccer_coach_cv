#!/usr/bin/env python3
"""Unit tests for in_pitch_bounds."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.pitch_bounds import in_pitch_bounds  # noqa: E402


def test_center_in():
    assert in_pitch_bounds(0.0, 0.0) is True


def test_corner_in():
    assert in_pitch_bounds(52.5, 34.0) is True
    assert in_pitch_bounds(-52.5, -34.0) is True


def test_just_outside():
    assert in_pitch_bounds(52.6, 0.0) is False
    assert in_pitch_bounds(0.0, 34.1) is False


def test_margin_allows_sideline():
    assert in_pitch_bounds(0.0, 34.5, margin_m=1.0) is True
    assert in_pitch_bounds(0.0, 35.5, margin_m=1.0) is False


def test_track_ball_rejected():
    # Spare ball on brown track beyond touchline
    assert in_pitch_bounds(10.0, 40.0, margin_m=0.5) is False


if __name__ == "__main__":
    test_center_in()
    test_corner_in()
    test_just_outside()
    test_margin_allows_sideline()
    test_track_ball_rejected()
    print("ok")
