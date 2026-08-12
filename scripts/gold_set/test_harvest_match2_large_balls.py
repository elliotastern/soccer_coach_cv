#!/usr/bin/env python3
"""Unit tests for large-ball harvest filters."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.state.types import Detection
from scripts.gold_set.harvest_match2_large_balls import (
    ball_side,
    pick_large_ball,
    too_static,
)


def test_pick_rejects_small():
    small = Detection(class_id=1, confidence=0.9, bbox=(0, 0, 20, 20), class_name="ball")
    assert pick_large_ball([small], min_side=40, min_conf=0.35) is None


def test_pick_keeps_large():
    large = Detection(class_id=1, confidence=0.5, bbox=(10, 10, 70, 72), class_name="ball")
    got = pick_large_ball([large], min_side=40, min_conf=0.35)
    assert got is large
    assert ball_side(got) >= 40


def test_pick_prefers_conf():
    a = Detection(class_id=1, confidence=0.4, bbox=(0, 0, 50, 50), class_name="ball")
    b = Detection(class_id=1, confidence=0.7, bbox=(0, 0, 45, 45), class_name="ball")
    got = pick_large_ball([a, b], min_side=40, min_conf=0.35)
    assert got is b


def test_too_static_caps_spot():
    det = Detection(class_id=1, confidence=0.5, bbox=(100, 100, 30, 30), class_name="ball")
    kept = [{"detections": [det]} for _ in range(2)]
    assert too_static(115, 115, kept, radius=80, max_per_spot=2) is True
    assert too_static(400, 400, kept, radius=80, max_per_spot=2) is False


if __name__ == "__main__":
    test_pick_rejects_small()
    test_pick_keeps_large()
    test_pick_prefers_conf()
    test_too_static_caps_spot()
    print("ok")
