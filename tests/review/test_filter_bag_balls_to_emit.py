"""Tests for fuse-backed ball overlay filter."""
from __future__ import annotations

from types import SimpleNamespace

from src.review.multicam_fuse import filter_bag_balls_to_emit, _is_ball_det


def _ball(conf=0.9, bbox=(10.0, 10.0, 20.0, 20.0)):
    return SimpleNamespace(class_name="ball", class_id=1, confidence=conf, bbox=bbox)


def _player():
    return SimpleNamespace(class_name="player", class_id=0, confidence=0.9, bbox=(0, 0, 40, 80))


def test_no_emit_strips_all_balls():
    bag = {"P10": [_player(), _ball()], "P10__wh": (1920, 1080)}
    out = filter_bag_balls_to_emit(bag, None, apply_undistort=False)
    assert len([d for d in out["P10"] if _is_ball_det(d)]) == 0
    assert len([d for d in out["P10"] if not _is_ball_det(d)]) == 1


def test_emit_cam_keeps_ball_without_calib_map():
    # Unknown cam id → no calib; still keep when cam matches emit.
    bag = {"NOCAL": [_ball(0.95)], "NOCAL__wh": (100, 100)}
    meta = {"xy": (1.0, 2.0), "cam": "NOCAL", "agree": False, "conf": 0.95}
    out = filter_bag_balls_to_emit(bag, meta, apply_undistort=False)
    assert len([d for d in out["NOCAL"] if _is_ball_det(d)]) == 1
