#!/usr/bin/env python3
"""Unit tests for live player pitch fuse (max-conf + gates)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.multicam_fuse import (  # noqa: E402
    PLAYER_MIN_CONF,
    PLAYER_SOLO_CONF,
    _cluster_players,
    _fuse_player_clusters,
    player_det_ok,
)


def test_player_det_ok_rejects_tiny_and_weak() -> None:
    weak = SimpleNamespace(class_name="player", class_id=0, confidence=0.2, bbox=(10, 10, 40, 80))
    tiny = SimpleNamespace(
        class_name="player", class_id=0, confidence=0.9, bbox=(10, 10, 10, 20)
    )
    ok = SimpleNamespace(
        class_name="player",
        class_id=0,
        confidence=max(PLAYER_MIN_CONF, 0.7),
        bbox=(10, 10, 40, 90),
    )
    assert player_det_ok(weak) is False
    assert player_det_ok(tiny) is False
    assert player_det_ok(ok) is True


def test_fuse_uses_max_conf_xy_not_mean() -> None:
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "team": 0, "pid": 1, "cam": "P10"},
        {"xy": (1.0, 0.0), "conf": 0.55, "team": 0, "pid": 2, "cam": "P7"},
    ]
    fused = _fuse_player_clusters(_cluster_players(pts, merge_m=1.8))
    assert len(fused) == 1
    assert fused[0][0] == 0.0 and fused[0][1] == 0.0


def test_weak_solo_dropped() -> None:
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "team": -1, "pid": -1, "cam": "P10"},
        {
            "xy": (20.0, 0.0),
            "conf": min(PLAYER_SOLO_CONF - 0.05, 0.5),
            "team": -1,
            "pid": -1,
            "cam": "P8",
        },
    ]
    fused = _fuse_player_clusters(_cluster_players(pts, merge_m=1.8))
    assert len(fused) == 1
    assert abs(fused[0][0]) < 0.01


def test_goal_box_wider_merge() -> None:
    # South box feet 2.5 m apart — base 1.8 would split; box merge 3.2 should join
    pts = [
        {"xy": (-24.0, 0.0), "conf": 0.9, "team": 0, "pid": 1, "cam": "P7"},
        {"xy": (-24.0, 2.5), "conf": 0.8, "team": 0, "pid": 2, "cam": "P8"},
    ]
    clusters = _cluster_players(pts, merge_m=1.8)
    assert len(clusters) == 1
    fused = _fuse_player_clusters(clusters)
    assert len(fused) == 1


if __name__ == "__main__":
    test_player_det_ok_rejects_tiny_and_weak()
    test_fuse_uses_max_conf_xy_not_mean()
    test_weak_solo_dropped()
    test_goal_box_wider_merge()
    print("ok")
