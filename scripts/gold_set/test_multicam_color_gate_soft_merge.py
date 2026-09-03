#!/usr/bin/env python3
"""Contract: color-gated soft merge clusters same-team near-misses only."""
from __future__ import annotations

from src.review.multicam_fuse import PLAYER_MERGE_SOFT_M_LIVE, _cluster_players


def test_soft_merge_same_team_cross_cam():
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (2.8, 0.0), "conf": 0.8, "cam": "P7", "team": 0},
    ]
    hard = _cluster_players(pts, merge_m=2.2)
    soft = _cluster_players(
        pts, merge_m=2.2, soft_m=PLAYER_MERGE_SOFT_M_LIVE, color_gate=True
    )
    assert len(hard) == 2
    assert len(soft) == 1


def test_soft_merge_skips_diff_team():
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (2.8, 0.0), "conf": 0.8, "cam": "P7", "team": 1},
    ]
    soft = _cluster_players(pts, merge_m=2.2, soft_m=3.2, color_gate=True)
    assert len(soft) == 2


def test_soft_merge_skips_same_cam():
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (2.8, 0.0), "conf": 0.8, "cam": "P10", "team": 0},
    ]
    soft = _cluster_players(pts, merge_m=2.2, soft_m=3.2, color_gate=True)
    assert len(soft) == 2


if __name__ == "__main__":
    test_soft_merge_same_team_cross_cam()
    test_soft_merge_skips_diff_team()
    test_soft_merge_skips_same_cam()
    print("ok")
