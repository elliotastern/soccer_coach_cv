#!/usr/bin/env python3
"""Contract: mutual-nearest color soft merge (anti-teammate)."""
from __future__ import annotations

from src.review.multicam_fuse import PLAYER_MERGE_SOFT_M_LIVE, _cluster_players


def test_mutual_soft_merges_reciprocal_pair():
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (4.0, 0.0), "conf": 0.8, "cam": "P7", "team": 0},
    ]
    soft = _cluster_players(
        pts, merge_m=2.2, soft_m=PLAYER_MERGE_SOFT_M_LIVE, color_gate=True
    )
    assert len(soft) == 1
    assert PLAYER_MERGE_SOFT_M_LIVE >= 4.0


def test_mutual_soft_skips_non_reciprocal_teammate_trap():
    # A nearest to B, but B nearer to C (same team) → no A-B merge; B-C merge
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (4.0, 0.0), "conf": 0.85, "cam": "P7", "team": 0},
        {"xy": (6.5, 0.0), "conf": 0.8, "cam": "P8", "team": 0},
    ]
    soft = _cluster_players(pts, merge_m=2.2, soft_m=4.5, color_gate=True)
    assert len(soft) == 2


def test_mutual_soft_skips_diff_team():
    pts = [
        {"xy": (0.0, 0.0), "conf": 0.9, "cam": "P10", "team": 0},
        {"xy": (4.0, 0.0), "conf": 0.8, "cam": "P7", "team": 1},
    ]
    soft = _cluster_players(pts, merge_m=2.2, soft_m=4.5, color_gate=True)
    assert len(soft) == 2


if __name__ == "__main__":
    test_mutual_soft_merges_reciprocal_pair()
    test_mutual_soft_skips_non_reciprocal_teammate_trap()
    test_mutual_soft_skips_diff_team()
    print("ok")
