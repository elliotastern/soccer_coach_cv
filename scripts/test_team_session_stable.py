#!/usr/bin/env python3
"""Unit tests for TeamSession stability (cheap team-ID lift)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import (  # noqa: E402
    FEAT_BASE,
    HUE_BINS,
    TEAM_MIN_CROPS,
    fit_team_centroids,
    jersey_feature,
    which_goal_box,
)
from src.review.team_live import (  # noqa: E402
    TeamSession,
    apply_goal_box_prior,
)
from src.perception import team_core as _tc  # noqa: E402

_adaptive_non_green = _tc._adaptive_non_green


def _paint(bgr, h=60, w=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    yy, xx = np.mgrid[0:h, 0:w]
    tex = (((xx + yy) % 7) - 3).astype(np.int16)[:, :, None]
    return np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)


def test_fit_order_stable_under_reshuffle() -> None:
    blues = [jersey_feature(_paint((220, 90, 40))) for _ in range(TEAM_MIN_CROPS)]
    whites = [jersey_feature(_paint((230, 230, 230))) for _ in range(TEAM_MIN_CROPS)]
    feats = [f for f in blues + whites if f is not None]
    fit_a = fit_team_centroids(feats)
    fit_b = fit_team_centroids(list(reversed(feats)))
    assert fit_a is not None and fit_b is not None
    assert fit_a[0][0, 0] >= fit_a[0][1, 0] - 1e-5
    assert fit_b[0][0, 0] >= fit_b[0][1, 0] - 1e-5


def test_session_no_centroid_swap() -> None:
    sess = TeamSession()
    blues = [jersey_feature(_paint((220, 90, 40))) for _ in range(TEAM_MIN_CROPS)]
    whites = [jersey_feature(_paint((230, 230, 230))) for _ in range(TEAM_MIN_CROPS)]
    feats = [f for f in blues + whites if f is not None]
    fit = fit_team_centroids(feats)
    assert fit is not None
    sess.centroids, sess.radius = fit
    c0_before = sess.centroids[0].copy()
    sess._ema(feats[:TEAM_MIN_CROPS], [0] * TEAM_MIN_CROPS)
    assert float(np.linalg.norm(sess.centroids[0] - c0_before)) < 1.0
    assert float(sess.centroids[0, 0]) >= float(sess.centroids[1, 0]) - 0.05


def test_sticky_keeps_prior() -> None:
    sess = TeamSession()
    sess.prev = [{"xy": (0.0, 0.0), "team": 0, "conf": 0.9}]
    pts = [{"xy": (0.5, 0.2), "team": 1, "team_conf": 0.40}]
    sess._sticky(pts)
    assert pts[0]["team"] == 0


def test_stabilize_fused_hard_sticky_and_hold() -> None:
    sess = TeamSession()
    sess.prev_fused = [{"xy": (1.0, 1.0), "team": 0, "pid": 1, "age": 0, "votes": [0, 0]}]
    # Gray observation near prior → gray-fill keeps 0
    out = sess.stabilize_fused([(1.2, 1.1, -1, 2)])
    assert out[0][2] == 0
    out2 = sess.stabilize_fused([])
    assert len(out2) == 1
    assert out2[0][2] == 0


def test_adaptive_mask_drops_green() -> None:
    import cv2

    grass = _paint((40, 180, 40), h=40, w=40)
    hsv = cv2.cvtColor(grass, cv2.COLOR_BGR2HSV)
    keep = _adaptive_non_green(hsv)
    assert float(keep.mean()) < 0.15
    assert jersey_feature(grass) is None


def test_hist_separates_blue_white() -> None:
    fb = jersey_feature(_paint((220, 90, 40)))
    fw = jersey_feature(_paint((230, 230, 230)))
    assert fb is not None and fw is not None
    assert fb.shape[0] == FEAT_BASE + HUE_BINS
    assert fw.shape[0] == FEAT_BASE + HUE_BINS
    assert float(fb[0]) > float(fw[0])
    assert float(fw[1]) > float(fb[1])


def test_vote_buffer_overrides_one_frame_flip() -> None:
    sess = TeamSession()
    sess.prev_fused = [
        {"xy": (0.0, 0.0), "team": 0, "pid": 1, "age": 0, "votes": [0, 0, 0]}
    ]
    # One opposite observation — vote majority keeps team 0
    out = sess.stabilize_fused([(0.3, 0.2, 1, 2)])
    assert out[0][2] == 0


def test_goal_box_prior_alone_keeps_clear() -> None:
    # South box roughly x ~ -26.95..-21, y within box width
    box = which_goal_box((-24.0, 0.0))
    assert box == "south"
    p = {"xy": (-24.0, 0.0), "team": 0}
    apply_goal_box_prior([p])
    assert p["team"] == 0


def test_goal_box_prior_conflict_grays() -> None:
    a = {"xy": (-24.0, 0.5), "team": 0}
    b = {"xy": (-24.0, -0.5), "team": 1}
    apply_goal_box_prior([a, b])
    assert a["team"] == -1 and b["team"] == -1


def test_no_hold_inside_goal_box() -> None:
    sess = TeamSession()
    # Prior in south goal box — must not hold when unmatched
    sess.prev_fused = [
        {"xy": (-24.0, 0.0), "team": 0, "pid": 1, "age": 0, "votes": [0, 0]}
    ]
    out = sess.stabilize_fused([])
    assert out == []


def test_soft_cap_goal_box_near_duplicates() -> None:
    from src.review.team_live import soft_cap_goal_box_duplicates

    # Three near-duplicates in south box + one midfield
    pts = [
        {"xy": (-24.0, 0.0), "team": 0, "pid": 1, "age": 0, "conf": 0.9},
        {"xy": (-23.5, 0.3), "team": 0, "pid": 2, "age": 0, "conf": 0.7},
        {"xy": (-23.2, -0.2), "team": 1, "pid": 3, "age": 1, "conf": 0.6},
        {"xy": (0.0, 0.0), "team": 0, "pid": 4, "age": 0, "conf": 0.8},
    ]
    out = soft_cap_goal_box_duplicates(pts)
    box_n = sum(1 for p in out if which_goal_box(p["xy"]) == "south")
    assert box_n <= 2
    assert any(which_goal_box(p["xy"]) is None for p in out)


if __name__ == "__main__":
    test_fit_order_stable_under_reshuffle()
    test_session_no_centroid_swap()
    test_sticky_keeps_prior()
    test_stabilize_fused_hard_sticky_and_hold()
    test_adaptive_mask_drops_green()
    test_hist_separates_blue_white()
    test_vote_buffer_overrides_one_frame_flip()
    test_goal_box_prior_alone_keeps_clear()
    test_goal_box_prior_conflict_grays()
    test_no_hold_inside_goal_box()
    test_soft_cap_goal_box_near_duplicates()
    print("ok")
