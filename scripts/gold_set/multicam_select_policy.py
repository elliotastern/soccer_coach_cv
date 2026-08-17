#!/usr/bin/env python3
"""Locked multicam pick policies (product + per-region notes).

Product lock (`pool8_largest_ball_p7_thr060` / select mode `locked`):
  pool = 6 P-cams + Cam4plus + Cam5plus
  thr  = P7≥0.60, others≥0.30
  pick = largest ball side, then conf

Use `pick_product(pred_map)` from live/batch paths. Top Left gold HIT;
other 4quad regions size-OK in `4quad_locked_policy_survey/`.
"""
from __future__ import annotations

# Practical HIT on Top Left dual-gold under P-cam-only max_conf (188 covered).
TOP_LEFT_THR_BY_CAM = {
    "_default": 0.30,
    "P7": 0.60,
}
TOP_LEFT_PICK_MODE = "largest_ball"
TOP_LEFT_POLICY_ID = "pool8_largest_ball_p7_thr060"
TOP_LEFT_POLICY_NOTE = (
    "Product lock (Match 2 8-cam): Cam4+/Cam5+/P-cams, P7≥0.60 others≥0.30, "
    "largest_ball then conf. Top Left gold HIT; other 4quads size-OK in survey. "
    "Wire via pick_product() / select mode `locked`."
)

# Aliases used by live/batch path
PRODUCT_POLICY_ID = TOP_LEFT_POLICY_ID
PRODUCT_THR_BY_CAM = TOP_LEFT_THR_BY_CAM
PRODUCT_PICK_MODE = TOP_LEFT_PICK_MODE

# Prior P-cam-only lock (kept for comparison / rollback).
TOP_LEFT_PCAM_ONLY_POLICY_ID = "p7_thr060_others030"

BASELINE_THR = 0.30
GOAL_R = 0.80
GOAL_P = 0.90

P_CAMS = ["P1", "P6", "P7", "P8", "P10", "P12"]
SURVEY_CAMS = P_CAMS + ["Cam4plus", "Cam5plus"]
TOP_LEFT_POOL = list(SURVEY_CAMS)
PRODUCT_POOL = list(TOP_LEFT_POOL)

QUAD_SLOTS = [
    {
        "slot": "top_left",
        "label": "Top Left",
        "stem": "quad_top_left_t00026.0s",
        "n_frames": 299,
        "locked_thr": TOP_LEFT_THR_BY_CAM,
        "locked_pick": TOP_LEFT_PICK_MODE,
        "locked_pool": TOP_LEFT_POOL,
    },
    {
        "slot": "top_right",
        "label": "Top Right",
        "stem": "quad_top_right_t00125.0s",
        "n_frames": 299,
        "locked_thr": None,
        "locked_pick": None,
        "locked_pool": None,
    },
    {
        "slot": "center_start",
        "label": "Center Start",
        "stem": "quad_center_start_t00008.0s",
        "n_frames": 300,
        "locked_thr": None,
        "locked_pick": None,
        "locked_pool": None,
    },
    {
        "slot": "bottom_right",
        "label": "Bottom Right",
        "stem": "quad_bottom_right_t00412.0s",
        "n_frames": 300,
        "locked_thr": None,
        "locked_pick": None,
        "locked_pool": None,
    },
]


def thr_for_cam(thr_by_cam: dict, cam: str) -> float:
    return float(thr_by_cam.get(cam, thr_by_cam.get("_default", BASELINE_THR)))


def filter_active(dets: dict, frame_i: int, cams: list, thr_by_cam: dict) -> dict:
    out = {}
    for cam in cams:
        if cam not in dets:
            continue
        thr = thr_for_cam(thr_by_cam, cam)
        rows = [
            row
            for row in dets[cam][frame_i]
            if float(row[1]) >= thr
        ]
        rows = sorted(rows, key=lambda r: -float(r[1]))[:2]
        if rows:
            out[cam] = rows
    return out


def locked_top_left_spec() -> dict:
    return {
        "id": TOP_LEFT_POLICY_ID,
        "pool": list(TOP_LEFT_POOL),
        "thr_by_cam": dict(TOP_LEFT_THR_BY_CAM),
        "pick_mode": TOP_LEFT_PICK_MODE,
        "note": TOP_LEFT_POLICY_NOTE,
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
    }


def filter_pred_map(pred_map: dict, thr_by_cam: dict | None = None) -> dict:
    """Drop cams whose best conf is below per-cam floor. preds = list of (box, conf, side)."""
    thr_map = thr_by_cam if thr_by_cam is not None else PRODUCT_THR_BY_CAM
    out = {}
    for cam, preds in pred_map.items():
        if not preds:
            continue
        thr = thr_for_cam(thr_map, cam)
        kept = [p for p in preds if float(p[1]) >= thr]
        if kept:
            out[cam] = kept
    return out


def pick_product(pred_map: dict, mode: str | None = None, thr_by_cam: dict | None = None):
    """Product multicam pick: thr floors then largest_ball (or override mode)."""
    from eval_match2_v10_video_system import pick_selected

    pick_mode = PRODUCT_PICK_MODE if mode in (None, "locked") else mode
    active = filter_pred_map(pred_map, thr_by_cam)
    return pick_selected(active, pick_mode)
