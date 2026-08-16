#!/usr/bin/env python3
"""Locked multicam pick policies (per-region notes).

Top Left lock from selection loop: P7 must clear 0.60 to compete; other cams
stay at 0.30. Other pitch regions may need different per-cam floors — survey
before copying this rule match-wide.
"""
from __future__ import annotations

# Practical HIT on Top Left dual-gold (188 covered, P=R=0.915).
TOP_LEFT_THR_BY_CAM = {
    "_default": 0.30,
    "P7": 0.60,
}
TOP_LEFT_POLICY_ID = "p7_thr060_others030"
TOP_LEFT_POLICY_NOTE = (
    "Top Left only: P7≥0.60, others≥0.30, then max_conf. "
    "Do not assume this thr map for Center/Bottom/Top Right without a survey."
)

BASELINE_THR = 0.30
GOAL_R = 0.80
GOAL_P = 0.90

P_CAMS = ["P1", "P6", "P7", "P8", "P10", "P12"]
# Other quads often pick wide cams in the 4quad emit table.
SURVEY_CAMS = P_CAMS + ["Cam4plus", "Cam5plus"]

QUAD_SLOTS = [
    {
        "slot": "top_left",
        "label": "Top Left",
        "stem": "quad_top_left_t00026.0s",
        "n_frames": 299,
        "locked_thr": TOP_LEFT_THR_BY_CAM,
    },
    {
        "slot": "top_right",
        "label": "Top Right",
        "stem": "quad_top_right_t00125.0s",
        "n_frames": 299,
        "locked_thr": None,  # unknown until surveyed
    },
    {
        "slot": "center_start",
        "label": "Center Start",
        "stem": "quad_center_start_t00008.0s",
        "n_frames": 300,
        "locked_thr": None,
    },
    {
        "slot": "bottom_right",
        "label": "Bottom Right",
        "stem": "quad_bottom_right_t00412.0s",
        "n_frames": 300,  # score first 300 of ~359
        "locked_thr": None,
    },
]


def thr_for_cam(thr_by_cam: dict, cam: str) -> float:
    return float(thr_by_cam.get(cam, thr_by_cam.get("_default", BASELINE_THR)))
