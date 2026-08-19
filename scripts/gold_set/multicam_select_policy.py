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

# Match 3 pitch fuse (Pitch 1): do not inherit Match 2 P7≥0.60.
# Detect floor 0.20 all cams (M1 strip A/B: unlocks weak dual-maps; emit stays ≥0.80).
MATCH3_THR_BY_CAM = {
    "_default": 0.20,
}
MATCH3_POLICY_ID = "match3_all_cam_thr020"
MATCH3_POLICY_NOTE = (
    "Match 3 multicam pitch fuse: all cams ≥0.20 (no P7@0.60). "
    "Emit still requires fuse conf ≥0.80. Match 2 product lock unchanged."
)

# Soft gates (MATCH2_NOISE_PRECISION_PLAN): cam hysteresis + emit stickiness
HYSTERESIS_K = 5  # challenger must win K frames to steal cam
EMIT_N = 3  # same cam must win N frames before emit/plot
SIDE_MARGIN_PX = 2.0  # challenger side must beat incumbent by this
CONF_MARGIN = 0.05

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


def pick_product(
    pred_map: dict,
    mode: str | None = None,
    thr_by_cam: dict | None = None,
    frames_by_cam: dict | None = None,
):
    """Product multicam pick: thr floors, optional on-pitch gate, then largest_ball."""
    from eval_match2_v10_video_system import pick_selected

    pick_mode = PRODUCT_PICK_MODE if mode in (None, "locked") else mode
    active = filter_pred_map(pred_map, thr_by_cam)
    if frames_by_cam:
        from src.perception.pitch_mask import filter_pred_map_on_pitch

        active = filter_pred_map_on_pitch(frames_by_cam, active)
    return pick_selected(active, pick_mode)


def challenger_beats(incumbent_pred, chall_pred, side_margin=SIDE_MARGIN_PX, conf_margin=CONF_MARGIN):
    """True if challenger is clearly better than incumbent (side then conf)."""
    if incumbent_pred is None:
        return True
    if chall_pred is None:
        return False
    i_side, i_conf = float(incumbent_pred[2]), float(incumbent_pred[1])
    c_side, c_conf = float(chall_pred[2]), float(chall_pred[1])
    if c_side >= i_side + side_margin:
        return True
    if c_side + 1e-6 >= i_side and c_conf >= i_conf + conf_margin:
        return True
    return False


class StickyCamPicker:
    """Cam hysteresis (K) + emit stickiness (N) around raw pick_product."""

    def __init__(self, k: int = HYSTERESIS_K, n: int = EMIT_N):
        self.k = int(k)
        self.n = int(n)
        self.cam = None
        self.pred = None
        self.challenger = None
        self.challenger_streak = 0
        self.hold_streak = 0

    def step(self, raw_cam, raw_pred):
        """Apply hysteresis. Returns (held_cam, held_pred)."""
        if raw_cam is None or raw_pred is None:
            self.challenger = None
            self.challenger_streak = 0
            self.hold_streak = 0
            self.cam, self.pred = None, None
            return None, None

        if self.cam is None:
            self.cam, self.pred = raw_cam, raw_pred
            self.hold_streak = 1
            self.challenger = None
            self.challenger_streak = 0
            return self.cam, self.pred

        if raw_cam == self.cam:
            self.pred = raw_pred
            self.hold_streak += 1
            self.challenger = None
            self.challenger_streak = 0
            return self.cam, self.pred

        # Different cam proposing
        if not challenger_beats(self.pred, raw_pred):
            self.hold_streak += 1
            self.challenger = None
            self.challenger_streak = 0
            return self.cam, self.pred

        if self.challenger != raw_cam:
            self.challenger = raw_cam
            self.challenger_streak = 1
        else:
            self.challenger_streak += 1

        if self.challenger_streak >= self.k:
            self.cam, self.pred = raw_cam, raw_pred
            self.hold_streak = 1
            self.challenger = None
            self.challenger_streak = 0
        else:
            self.hold_streak += 1
        return self.cam, self.pred

    def emit(self, cam, pred):
        """Gate emit/plot until same cam held for N frames."""
        if cam is None or pred is None:
            return None, None
        if self.hold_streak < self.n:
            return None, None
        return cam, pred


def pick_product_sticky(
    pred_map: dict,
    state: StickyCamPicker,
    mode: str | None = None,
    thr_by_cam: dict | None = None,
    frames_by_cam: dict | None = None,
    apply_emit_gate: bool = True,
):
    """pick_product + hysteresis (+ optional N-frame emit gate)."""
    raw_cam, raw_pred = pick_product(
        pred_map, mode=mode, thr_by_cam=thr_by_cam, frames_by_cam=frames_by_cam
    )
    cam, pred = state.step(raw_cam, raw_pred)
    if apply_emit_gate:
        return state.emit(cam, pred)
    return cam, pred
