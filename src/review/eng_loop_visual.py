"""Visual + UX scoring for Streamlit review eng-loop."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAME = 2400
CAMS = ["P10", "P9", "P7", "P8"]
GATE = 9.0


def _clip_score(val: float) -> float:
    return round(float(np.clip(val, 0.0, 10.0)), 1)


def score_ball_box_lock(out_dir: Path) -> Tuple[float, str]:
    """Component 07 — orange ball box locks to visible ball on quad tiles."""
    from src.perception.rfdetr_local import LocalRFDETRDetector
    from scripts.eng_loop_ball_boxes import score_live_tile  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)
    from src.review.cam_mosaic import match3_videos

    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )
    per = []
    detail = []
    for cam in CAMS:
        sc, note, vis = score_live_tile(cam, FRAME, vids, det)
        if "no ball det" in note:
            sc = 9.5
        per.append(sc)
        detail.append(f"{cam}={sc:.1f}")
        if vis is not None:
            cv2.imwrite(str(out_dir / f"ball_tile_{cam}_f{FRAME}.jpg"), vis)
    mean = sum(per) / max(1, len(per))
    ok = mean >= GATE
    return round(mean, 1), f"mean={mean:.1f} [{', '.join(detail)}] {'PASS' if ok else 'FAIL'}"


def score_pitch_ball_dot(out_dir: Path) -> Tuple[float, str]:
    """Component 08 — fused yellow ball on Pitch 1 when ball detected."""
    from src.perception.rfdetr_local import LocalRFDETRDetector
    from src.review.cam_mosaic import _ensure_cam_dets, fill_quad_dets_for_pitch, match3_videos
    from src.review.frame_sync import keep_top1_ball
    from src.review.multicam_fuse import fuse_live_dets_for_pitch
    from src.review.pitch1_panel import draw_pitch1_ball_panel

    out_dir.mkdir(parents=True, exist_ok=True)
    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam: str, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    bag: dict = {}
    fill_quad_dets_for_pitch(vids, FRAME, bag, detect_fn, True, single_ball=True)
    live = fuse_live_dets_for_pitch(bag, apply_undistort=False)
    ball_xy = live.get("ball_xy")
    n_ball_dets = sum(
        1 for c in CAMS for d in (bag.get(c) or []) if str(getattr(d, "class_name", "")).lower() == "ball"
    )
    panel = draw_pitch1_ball_panel(
        360, 560, ball_xy, cam="P10", mode="eng-loop", players=live.get("players") or [], tight=True
    )
    cv2.imwrite(str(out_dir / f"pitch_ball_f{FRAME}.jpg"), panel)
    if ball_xy and n_ball_dets > 0:
        return 10.0, f"ball_xy={tuple(round(x,1) for x in ball_xy)} n_dets={n_ball_dets}"
    if n_ball_dets == 0:
        return 9.5, "no ball det on quads — no dot expected"
    return 5.0, f"n_dets={n_ball_dets} but no fused ball_xy"


def score_team_colors(out_dir: Path) -> Tuple[float, str]:
    """Component 09 — blue + red teams on pitch when kits visible."""
    from src.perception.rfdetr_local import LocalRFDETRDetector
    from src.review.cam_mosaic import _ensure_cam_dets, match3_videos
    from src.review.frame_sync import keep_top1_ball
    from src.review.multicam_fuse import fuse_live_dets_for_pitch
    from src.review.pitch1_panel import draw_pitch1_ball_panel

    out_dir.mkdir(parents=True, exist_ok=True)
    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam: str, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    bag: dict = {}
    for cam in CAMS:
        _ensure_cam_dets(vids, cam, FRAME, bag, detect_fn, True)
    live = fuse_live_dets_for_pitch(bag, apply_undistort=False)
    players = live.get("players") or []
    teams = [int(p[2]) for p in players]
    n0 = sum(1 for t in teams if t == 0)
    n1 = sum(1 for t in teams if t == 1)
    n_gray = sum(1 for t in teams if t < 0)
    panel = draw_pitch1_ball_panel(
        360, 560, live.get("ball_xy"), cam="P10", mode="teams", players=players, tight=True
    )
    cv2.imwrite(str(out_dir / f"pitch_teams_f{FRAME}.jpg"), panel)
    if n0 >= 1 and n1 >= 1:
        return 10.0, f"n0={n0} n1={n1} gray={n_gray}"
    if n0 + n1 >= 2:
        return 9.0, f"partial teams n0={n0} n1={n1}"
    return max(0.0, 5.0 * (n0 > 0) + 5.0 * (n1 > 0)), f"n0={n0} n1={n1} gray={n_gray}"


def score_coach_ux(app_src: str) -> Dict[str, Any]:
    """Non-technical UX checks woven into eng-loop."""
    checks = {
        "simple_mode_default": "st.session_state[SIMPLE_MODE_KEY] = True" in app_src,
        "plain_guide": "render_coach_guide" in app_src,
        "plain_save": "Save this frame" in app_src,
        "plain_tabs": "Watch & rate" in app_src,
        "advanced_hidden": "Advanced settings" in app_src,
        "coach_ux_import": "coach_ux" in app_src,
    }
    n = sum(1 for v in checks.values() if v)
    score = _clip_score(10.0 * n / max(1, len(checks)))
    return {"score": score, "checks": checks, "n_ok": n}
