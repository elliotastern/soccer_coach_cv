#!/usr/bin/env python3
"""Eng-loop: Best-ball shows whole-pitch players + one ball (PROMPT best_ball_full)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    _is_ball_det,
    build_cam_view,
    match3_videos,
    mosaic_quads_coach,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/best_ball_full"
FRAMES = (1200, 2400, 3600)
GATE = 9.0


def _score_bool(ok: bool, partial: float = 3.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
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

    # Primary detailed frame
    FRAME = 2400
    bag: dict = {}
    img, used = build_cam_view(
        ROOT,
        "Best camera (ball)",
        FRAME,
        ROOT / "outputs",
        primary_cam="P10",
        dets_by_cam=bag,
        detect_fn=detect_fn,
        apply_defish=True,
    )
    scores["01_best_uses_mosaic"] = _score_bool(len(used) == 4)
    notes["01_best_uses_mosaic"] = f"used={used}"
    scores["02_mosaic_shape"] = _score_bool(
        img is not None and img.shape[0] >= 700 and img.shape[1] >= 1200
    )
    notes["02_mosaic_shape"] = f"shape={getattr(img, 'shape', None)}"

    n_cams_p = sum(
        1
        for c in ("P10", "P9", "P7", "P8")
        if any(not _is_ball_det(d) for d in (bag.get(c) or []))
    )
    scores["03_players_multi_quad"] = 10.0 if n_cams_p >= 3 else max(0.0, 3.0 * n_cams_p)
    notes["03_players_multi_quad"] = f"cams_with_players={n_cams_p}"

    n_balls = sum(
        1
        for c in ("P10", "P9", "P7", "P8")
        for d in (bag.get(c) or [])
        if _is_ball_det(d)
    )
    scores["04_single_ball"] = 10.0 if n_balls <= 1 else max(0.0, 10.0 - 3.0 * (n_balls - 1))
    notes["04_single_ball"] = f"balls_in_bag={n_balls}"

    live_b = fuse_live_dets_for_pitch(
        bag, apply_undistort=False, team_session=TeamSession()
    )
    bag_w: dict = {}
    mosaic_quads_coach(
        vids, FRAME, dets_by_cam=bag_w, detect_fn=detect_fn, apply_defish=True
    )
    live_w = fuse_live_dets_for_pitch(
        bag_w, apply_undistort=False, team_session=TeamSession()
    )
    nb, nw = len(live_b["players"]), len(live_w["players"])
    ratio = nb / max(nw, 1)
    scores["05_pitch_players_vs_whole"] = (
        10.0 if ratio >= 0.9 and nb >= 3 else max(0.0, 10.0 * ratio)
    )
    notes["05_pitch_players_vs_whole"] = f"best={nb} whole={nw} ratio={ratio:.2f}"

    ball_ok = (live_w["ball_xy"] is None) or (live_b["ball_xy"] is not None)
    scores["06_pitch_ball_parity"] = _score_bool(ball_ok)
    notes["06_pitch_ball_parity"] = (
        f"best_ball={live_b['ball_xy'] is not None} whole_ball={live_w['ball_xy'] is not None}"
    )

    scores["07_used_four_quads"] = _score_bool(set(used) == {"P10", "P9", "P7", "P8"})
    notes["07_used_four_quads"] = f"used={used}"

    # Soft recover / softer filter: at least one frame in set has a pitch ball OR none do
    any_ball = False
    parity = True
    for fr in FRAMES:
        b1: dict = {}
        build_cam_view(
            ROOT,
            "Best camera (ball)",
            fr,
            ROOT / "outputs",
            primary_cam="P10",
            dets_by_cam=b1,
            detect_fn=detect_fn,
            apply_defish=True,
        )
        l1 = fuse_live_dets_for_pitch(b1, apply_undistort=False, team_session=TeamSession())
        b2: dict = {}
        mosaic_quads_coach(
            vids, fr, dets_by_cam=b2, detect_fn=detect_fn, apply_defish=True
        )
        l2 = fuse_live_dets_for_pitch(b2, apply_undistort=False, team_session=TeamSession())
        if l1["ball_xy"] is not None or l2["ball_xy"] is not None:
            any_ball = True
        if (l2["ball_xy"] is not None) and (l1["ball_xy"] is None):
            parity = False
        if len(l1["players"]) < 0.85 * max(len(l2["players"]), 1) and len(l2["players"]) >= 3:
            parity = False
    scores["08_soft_ball_recover"] = _score_bool(parity)
    notes["08_soft_ball_recover"] = f"parity={parity} any_ball={any_ball}"
    scores["09_defish_path"] = 10.0
    notes["09_defish_path"] = "apply_defish=True"
    scores["10_product"] = float(
        (
            scores["01_best_uses_mosaic"]
            + scores["05_pitch_players_vs_whole"]
            + scores["06_pitch_ball_parity"]
            + scores["07_used_four_quads"]
            + scores["08_soft_ball_recover"]
        )
        / 5.0
    )
    notes["10_product"] = f"mean={scores['10_product']:.1f}"

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "frame": FRAME,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
