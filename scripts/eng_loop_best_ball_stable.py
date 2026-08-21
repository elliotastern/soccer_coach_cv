#!/usr/bin/env python3
"""Eng-loop: Best camera (ball) stays stable — 1 stage cam, all players on pitch.

Gate: every check ≥ 9/10. Report under
reports/eval_match3/improve_eng_loop/best_ball_stable/.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/eval_match3/improve_eng_loop/best_ball_stable"
FRAME = 2400
GATE = 9.0


def _score_bool(ok: bool, partial: float = 3.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    # 1 import symbols (fresh + reload path)
    try:
        from src.review import cam_mosaic as cm0

        importlib.reload(cm0)
        assert hasattr(cm0, "fill_quad_dets_for_pitch")
        assert hasattr(cm0, "build_cam_view")
        assert hasattr(cm0, "match3_videos")
        scores["01_import_fill_quad"] = 10.0
        notes["01_import_fill_quad"] = "reload + hasattr ok"
    except Exception as exc:  # noqa: BLE001
        scores["01_import_fill_quad"] = 0.0
        notes["01_import_fill_quad"] = f"{exc}"
        (OUT / "scores.json").write_text(
            json.dumps({"pass": False, "scores": scores, "notes": notes}, indent=2),
            encoding="utf-8",
        )
        print("FAIL import")
        return 1

    from src.perception.rfdetr_local import LocalRFDETRDetector
    from src.review.cam_mosaic import (
        _is_ball_det,
        best_cam_for_frame,
        build_cam_view,
        fill_quad_dets_for_pitch,
        match3_videos,
    )
    from src.review.frame_sync import keep_top1_ball
    from src.review.multicam_fuse import fuse_live_dets_for_pitch

    vids = match3_videos(ROOT)
    scores["02_videos_present"] = _score_bool(
        all(c in vids and vids[c].is_file() for c in ("P10", "P9", "P7", "P8"))
    )
    notes["02_videos_present"] = ",".join(sorted(vids.keys()))

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

    # 3 Best-camera view builds without exception
    bag: dict = {}
    try:
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
        scores["03_build_best_view"] = 10.0 if img is not None and len(used) == 1 else 4.0
        notes["03_build_best_view"] = f"used={used} shape={getattr(img, 'shape', None)}"
    except Exception as exc:  # noqa: BLE001
        scores["03_build_best_view"] = 0.0
        notes["03_build_best_view"] = traceback.format_exc()
        used = []

    # 4 after build_cam_view, bag already filled with quads (in-module path)
    n_cams = sum(
        1
        for c in ("P10", "P9", "P7", "P8")
        if bag.get(c) is not None and f"{c}__wh" in bag
    )
    scores["04_bag_has_quads"] = 10.0 if n_cams >= 3 else max(0.0, 3.0 * n_cams)
    notes["04_bag_has_quads"] = f"quad_cams_in_bag={n_cams}"

    n_players = sum(
        1
        for c in ("P10", "P9", "P7", "P8")
        for d in (bag.get(c) or [])
        if not _is_ball_det(d)
    )
    n_balls = sum(
        1
        for c in ("P10", "P9", "P7", "P8")
        for d in (bag.get(c) or [])
        if _is_ball_det(d)
    )
    scores["05_players_multi_cam"] = 10.0 if n_players >= 4 else max(0.0, 2.0 * n_players)
    notes["05_players_multi_cam"] = f"video_players={n_players}"
    scores["06_single_ball"] = 10.0 if n_balls <= 1 else max(0.0, 10.0 - 3.0 * (n_balls - 1))
    notes["06_single_ball"] = f"balls_in_bag={n_balls}"

    live = fuse_live_dets_for_pitch(bag, apply_undistort=False)
    scores["07_pitch_players"] = (
        10.0 if len(live["players"]) >= 4 else max(0.0, 2.0 * len(live["players"]))
    )
    notes["07_pitch_players"] = (
        f"fused={len(live['players'])} cams={live['cams']} ball={live['ball_xy'] is not None}"
    )
    scores["08_pitch_more_than_stage"] = _score_bool(
        len(live["players"]) >= max(2, len(used)), 4.0
    )
    notes["08_pitch_more_than_stage"] = (
        f"stage_cams={len(used)} pitch_players={len(live['players'])}"
    )

    # 9 fill_quad idempotent / callable again
    try:
        fill_quad_dets_for_pitch(
            vids, FRAME, bag, detect_fn, True, single_ball=True
        )
        scores["09_fill_idempotent"] = 10.0
        notes["09_fill_idempotent"] = "second fill ok"
    except Exception as exc:  # noqa: BLE001
        scores["09_fill_idempotent"] = 0.0
        notes["09_fill_idempotent"] = str(exc)

    # 10 app import path simulation
    try:
        from src.review.cam_mosaic import (
            build_cam_view as b2,
            fill_quad_dets_for_pitch as f2,
            match3_videos as m2,
        )

        assert callable(b2) and callable(f2) and callable(m2)
        scores["10_app_import_triplet"] = 10.0
        notes["10_app_import_triplet"] = "triplet import ok"
    except Exception as exc:  # noqa: BLE001
        scores["10_app_import_triplet"] = 0.0
        notes["10_app_import_triplet"] = str(exc)

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "frame": FRAME,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
        "best_cam_hint": best_cam_for_frame(ROOT / "outputs", FRAME, fallback="P10"),
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
