#!/usr/bin/env python3
"""Eng-loop: P8/P9 north-end congruence after quadrant swap (≥9/10)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/eval_match3/improve_eng_loop/p8_p9_congruence"
PASS = 9.0
FRAME = 3050
CALIB = ROOT / "reports/eval_match3/match3_pitch_calib"
DASH = ROOT / "reports/eval_match3/landmark_dashboard"


def clamp(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 1)


def score_quad_grid() -> tuple[float, list[str]]:
    from src.review.cam_mosaic import QUAD_GRID, QUAD_ROTATE_180

    notes = []
    score = 10.0
    want = [["P10", "P9"], ["P7", "P8"]]
    if QUAD_GRID != want:
        score -= 6.0
        notes.append(f"QUAD_GRID {QUAD_GRID} want {want}")
    if set(QUAD_ROTATE_180) != {"P10", "P9"}:
        score -= 3.0
        notes.append(f"ROT180 {set(QUAD_ROTATE_180)} want P10+P9")
    return clamp(score), notes


def score_calibs_exist() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    for cam in ("P8", "P9"):
        p = CALIB / f"{cam}_manual.json"
        if not p.is_file():
            score -= 5.0
            notes.append(f"missing {p.name}")
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if len(rec.get("landmark_names") or []) < 4:
            score -= 3.0
            notes.append(f"{cam} <4 landmarks")
    return clamp(score), notes


def score_diagram_cam_xy() -> tuple[float, list[str]]:
    """After swap: P9 left (+y), P8 right (-y) in pitch meters."""
    notes = []
    score = 10.0
    html = (DASH / "index.html").read_text(encoding="utf-8")
    if "CAM_XY.P8 = [hx * 0.4, hy + 8]" in html:
        score -= 5.0
        notes.append("P8 chip still north-left (+y) in CAM_XY")
    if "CAM_XY.P9 = [hx * 0.45, -hy - 10]" in html:
        score -= 5.0
        notes.append("P9 chip still north-right (-y) in CAM_XY")
    if score >= 9.0:
        notes.append("diagram CAM_XY swapped (or updated)")
    return clamp(score), notes


def score_map_sides() -> tuple[float, list[str]]:
    """P8 maps mostly −y (north-right); P9 mostly +y (north-left) at sample frame."""
    notes = []
    score = 10.0
    try:
        from src.perception.rfdetr_local import LocalRFDETRDetector
        from src.review.cam_mosaic import match3_videos, mosaic_quads_coach
        from src.review.frame_sync import keep_top1_ball
        from src.review.multicam_fuse import fuse_live_dets_for_pitch
    except ImportError as exc:
        return 0.0, [f"import failed: {exc}"]

    vids = match3_videos(ROOT)
    if not vids.get("P8") or not vids.get("P9"):
        return 0.0, ["P8/P9 videos missing"]

    bag: dict = {}
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
    )

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    mosaic_quads_coach(
        vids, FRAME, tile_w=320, tile_h=180,
        dets_by_cam=bag, detect_fn=detect_fn, apply_defish=True,
    )
    live = fuse_live_dets_for_pitch(bag, apply_undistort=False, debug_cam=True)
    raw = list(live.get("player_maps_all") or [])
    p8_y = [ym for xm, ym, c in raw if c == "P8"]
    p9_y = [ym for xm, ym, c in raw if c == "P9"]
    if len(p8_y) < 2:
        score -= 4.0
        notes.append(f"P8 raw maps n={len(p8_y)}")
    elif sum(1 for y in p8_y if y < 0) < len(p8_y) * 0.6:
        score -= 4.0
        notes.append(f"P8 y not mostly right (−y): {p8_y[:6]}")
    if len(p9_y) < 2:
        score -= 4.0
        notes.append(f"P9 raw maps n={len(p9_y)}")
    elif sum(1 for y in p9_y if y > 0) < len(p9_y) * 0.6:
        score -= 4.0
        notes.append(f"P9 y not mostly left (+y): {p9_y[:6]}")
    if not notes:
        notes.append(f"P8 n={len(p8_y)} P9 n={len(p9_y)} side OK")
    return clamp(score), notes


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    for name, fn in (
        ("01_quad_grid", score_quad_grid),
        ("02_calibs", score_calibs_exist),
        ("03_diagram_xy", score_diagram_cam_xy),
        ("04_map_sides_fr3050", score_map_sides),
    ):
        sc, notes = fn()
        parts[name] = {"score": sc, "notes": notes}

    scores = [p["score"] for p in parts.values()]
    total = clamp(sum(scores) / max(len(scores), 1))
    report = {
        "score": total,
        "pass": total >= PASS,
        "gate": PASS,
        "parts": parts,
        "baseline_commit": "749eaae",
        "prompt": str(OUT / "PROMPT.md"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "congruence_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if total >= PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
