#!/usr/bin/env python3
"""Eng-loop: best-ball cam framing is readable + ball visible (≥9/10)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/events_testing/coach_best_ball_ux"
VIDEO = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_first_90s.mp4"
BATCH = ROOT / "data/output/match_4_5min"
PASS = 9.0
# Cam panel must keep most of a mosaic tile (anti micro-zoom).
MIN_CAM_W_FRAC = 0.55
MIN_CAM_H_FRAC = 0.55
MIN_ORANGE = 80


def orange_px(bgr) -> int:
    mask = cv2.inRange(bgr, np.array([175, 85, 0]), np.array([255, 215, 95]))
    return int(mask.sum() // 255)


def cases() -> list[dict]:
    return [
        {"name": "dribble_1_8s", "t": 1.8},
        {"name": "early_1_25s", "t": 1.25},
        {"name": "movement_3_5s", "t": 3.5},
    ]


def run_case(case: dict) -> dict:
    from apps.coach_emit_label_dashboard import (
        RENDER_TILE_H,
        RENDER_TILE_W,
        QUAD_CAMS,
        cached_read_frame,
        compose_best_ball_stack,
        load_video_meta,
        split_coach_stack,
        tile_ball_box_rect,
        tile_ball_box_score,
        vid_idx_for_match_t,
    )

    meta = load_video_meta(VIDEO)
    idx = vid_idx_for_match_t(meta, case["t"], 400)
    src_fr = meta["start"] + idx * meta["stride"]
    rgb = cached_read_frame(str(VIDEO), idx)
    stack, cam = compose_best_ball_stack(rgb, src_fr, BATCH)
    mosaic, pitch, _, offsets, tw, th = split_coach_stack(rgb)
    tiles = {c: mosaic[y : y + th, x : x + tw] for c, (x, y) in offsets.items()}
    scores = {c: tile_ball_box_score(tiles[c]) for c in tiles}

    # Cam panel = everything above pitch (best-ball tile + label bar).
    pitch_h = pitch.shape[0] if pitch is not None else 0
    cam_h = max(1, stack.shape[0] - pitch_h - 56)
    cam_panel = stack[:cam_h]
    # Drop black label strip for orange/score checks.
    body = cam_panel[:-40] if cam_panel.shape[0] > 80 else cam_panel
    bgr = cv2.cvtColor(body, cv2.COLOR_RGB2BGR)
    ox = orange_px(bgr)
    has_box = tile_ball_box_rect(body) is not None
    frac_w = body.shape[1] / float(RENDER_TILE_W)
    frac_h = body.shape[0] / float(RENDER_TILE_H)
    framing_ok = frac_w >= MIN_CAM_W_FRAC and frac_h >= MIN_CAM_H_FRAC
    cam_ok = cam in QUAD_CAMS and scores.get(cam, 0) >= 80
    ball_ok = has_box and ox >= MIN_ORANGE
    ok = cam_ok and ball_ok and framing_ok

    shot = OUT / f"frame_{case['name']}.jpg"
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(shot), cv2.cvtColor(stack, cv2.COLOR_RGB2BGR))
    return {
        "name": case["name"],
        "pass": ok,
        "cam": cam,
        "orange_px": ox,
        "has_box": has_box,
        "frac_w": round(frac_w, 3),
        "frac_h": round(frac_h, 3),
        "framing_ok": framing_ok,
        "cam_ok": cam_ok,
        "shot": str(shot),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = [run_case(c) for c in cases()]
    passed = sum(1 for c in checks if c.get("pass"))
    total = len(checks)
    payload = {
        "checks": checks,
        "score": round(10.0 * passed / max(total, 1), 2),
        "pass": passed / max(total, 1) * 10 >= PASS,
        "gate": PASS,
        "passed": passed,
        "total": total,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "pw_best_ball_frame_score.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
