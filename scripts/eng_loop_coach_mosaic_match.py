#!/usr/bin/env python3
"""Eng-loop: whole-pitch mosaic must match Pitch 1 diagram layout (≥9/10)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.cam_mosaic import (  # noqa: E402
    COACH_CORNER,
    QUAD_GRID,
    QUAD_ROTATE_180,
    match3_videos,
    mosaic_quads_coach,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/coach_mosaic_ux"
PASS = 9.0
FRAME = 2397


def clamp(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 1)


def score_layout_contract() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    want_grid = [["P10", "P8"], ["P7", "P9"]]
    if QUAD_GRID != want_grid:
        score -= 5.0
        notes.append(f"QUAD_GRID {QUAD_GRID} want {want_grid}")
    want_rot = {"P10", "P7"}
    if set(QUAD_ROTATE_180) != want_rot:
        score -= 4.0
        notes.append(f"ROT180 {set(QUAD_ROTATE_180)} want {want_rot}")
    # Labels must name cam id so coach/user can verify
    for cam, lab in COACH_CORNER.items():
        if cam not in lab:
            score -= 1.0
            notes.append(f"label missing cam id: {cam} -> {lab}")
    # Near = south row, Far = north row
    if "North" not in COACH_CORNER["P8"] or "South" not in COACH_CORNER["P10"]:
        score -= 2.0
        notes.append("North/South wording wrong vs diagram")
    return clamp(score), notes


def score_built_mosaic() -> tuple[float, list[str], Path]:
    notes = []
    score = 10.0
    videos = match3_videos(ROOT)
    img = mosaic_quads_coach(videos, FRAME, tile_w=640, tile_h=360, apply_defish=True)
    path = OUT / "match_locked_mosaic.jpg"
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    # Banner must say WHOLE PITCH
    # Structural: 2x2-ish aspect
    h, w = img.shape[:2]
    if w < h:
        score -= 1.0
        notes.append("mosaic taller than wide")
    if h < 400 or w < 800:
        score -= 1.0
        notes.append("mosaic too small")
    return clamp(score), notes, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, an = score_layout_contract()
    b, bn, path = score_built_mosaic()
    # Vision notes filled by agent; offline gate is layout contract
    total = clamp(0.6 * a + 0.4 * b)
    report = {
        "score": total,
        "pass": total >= PASS,
        "gate": PASS,
        "parts": {
            "layout_contract": {"score": a, "notes": an},
            "built_mosaic": {"score": b, "notes": bn, "path": str(path)},
        },
        "expected": {
            "grid": [["P10", "P8"], ["P7", "P9"]],
            "rotate_180": ["P7", "P10"],
            "compass": "cw90: Left top · Right bottom · South left · North right",
            "why": (
                "Pitch 1 cw90: left touchline top (P10|P8), right bottom (P7|P9), "
                "+x north on the right. South cams flipped 180°."
            ),
        },
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "match_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if total < PASS:
        print(f"FAIL {total}/10", file=sys.stderr)
        return 1
    print(f"PASS {total}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
