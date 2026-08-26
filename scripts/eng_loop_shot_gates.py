#!/usr/bin/env python3
"""Eng-loop: shot gate patch — kickoff floor, goal-mouth, multi-frame, step cap."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events.events import (  # noqa: E402
    KICKOFF_FLOOR_S,
    SHOT_GOALWARD_STREAK_MIN,
    SHOT_MAX_STEP_M,
    EventDetector,
)
from src.state.types import Ball, FrameData, Player  # noqa: E402

OUT = ROOT / "reports/events_testing/shot_gates_eng_loop"
PASS = 9.0


def _run_unit() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_heuristic_events_e0.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-400:],
        "stderr": proc.stderr[-400:],
    }


def _early_glitch_shot_blocked() -> dict:
    det = EventDetector()
    dt = 0.25
    steps = [
        (0, 1.0, 5.0, 0.0),
        (1, 1.25, 8.0, 0.0),
        (2, 1.5, 24.0, 0.0),
    ]
    prev = None
    emits = []
    for fr, t, bx, by in steps:
        cur = FrameData(
            fr, t,
            [Player(1, 0, 5.0, 0.0, (0, 0, 10, 10), fr, t)],
            Ball(bx, by, (0, 0, 4, 4), fr, t),
        )
        if prev is not None:
            emits.extend(det.detect_events(cur, prev))
        prev = cur
    return {"ok": len(emits) == 0, "emits": emits}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    unit = _run_unit()
    glitch = _early_glitch_shot_blocked()
    gates = {
        "kickoff_floor_s": KICKOFF_FLOOR_S,
        "goalward_streak_min": SHOT_GOALWARD_STREAK_MIN,
        "shot_max_step_m": SHOT_MAX_STEP_M,
    }
    score = 10.0 if unit["ok"] and glitch["ok"] else 5.0
    report = {
        "score": score,
        "pass": score >= PASS,
        "gate": PASS,
        "gates": gates,
        "unit": unit,
        "early_glitch_shot": glitch,
        "note": "Re-render mosaic (emits_render.json) to apply on coach MP4.",
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
