#!/usr/bin/env python3
"""Run dribble_precision PROMPT workflow (gates + render)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/dribble_precision"
PASS = 9.0

STEPS = [
    ("build_gold", [sys.executable, "scripts/gold_set/build_match3_dribble_gold.py"]),
    ("build_events_v1", [sys.executable, "scripts/gold_set/build_match3_events_gold.py"]),
    ("unit_e0", [sys.executable, "scripts/test_heuristic_events_e0.py"]),
    ("heuristic_events", [sys.executable, "scripts/eng_loop_heuristic_events.py"]),
    ("dribble_precision", [sys.executable, "scripts/eng_loop_dribble_precision.py"]),
    (
        "render_15s",
        [
            sys.executable,
            "scripts/gold_set/render_phase1_check_mosaic.py",
            "--start",
            "2390",
            "--match-sec",
            "15",
            "--stride",
            "4",
            "--out-fps",
            "15",
            "--out-dir",
            "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4",
            "--out-file",
            "coach_mosaic_pitch_min.mp4",
        ],
    ),
    ("handover_sync", [sys.executable, "scripts/gold_set/build_phase1_handover_dashboard.py"]),
]


def run_step(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    return {"name": name, "ok": ok, "exit": proc.returncode, "tail": out[-500:]}


def main() -> int:
    results = [run_step(name, cmd) for name, cmd in STEPS]
    # Re-score after render for gate 12
    proc = subprocess.run(
        [sys.executable, "scripts/eng_loop_dribble_precision.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    results.append(
        {
            "name": "dribble_precision_post_render",
            "ok": proc.returncode == 0,
            "exit": proc.returncode,
            "tail": (proc.stdout or proc.stderr or "")[-500:],
        }
    )
    report = {"steps": results, "gate": PASS}
    report["pass"] = all(r["ok"] for r in results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_prompt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
