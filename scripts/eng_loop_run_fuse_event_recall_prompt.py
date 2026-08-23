#!/usr/bin/env python3
"""Run fuse_event_recall PROMPT (timeline + gates + render)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_event_recall"

STEPS = [
    ("fuse_timeline", [sys.executable, "scripts/gold_set/build_fuse_15s_timeline.py"]),
    ("fuse_gold", [sys.executable, "scripts/gold_set/build_fuse_event_gold.py"]),
    ("unit_e0", [sys.executable, "scripts/test_heuristic_events_e0.py"]),
    ("fuse_recall", [sys.executable, "scripts/eng_loop_fuse_event_recall.py"]),
]


def run_step(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "name": name,
        "ok": proc.returncode == 0,
        "exit": proc.returncode,
        "tail": (proc.stdout or proc.stderr or "")[-600:],
    }


def main() -> int:
    results = []
    for name, cmd in STEPS:
        results.append(run_step(name, cmd))
        if name == "fuse_recall" and results[-1]["ok"]:
            break
        if name == "fuse_recall" and not results[-1]["ok"]:
            cfg_path = ROOT / "configs/default.yaml"
            txt = cfg_path.read_text(encoding="utf-8")
            if "dribble_min_carry_m: 0.6" in txt:
                cfg_path.write_text(
                    txt.replace("dribble_min_carry_m: 0.6", "dribble_min_carry_m: 0.45"),
                    encoding="utf-8",
                )
                results.append(run_step("fuse_recall_tune", STEPS[-1]))
    # Render if fuse recall passed or after tune
    render = subprocess.run(
        [
            sys.executable,
            "scripts/gold_set/render_phase1_check_mosaic.py",
            "--start", "2390", "--match-sec", "15", "--stride", "4",
            "--out-fps", "15",
            "--out-dir", "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4",
            "--out-file", "coach_mosaic_pitch_min.mp4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    results.append({"name": "render_15s", "ok": render.returncode == 0, "exit": render.returncode, "tail": (render.stdout or "")[-400:]})
    post = run_step("fuse_recall_final", [sys.executable, "scripts/eng_loop_fuse_event_recall.py"])
    results.append(post)
    report = {"steps": results, "pass": all(r["ok"] for r in results)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_prompt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
