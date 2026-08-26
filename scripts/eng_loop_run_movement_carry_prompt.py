#!/usr/bin/env python3
"""Run movement_carry PROMPT (gold rebuild + unit + gates)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/movement_carry"

STEPS = [
    ("gold_rebuild", [sys.executable, "scripts/gold_set/build_match3_events_gold.py"]),
    ("unit_e0", [sys.executable, "scripts/test_heuristic_events_e0.py"]),
    ("movement_carry", [sys.executable, "scripts/eng_loop_movement_carry.py"]),
]


def run_step(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "name": name,
        "ok": proc.returncode == 0,
        "exit": proc.returncode,
        "tail": (proc.stdout or proc.stderr or "")[-800:],
    }


def main() -> int:
    results = [run_step(name, cmd) for name, cmd in STEPS]
    report = {"steps": results, "pass": all(r["ok"] for r in results)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_prompt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
