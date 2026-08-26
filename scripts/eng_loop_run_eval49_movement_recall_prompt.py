#!/usr/bin/env python3
"""Run eval49_movement_recall PROMPT."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/eval49_movement_recall"

STEPS = [
    ("unit_e0", [sys.executable, "scripts/test_heuristic_events_e0.py"]),
    ("eval49_recall", [sys.executable, "scripts/eng_loop_eval49_movement_recall.py"]),
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
