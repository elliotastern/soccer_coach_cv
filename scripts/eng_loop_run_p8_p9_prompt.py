#!/usr/bin/env python3
"""Run full p8_p9_congruence PROMPT.md workflow (gates + proof + render)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/p8_p9_congruence"
PASS = 9.0

STEPS = [
    ("mirror", [sys.executable, "scripts/gold_set/mirror_p8_p9_lr_landmarks.py"]),
    ("congruence", [sys.executable, "scripts/eng_loop_p8_p9_congruence.py"]),
    ("mosaic_match", [sys.executable, "scripts/eng_loop_coach_mosaic_match.py"]),
    ("ball_boxes", [sys.executable, "scripts/eng_loop_ball_boxes.py"]),
    ("player_map", [sys.executable, "scripts/gold_set/eng_loop_player_map.py"]),
    ("streamlit_review", [sys.executable, "scripts/eng_loop_streamlit_review.py"]),
]


def run_step(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    score = None
    passed = proc.returncode == 0
    for line in out.splitlines():
        if line.startswith("{") and "score" in line:
            try:
                rec = json.loads(line)
                score = rec.get("score")
                passed = bool(rec.get("pass", passed))
            except json.JSONDecodeError:
                pass
    if "PASS" in out and score is None:
        passed = True
    if "BALL_BOX_SCORE" in out:
        for line in out.splitlines():
            if "BALL_BOX_SCORE" in line:
                score = float(line.split()[1].split("/")[0])
    if "STREAMLIT_REVIEW_SCORE" in out:
        for line in out.splitlines():
            if "STREAMLIT_REVIEW_SCORE" in line:
                score = float(line.split()[1].split("/")[0])
    return {
        "name": name,
        "ok": passed and (score is None or score >= PASS),
        "score": score,
        "exit": proc.returncode,
        "tail": (out or err)[-400:],
    }


def main() -> int:
    results = [run_step(name, cmd) for name, cmd in STEPS]
    report = {"steps": results, "gate": PASS}
    report["pass"] = all(r["ok"] for r in results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_prompt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
