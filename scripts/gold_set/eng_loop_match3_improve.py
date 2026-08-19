#!/usr/bin/env python3
"""Score Match 3 multicam improve plan (need 9+/10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/product/MATCH3_MULTICAM_IMPROVE_PLAN.md"
OUT = ROOT / "reports/eval_match3/improve_eng_loop"
PASS = 9.0


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def score_plan() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    text = PLAN.read_text(encoding="utf-8") if PLAN.is_file() else ""
    if not text:
        return 0.0, ["plan missing"]
    low = text.lower()
    need = [
        ("p_emit", "P_emit goal"),
        ("0.80", "0.80 gate"),
        ("clear-ball", "clear-ball R"),
        ("4 m", "agree m"),
        ("do not average", "no midpoint / hard no"),
        ("match-3 detect thr", "T1 thr"),
        ("overlapping", "L2 landmarks"),
        ("min_support", "H1 hull"),
        ("pitch 1", "Pitch 1 meters"),
        ("phase 2", "no Phase 2 fusion"),
        ("video title", "cam id"),
        ("eng_loop_match3_improve", "eng-loop wire"),
    ]
    for needle, label in need:
        if needle not in low:
            score -= 0.8
            notes.append(f"plan missing {label}")
    if "≥ 0.80" not in text and ">= 0.80" not in text:
        score -= 1.0
        notes.append("plan missing ≥ 0.80 wording")
    if "p7" not in low and "0.60" not in text:
        score -= 0.5
        notes.append("plan should call out P7@0.60")
    return clamp(score), notes


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores, notes = {}, {}
    scores["plan"], notes["plan"] = score_plan()
    fails = [f"plan={scores['plan']} {notes['plan']}" for _ in [0] if scores["plan"] < PASS]
    summary = {"scores": scores, "notes": notes, "pass": PASS, "fails": fails}
    (OUT / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(scores)
    if fails:
        print("BELOW 9")
        for f in fails:
            print(" ", f)
        return 1
    print("all subgoals >= 9/10")
    print(f"wrote {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
