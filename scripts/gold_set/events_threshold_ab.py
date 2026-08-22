#!/usr/bin/env python3
"""T1: threshold A/B for heuristic events — keep defaults if P_emit holds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eng_loop_heuristic_events import GOLD, match_emits, run_timeline  # noqa: E402
from src.events.events import EventDetector  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/heuristic_events"


def score_pack(det: EventDetector) -> dict:
    manifest = json.loads(GOLD.read_text(encoding="utf-8"))
    tp = fp = fn = 0
    for clip in manifest["clips"]:
        if not clip.get("timeline"):
            continue
        tl = json.loads((ROOT / clip["timeline"]).read_text(encoding="utf-8"))
        lab = json.loads((ROOT / clip["labels"]).read_text(encoding="utf-8"))
        emits = run_timeline(tl, det)
        m = match_emits(lab.get("events") or [], emits)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"p_emit": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    variants = {
        "default": EventDetector(),
        "pass_thr_8": EventDetector(pass_velocity_threshold=8.0),
        "shot_thr_20": EventDetector(shot_velocity_threshold=20.0),
        "recovery_0_5": EventDetector(recovery_proximity=0.5),
    }
    rows = {}
    for name, det in variants.items():
        rows[name] = score_pack(det)
        print(name, rows[name])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t1_threshold_ab.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Promote only if default still best on P_emit
    best = max(rows.items(), key=lambda kv: (kv[1]["p_emit"], kv[1]["recall"]))
    print("best", best[0], best[1])
    if rows["default"]["p_emit"] < 0.80:
        print("FAIL default below gate")
        return 1
    print("T1 keep default thresholds (P_emit holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
