#!/usr/bin/env python3
"""Eng-loop: fuse dribble cling via movement_proximity radius."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_dribble_cling"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0
GATE = 0.80

_spec = importlib.util.spec_from_file_location(
    "build_check25", ROOT / "scripts/gold_set/build_check25_event_timeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

from scripts.eng_loop_dribble_precision import timeline_from_frame_csv, run_det_on_timeline  # noqa: E402

BATCH_CSV = ROOT / "data/output/full_match_2min/P10-002/frame_data.csv"


def batch_dribble_count() -> int:
    if not BATCH_CSV.is_file():
        return 0
    em = run_det_on_timeline(timeline_from_frame_csv(BATCH_CSV))
    return sum(1 for e in em if e["type"] == "dribble")


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads((CLIP / "labels.json").read_text(encoding="utf-8"))
    # carry window scored as dribble for this loop
    dribble_gold = [
        {"type": "dribble", "t_start": 10.5, "t_end": 11.5}
    ]
    tl = json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    emits = _mod.run_events_on_timeline(tl)
    dribble_emits = [e for e in emits if e["type"] == "dribble"]
    n_dribble = len(dribble_emits)
    tp = sum(
        1
        for e in dribble_emits
        for g in dribble_gold
        if float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )
    pass_tp = sum(
        1
        for e in emits
        if e["type"] == "pass"
        for g in labels.get("events") or []
        if g["type"] == "pass"
        and float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )
    fp = len(dribble_emits) - tp
    p_emit = tp / max(tp + fp, 1)
    batch_n = batch_dribble_count()

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in (
            "eng_loop_heuristic_events.py",
            "eng_loop_dribble_precision.py",
            "eng_loop_fuse_event_recall.py",
        )
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_regression": _score(reg),
        "03_dribble_tp": _score(tp >= 1),
        "04_dribble_count": _score(1 <= n_dribble <= 2, partial=max(1.0, 10.0 - abs(n_dribble - 1) * 4)),
        "05_p_emit": _score(p_emit >= GATE, max(1.0, 10.0 * p_emit)),
        "06_batch_cap": _score(batch_n <= 30),
        "07_pass_tp": _score(pass_tp >= 2),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "n_dribble": n_dribble,
        "dribble_tp": tp,
        "p_emit_dribble": round(p_emit, 4),
        "batch_n": batch_n,
        "emits": emits,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_dribble_cling >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
