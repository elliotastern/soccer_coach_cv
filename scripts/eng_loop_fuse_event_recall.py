#!/usr/bin/env python3
"""Eng-loop: fuse timeline dribble recall on product 15s clip."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = 9.0
GATE_P_EMIT = 0.80
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_event_recall"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
META_15S = (
    ROOT
    / "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4/meta.json"
)
BATCH_CSV = ROOT / "data/output/full_match_2min/P10-002/frame_data.csv"

_spec = importlib.util.spec_from_file_location(
    "eh", ROOT / "scripts/eng_loop_heuristic_events.py"
)
_eh = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eh)
match_emits = _eh.match_emits

from src.events.events import EventDetector  # noqa: E402


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def run_fuse_events() -> list[dict]:
    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    tl = json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    return mod.run_events_on_timeline(tl)


def batch_dribble_count() -> int:
    from scripts.eng_loop_dribble_precision import run_det_on_timeline, timeline_from_frame_csv

    if not BATCH_CSV.is_file():
        return 0
    em = run_det_on_timeline(timeline_from_frame_csv(BATCH_CSV))
    return sum(1 for e in em if e["type"] == "dribble")


def fuse_window_score(labels: dict, emits: list[dict]) -> dict:
    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.score_windows(labels.get("events") or [], emits)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (CLIP / "timeline.json").is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts/gold_set/build_fuse_15s_timeline.py")],
            cwd=str(ROOT),
        )
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_fuse_event_gold.py")],
        cwd=str(ROOT),
    )

    parent_h = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_heuristic_events.py")],
        cwd=str(ROOT),
        capture_output=True,
    )
    parent_d = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_dribble_precision.py")],
        cwd=str(ROOT),
        capture_output=True,
    )

    labels = json.loads((CLIP / "labels.json").read_text(encoding="utf-8"))
    emits = run_fuse_events()
    (CLIP / "emits.json").write_text(
        json.dumps({"emits": emits}, indent=2), encoding="utf-8"
    )
    sc = fuse_window_score(labels, emits)
    dribble_emits = [e for e in emits if e["type"] == "dribble"]
    carry_emits = [e for e in emits if e["type"] in ("dribble", "movement")]
    n_dribble = len(dribble_emits)
    n_carry = len(carry_emits)
    carry_tp = sum(
        1
        for e in carry_emits
        for g in labels.get("events") or []
        if g["type"] in ("dribble", "movement")
        and float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )
    pass_gold = sum(1 for g in labels.get("events") or [] if g["type"] == "pass")
    pass_tp = sum(
        1
        for e in emits
        if e["type"] == "pass"
        for g in labels.get("events") or []
        if g["type"] == "pass"
        and float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )

    meta_n = 99
    meta_carry = 99
    if META_15S.is_file():
        meta = json.loads(META_15S.read_text(encoding="utf-8"))
        meta_em = meta.get("emits") or []
        meta_n = sum(1 for e in meta_em if e.get("type") == "dribble")
        meta_carry = sum(
            1 for e in meta_em if e.get("type") in ("dribble", "movement")
        )

    batch_n = batch_dribble_count()
    p_emit = float(sc.get("p_emit", 0.0))
    count_ok = 1 <= n_carry <= 4
    meta_ok = 1 <= meta_carry <= 4

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_heuristic_regression": _score(parent_h.returncode == 0),
        "03_dribble_precision_regression": _score(parent_d.returncode == 0),
        "04_fuse_gold": _score((CLIP / "timeline.json").is_file()),
        "05_carry_recall": _score(carry_tp >= 1),
        "06_fuse_carry_count": _score(
            count_ok, partial=max(1.0, 10.0 - abs(n_carry - 2) * 3)
        ),
        "07_fuse_p_emit": _score(p_emit >= GATE_P_EMIT, max(1.0, 10.0 * p_emit)),
        "08_pass_regression": _score(pass_tp >= pass_gold),
        "09_batch_cap": _score(batch_n <= 30),
        "10_meta_carry_count": _score(
            meta_ok, partial=max(1.0, 10.0 - abs(meta_carry - 2) * 3)
        ),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "fuse_score": sc,
        "n_dribble_fuse": n_dribble,
        "n_carry_fuse": n_carry,
        "carry_tp": carry_tp,
        "batch_n_dribble": batch_n,
        "meta_n_dribble": meta_n,
        "meta_n_carry": meta_carry,
        "emits": emits,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        print("FAIL eng-loop fuse_event_recall")
        return 1
    print("all fuse_event_recall >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
