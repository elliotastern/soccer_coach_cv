#!/usr/bin/env python3
"""Eng-loop: movement temporal window + eval49 holdout."""
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
OUT = ROOT / "reports/eval_match3/improve_eng_loop/movement_carry"
PROMPT = OUT / "PROMPT.md"
GOLD = ROOT / "data/processed/gold_sets/match3_events_v1/manifest.json"
EVAL49 = (
    ROOT
    / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_eval_49s"
)
FUSE15 = (
    ROOT
    / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
)
HOLDOUT = (
    ROOT
    / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_holdout_pass"
)

_spec = importlib.util.spec_from_file_location(
    "eh", ROOT / "scripts/eng_loop_heuristic_events.py"
)
_eh = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eh)
score_clip = _eh.score_clip

from src.events.events import EventDetector  # noqa: E402


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def _score_fuse_clip(clip_dir: Path) -> dict | None:
    if not (clip_dir / "timeline.json").is_file():
        return None
    if not (clip_dir / "labels.json").is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    tl = json.loads((clip_dir / "timeline.json").read_text(encoding="utf-8"))
    labs = json.loads((clip_dir / "labels.json").read_text(encoding="utf-8"))
    emits = mod.run_events_on_timeline(tl)
    return mod.score_windows(labs.get("events") or [], emits)


def _fp_outside_gold(clip_dir: Path, emits: list | None = None) -> int:
    if emits is None:
        spec = importlib.util.spec_from_file_location(
            "build_check25",
            ROOT / "scripts/gold_set/build_check25_event_timeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        tl = json.loads((clip_dir / "timeline.json").read_text(encoding="utf-8"))
        emits = mod.run_events_on_timeline(tl)
    labs = json.loads((clip_dir / "labels.json").read_text(encoding="utf-8"))
    gold = labs.get("events") or []
    used = set()
    tp = 0
    for em in emits:
        matched = False
        for i, g in enumerate(gold):
            if i in used or g["type"] != em["type"]:
                continue
            gt = float(g.get("t_end", g.get("t_start", 0.0)))
            et = float(em.get("t_end", em.get("t_start", 0.0)))
            if abs(gt - et) <= 0.55:
                used.add(i)
                matched = True
                break
        if matched:
            tp += 1
    return len(emits) - tp


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    det = EventDetector()
    cfg = (ROOT / "configs/default.yaml").read_text(encoding="utf-8")

    manifest = json.loads(GOLD.read_text(encoding="utf-8"))
    synth_mv = next(
        (score_clip(c, det) for c in manifest["clips"] if c["id"] == "synth_movement_midfield"),
        None,
    )
    eval49 = _score_fuse_clip(EVAL49)
    fuse15 = _score_fuse_clip(FUSE15)
    holdout = _score_fuse_clip(HOLDOUT)

    unit_ok = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_heuristic_events_e0.py")],
        cwd=str(ROOT),
        capture_output=True,
    ).returncode == 0
    parent_ok = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_heuristic_events.py")],
        cwd=str(ROOT),
        capture_output=True,
    ).returncode == 0
    dribble_ok = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_dribble_precision.py")],
        cwd=str(ROOT),
        capture_output=True,
    ).returncode == 0

    eval49_p = float(eval49["p_emit"]) if eval49 else 0.0
    fuse15_p = float(fuse15["p_emit"]) if fuse15 else 0.0
    holdout_p = float(holdout["p_emit"]) if holdout else 0.0
    synth_tp = int(synth_mv["tp"]) if synth_mv else 0
    eval49_fp = _fp_outside_gold(EVAL49) if eval49 else 99

    comps = {
        "01_prompt": _score(PROMPT.is_file()),
        "02_unit_e0": _score(unit_ok),
        "03_parent_heuristic": _score(parent_ok),
        "04_synth_movement_tp": _score(synth_tp >= 1),
        "05_eval49_p_emit": _score(
            eval49_p >= GATE_P_EMIT, partial=max(1.0, 10.0 * eval49_p)
        ),
        "06_eval49_no_extra_fp": _score(eval49_fp == 0),
        "07_fuse15_no_regress": _score(fuse15_p >= GATE_P_EMIT),
        "08_holdout_no_regress": _score(
            holdout_p >= GATE_P_EMIT if holdout else True
        ),
        "09_dribble_precision": _score(dribble_ok),
        "10_config_movement_knobs": _score(
            "movement_window_frames" in cfg and "movement_min_carry_m" in cfg
        ),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "eval49_score": eval49,
        "fuse15_score": fuse15,
        "holdout_score": holdout,
        "synth_movement": synth_mv,
        "eval49_fp_outside_gold": eval49_fp,
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "failed": failed, "eval49_p": eval49_p}, indent=2))
    print("WROTE", OUT / "scores.json")
    if not payload["pass"]:
        print("FAIL eng-loop movement_carry")
        return 1
    print("all movement_carry >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
