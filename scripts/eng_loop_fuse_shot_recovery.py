#!/usr/bin/env python3
"""Eng-loop: fuse shot/recovery negatives + synth TP."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_shot_recovery"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
V1 = ROOT / "data/processed/gold_sets/match3_events_v1/manifest.json"
PASS = 9.0


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_fuse_shot_recovery_gold.py")],
        cwd=str(ROOT),
    )
    labels = json.loads((CLIP / "labels.json").read_text(encoding="utf-8"))

    from scripts.eng_loop_heuristic_events import score_clip  # noqa: E402
    from src.events.events import EventDetector  # noqa: E402

    det = EventDetector()
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    synth_shot = next(c for c in v1["clips"] if c["id"] == "synth_shot_goal2")
    synth_rec = next(c for c in v1["clips"] if c["id"] == "synth_recovery")
    shot_sc = score_clip(synth_shot, det)
    rec_sc = score_clip(synth_rec, det)

    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    fuse_em = mod.run_events_on_timeline(
        json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    )
    shot_fps = 0
    for neg in labels.get("shot_negatives") or []:
        t0, t1 = float(neg["t_start"]), float(neg["t_end"])
        for em in fuse_em:
            if em["type"] != "shot":
                continue
            te = float(em["t_end"])
            if t0 - 0.25 <= te <= t1 + 0.25:
                shot_fps += 1
    rec_n = sum(1 for e in fuse_em if e["type"] == "recovery")

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in ("eng_loop_fuse_event_recall.py", "eng_loop_dribble_precision.py")
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_synth_shot_tp": _score(int(shot_sc.get("tp", 0)) >= 1),
        "03_synth_recovery_tp": _score(int(rec_sc.get("tp", 0)) >= 1),
        "04_fuse_shot_fp": _score(shot_fps == 0),
        "05_fuse_recovery_zero": _score(rec_n == 0),
        "06_regression": _score(reg),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "shot_sc": shot_sc,
        "recovery_sc": rec_sc,
        "fuse_shot_fp": shot_fps,
        "fuse_recovery_n": rec_n,
        "fuse_emits": fuse_em,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_shot_recovery >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
