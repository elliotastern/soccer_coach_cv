#!/usr/bin/env python3
"""Eng-loop: dribble on stable-id linked fuse timeline."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_dribble_linked"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0
CARRIER_ID = 34


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.gold_set.timeline_player_ids import (  # noqa: E402
        carry_window_id_swaps,
        relink_timeline_players,
    )

    raw = json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    linked = relink_timeline_players(raw)
    if (CLIP / "timeline_linked.json").is_file():
        linked = json.loads((CLIP / "timeline_linked.json").read_text(encoding="utf-8"))

    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    raw_em = mod.run_events_on_timeline(raw)
    linked_em = mod.run_events_on_timeline(linked)
    linked_d = [e for e in linked_em if e["type"] == "dribble"]
    raw_d = [e for e in raw_em if e["type"] == "dribble"]
    linked_tp = sum(
        1 for e in linked_d if 10.25 <= float(e["t_end"]) <= 11.55
    )
    raw_tp = sum(1 for e in raw_d if 10.25 <= float(e["t_end"]) <= 11.55)
    carrier_ok = any(
        CARRIER_ID in (e.get("players") or []) for e in linked_d
    )
    swaps = carry_window_id_swaps(linked)

    batch_n = 0
    from scripts.eng_loop_dribble_precision import (  # noqa: E402
        run_det_on_timeline,
        timeline_from_frame_csv,
    )

    csv = ROOT / "data/output/full_match_2min/P10-002/frame_data.csv"
    if csv.is_file():
        batch_n = sum(
            1
            for e in run_det_on_timeline(timeline_from_frame_csv(csv))
            if e["type"] == "dribble"
        )

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
            "eng_loop_fuse_carry_dribble.py",
            "eng_loop_fuse_player_id_stable.py",
        )
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_regression": _score(reg),
        "03_linked_dribble_tp": _score(linked_tp >= 1),
        "04_carrier_pid": _score(carrier_ok),
        "05_raw_dribble_tp": _score(raw_tp >= 1),
        "06_batch_cap": _score(batch_n <= 30),
        "07_id_stable": _score(swaps["swaps"] <= 1),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "linked_tp": linked_tp,
        "raw_tp": raw_tp,
        "carrier_ok": carrier_ok,
        "batch_n": batch_n,
        "linked_emits": linked_em,
        "carrier_swaps": swaps,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_dribble_linked >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
