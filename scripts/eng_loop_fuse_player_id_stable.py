#!/usr/bin/env python3
"""Eng-loop: fuse player id stability on product 15s timeline."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_player_id_stable"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.gold_set.timeline_player_ids import (  # noqa: E402
        carry_window_id_swaps,
        relink_timeline_players,
    )

    raw = json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    before = carry_window_id_swaps(raw)
    linked = relink_timeline_players(raw)
    after = carry_window_id_swaps(linked)

    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    raw_emits = mod.run_events_on_timeline(raw)
    dribbles = [e for e in raw_emits if e["type"] == "dribble"]
    tp = sum(
        1
        for e in dribbles
        if 10.25 <= float(e["t_end"]) <= 11.55
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
            "eng_loop_team_stable.py",
        )
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_regression": _score(reg),
        "03_carry_id_swaps": _score(after["swaps"] <= 1, max(1.0, 10.0 - after["swaps"])),
        "04_unique_carrier_ids": _score(
            after["unique_ids"] <= 3,
            max(1.0, 10.0 - after["unique_ids"]),
        ),
        "05_dribble_tp": _score(tp >= 1),
        "06_swap_reduction": _score(after["swaps"] < before["swaps"]),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "before": before,
        "after": after,
        "dribble_tp": tp,
        "raw_emits": raw_emits,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (CLIP / "timeline_linked.json").write_text(
        json.dumps(linked, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_player_id_stable >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
