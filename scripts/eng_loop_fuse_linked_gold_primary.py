#!/usr/bin/env python3
"""Eng-loop: fuse event gold scored on linked stable-id timeline."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_linked_gold_primary"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_fuse_event_gold.py")],
        cwd=str(ROOT),
    )

    from scripts.gold_set.timeline_player_ids import carry_window_id_swaps  # noqa: E402

    linked_path = CLIP / "timeline_linked.json"
    labels = json.loads((CLIP / "labels.json").read_text(encoding="utf-8"))
    primary = labels.get("timeline_primary") or "timeline_linked.json"
    dribble_gold = next(
        (g for g in labels.get("events") or [] if g.get("type") == "dribble"),
        {},
    )
    expected_pid = int(dribble_gold.get("expected_carrier_pid") or 0)

    spec = importlib.util.spec_from_file_location(
        "build_check25",
        ROOT / "scripts/gold_set/build_check25_event_timeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    linked = json.loads(linked_path.read_text(encoding="utf-8"))
    emits = mod.run_events_on_timeline(linked)
    (CLIP / "emits_linked.json").write_text(
        json.dumps({"timeline": primary, "emits": emits}, indent=2),
        encoding="utf-8",
    )

    dribbles = [e for e in emits if e["type"] == "dribble"]
    passes = [e for e in emits if e["type"] == "pass"]
    tp = sum(
        1
        for e in dribbles
        for g in labels.get("events") or []
        if g["type"] == "dribble"
        and float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )
    carrier_ok = any(expected_pid in (e.get("players") or []) for e in dribbles)
    swaps = carry_window_id_swaps(linked)

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in (
            "eng_loop_fuse_teleport_mask.py",
            "eng_loop_fuse_event_recall.py",
            "eng_loop_fuse_shot_recovery.py",
        )
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_linked_timeline": _score(linked_path.is_file() and primary == "timeline_linked.json"),
        "03_linked_dribble_tp": _score(tp >= 1),
        "04_emit_counts": _score(len(passes) == 2 and len(dribbles) == 1),
        "05_carrier_pid": _score(carrier_ok),
        "06_carry_id_stable": _score(swaps["swaps"] == 0),
        "07_regression": _score(reg),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "expected_carrier_pid": expected_pid,
        "carrier_ok": carrier_ok,
        "linked_tp": tp,
        "n_pass": len(passes),
        "n_dribble": len(dribbles),
        "carrier_swaps": swaps,
        "emits": emits,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_linked_gold_primary >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
