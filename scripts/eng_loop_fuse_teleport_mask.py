#!/usr/bin/env python3
"""Eng-loop: fuse ball + near-ball player teleport masking."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse_teleport_mask"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0
GATE = 0.80

_spec = importlib.util.spec_from_file_location(
    "build_check25", ROOT / "scripts/gold_set/build_check25_event_timeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

from scripts.gold_set.audit_fuse_teleports import audit_timeline  # noqa: E402
from scripts.gold_set.timeline_player_ids import relink_timeline_players  # noqa: E402


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((CLIP / "timeline.json").read_text(encoding="utf-8"))
    relinked = relink_timeline_players(raw)
    audit = audit_timeline(relinked)
    emits = _mod.run_events_on_timeline(raw)
    dribble_gold = [{"type": "dribble", "t_start": 10.5, "t_end": 11.5}]
    dribbles = [e for e in emits if e["type"] == "dribble"]
    passes = [e for e in emits if e["type"] == "pass"]
    tp = sum(
        1
        for e in dribbles
        for g in dribble_gold
        if float(g["t_start"]) - 0.25 <= float(e["t_end"]) <= float(g["t_end"]) + 0.25
    )
    teleport_ts = [float(x["t"]) for x in audit.get("ball_teleports") or []]
    emit_on_teleport = 0
    for em in emits:
        te = float(em["t_end"])
        for tt in teleport_ts:
            if 0 <= te - tt <= 0.15:
                emit_on_teleport += 1
                break

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in (
            "eng_loop_fuse_event_recall.py",
            "eng_loop_fuse_carry_dribble.py",
            "eng_loop_fuse_shot_recovery.py",
        )
    )

    n_pass, n_dribble = len(passes), len(dribbles)
    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_audit_ball_teleports": _score(int(audit["ball_teleport_n"]) >= 10),
        "03_dribble_tp": _score(tp >= 1),
        "04_emit_counts": _score(n_pass == 2 and n_dribble == 1),
        "05_no_emit_on_teleport": _score(emit_on_teleport == 0),
        "06_regression": _score(reg),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "audit": audit,
        "n_pass": n_pass,
        "n_dribble": n_dribble,
        "dribble_tp": tp,
        "emit_on_teleport": emit_on_teleport,
        "emits": emits,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all fuse_teleport_mask >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
