#!/usr/bin/env python3
"""Eng-loop: rerender coach 15s clip with current event emits."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/coach_clip_events_sync"
CLIP_DIR = ROOT / "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4"
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"
PASS = 9.0


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def _carry_emits(emits: list[dict]) -> list[dict]:
    return [
        e
        for e in emits
        if 10.5 <= float(e.get("t_end", -1)) <= 11.5
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts/gold_set/render_phase1_check_mosaic.py"),
            "--start",
            "2390",
            "--match-sec",
            "15",
            "--stride",
            "4",
            "--out-fps",
            "15",
            "--out-dir",
            str(CLIP_DIR.relative_to(ROOT)),
            "--out-file",
            "coach_mosaic_pitch_min.mp4",
        ],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_phase1_handover_dashboard.py")],
        cwd=str(ROOT),
    )

    meta = json.loads((CLIP_DIR / "meta.json").read_text(encoding="utf-8"))
    emits = meta.get("emits") or []
    carry = _carry_emits(emits)
    dribble_carry = [e for e in carry if e.get("type") == "dribble"]
    movement_carry = [e for e in carry if e.get("type") == "movement"]

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in ("eng_loop_heuristic_events.py", "eng_loop_fuse_dribble_linked.py")
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_regression": _score(reg),
        "03_dribble_carry": _score(len(dribble_carry) >= 1),
        "04_no_movement_carry": _score(len(movement_carry) == 0),
        "05_handover": _score(
            (HANDOVER / "index.html").is_file()
            and (HANDOVER / "coach_mosaic_pitch_min.mp4").is_file()
        ),
        "06_emit_count": _score(len(emits) == 3, max(1.0, 10.0 - abs(len(emits) - 3))),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "n_emits": len(emits),
        "emits": emits,
        "carry": carry,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all coach_clip_events_sync >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
