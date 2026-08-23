#!/usr/bin/env python3
"""Eng-loop: coach bulk confirm suggested events → fuse gold merge."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/coach_handover_confirm"
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
PASS = 9.0


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_phase1_handover_dashboard.py")],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_fuse_event_gold.py")],
        cwd=str(ROOT),
    )

    from scripts.gold_set.confirm_handover_suggested_events import (  # noqa: E402
        confirm_suggested_events,
    )
    from scripts.gold_set.merge_handover_fuse_gold import merge_handover_labels  # noqa: E402

    labels = json.loads((HANDOVER / "labels.json").read_text(encoding="utf-8"))
    n_suggested = len(labels.get("suggested_events") or [])
    backup = labels.copy()

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "handover"
        shutil.copytree(HANDOVER, work)
        confirm_out = confirm_suggested_events(work, reviewer="eng_loop_fixture")
        work_labels = json.loads((work / "labels.json").read_text(encoding="utf-8"))
        n_good_frames = sum(
            1
            for fr in (work_labels.get("frames") or {}).values()
            if fr.get("event_ok") == "good"
        )
        base = json.loads((CLIP / "labels.json").read_text(encoding="utf-8"))
        merged = merge_handover_labels(work, base)
        coach_sourced = sum(
            1 for e in merged.get("events") or [] if e.get("source") == "handover"
        )

    (HANDOVER / "labels.json").write_text(json.dumps(backup, indent=2), encoding="utf-8")

    reg_recall = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_fuse_event_recall.py")],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0
    reg_linked = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_fuse_linked_gold_primary.py")],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0

    ui_ok = "confirmAll" in (HANDOVER / "index.html").read_text(encoding="utf-8")

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_suggested_events": _score(n_suggested >= 3),
        "03_confirm_frames": _score(
            confirm_out["confirmed"] >= 3 and n_good_frames >= 3
        ),
        "04_coach_sourced_merge": _score(coach_sourced >= 3),
        "05_fuse_recall": _score(reg_recall),
        "06_linked_regression": _score(reg_linked and ui_ok),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "n_suggested": n_suggested,
        "confirm_out": confirm_out,
        "n_good_frames": n_good_frames,
        "coach_sourced": coach_sourced,
        "merged_events": len(merged.get("events") or []),
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all coach_handover_confirm >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
