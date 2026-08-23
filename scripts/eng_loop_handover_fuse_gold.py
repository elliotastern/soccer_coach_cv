#!/usr/bin/env python3
"""Eng-loop: merge coach handover QA into fuse event gold."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/handover_fuse_gold"
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"
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
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/build_phase1_handover_dashboard.py")],
        cwd=str(ROOT),
    )

    from scripts.gold_set.merge_handover_fuse_gold import merge_handover_labels  # noqa: E402

    labels = json.loads((HANDOVER / "labels.json").read_text(encoding="utf-8"))
    n_suggested = len(labels.get("suggested_events") or [])

    base = {
        "events": [
            {"type": "pass", "t_start": 1.5, "t_end": 2.0},
            {"type": "pass", "t_start": 4.8, "t_end": 5.5},
            {"type": "dribble", "t_start": 10.5, "t_end": 11.5},
        ]
    }
    emits = json.loads((HANDOVER / "emits_render.json").read_text(encoding="utf-8")).get(
        "emits", []
    )
    fixture_frames = {}
    for em in emits:
        fid = int(em["frame_id"])
        fixture_frames[f"fr_{fid}"] = {
            "event_ok": "good",
            "reviewed": True,
            "ball_visible": "yes",
            "pitch_ball_ok": "good",
            "team_ok": "unset",
            "flag": False,
            "note": "fixture confirm",
        }
    fixture_handover = {
        "reviewer": "eng_loop_fixture",
        "frames": fixture_frames,
        "suggested_events": labels.get("suggested_events") or [],
    }
    with tempfile.TemporaryDirectory() as td:
        fix_dir = Path(td) / "handover"
        fix_dir.mkdir()
        shutil.copy2(HANDOVER / "emits_render.json", fix_dir / "emits_render.json")
        (fix_dir / "labels.json").write_text(json.dumps(fixture_handover, indent=2))
        merged_fixture = merge_handover_labels(fix_dir, base)
    coach_sourced = sum(
        1 for e in merged_fixture.get("events") or [] if e.get("source") == "handover"
    )

    reg = all(
        subprocess.run(
            [sys.executable, str(ROOT / f"scripts/{s}")],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for s in ("eng_loop_fuse_event_recall.py", "eng_loop_fuse_shot_recovery.py")
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_suggested_events": _score(n_suggested >= 3, max(1.0, 10.0 * n_suggested / 3)),
        "03_fixture_coach_events": _score(coach_sourced >= 3),
        "04_fuse_recall": _score(reg),
        "05_labels_merged": _score((CLIP / "labels_merged.json").is_file()),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "n_suggested": n_suggested,
        "coach_sourced_fixture": coach_sourced,
        "merged_events": len(merged_fixture.get("events") or []),
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all handover_fuse_gold >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
