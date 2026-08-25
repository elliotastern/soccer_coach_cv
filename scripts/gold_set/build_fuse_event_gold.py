#!/usr/bin/env python3
"""Labels for real_fuse_15s carry windows."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
sys.path.insert(0, str(ROOT))

from scripts.gold_set.build_fuse_linked_timeline import build_linked_timeline  # noqa: E402


def main() -> int:
    CLIP.mkdir(parents=True, exist_ok=True)
    build_linked_timeline()
    labels = {
        "source": "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4",
        "timeline_primary": "timeline_linked.json",
        "start_frame": 2390,
        "stride": 4,
        "match_sec": 15.0,
        "events": [
            {
                "type": "pass",
                "t_start": 1.5,
                "t_end": 2.0,
                "note": "High-speed leave Goal2 — pass not dribble",
            },
            {
                "type": "pass",
                "t_start": 4.8,
                "t_end": 5.5,
                "note": "Second pass window",
            },
            {
                "type": "dribble",
                "t_start": 10.5,
                "t_end": 11.5,
                "expected_carrier_pid": 42,
                "note": "team_core linked fuse id (stride-4 window)",
            },
        ],
        "negatives": [
            {"t_start": 0.0, "t_end": 1.0, "note": "Open play setup — no dribble"},
        ],
    }
    (CLIP / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (CLIP / "note.txt").write_text(
        "Product fuse 15s labels for fuse_event_recall eng-loop.\n",
        encoding="utf-8",
    )
    print("WROTE", CLIP / "labels.json")
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/gold_set/merge_handover_fuse_gold.py")],
        cwd=str(ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
