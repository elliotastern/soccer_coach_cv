#!/usr/bin/env python3
"""Holdout fuse window: check25 pass #3 (outside handover 15s clip)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_holdout_pass"
V2_MANIFEST = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/manifest.json"

# Handover clip origin + third pass label from check25_human (not in first 15s).
ORIG_START = 2390
HOLDOUT_START = 3260  # 14.5 s into match from ORIG_START
MATCH_SEC = 3.0
STRIDE = 4


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/gold_set/build_check25_event_timeline.py"),
        "--start",
        str(HOLDOUT_START),
        "--match-sec",
        str(MATCH_SEC),
        "--stride",
        str(STRIDE),
        "--out-dir",
        str(OUT),
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc

    # Holdout pass window lands ~1.8s at stride 4 (check25 label was stride 15).
    labels = {
        "source": "check25_human pass #3 holdout (outside check_15s_s4)",
        "origin_start_frame": ORIG_START,
        "start_frame": HOLDOUT_START,
        "stride": STRIDE,
        "match_sec": MATCH_SEC,
        "holdout": True,
        "events": [
            {
                "type": "pass",
                "t_start": 1.0,
                "t_end": 2.0,
                "note": "Third Goal2 leave — stride-4 emit ~1.8s (outside handover 15s)",
            }
        ],
        "negatives": [
            {"t_start": 0.0, "t_end": 0.5, "note": "Pre-pass setup"},
        ],
    }
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (OUT / "note.txt").write_text(
        "Holdout fuse window for eng-loop 08c (not handover clip).\n",
        encoding="utf-8",
    )

    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "id": "real_fuse_holdout_pass",
        "timeline": str((OUT / "timeline.json").relative_to(ROOT)),
        "labels": str((OUT / "labels.json").relative_to(ROOT)),
        "note": "Holdout pass window outside check_15s_s4",
        "holdout": True,
    }
    clips = [c for c in manifest.get("clips") or [] if c.get("id") != entry["id"]]
    clips.append(entry)
    manifest["clips"] = clips
    V2_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE", OUT / "labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
