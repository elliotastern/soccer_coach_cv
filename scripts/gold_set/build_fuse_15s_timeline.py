#!/usr/bin/env python3
"""Build product fuse timeline for check_15s_s4 (no video)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/gold_set/build_check25_event_timeline.py"),
        "--start",
        "2390",
        "--match-sec",
        "15",
        "--stride",
        "1",
        "--out-dir",
        str(OUT),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
