#!/usr/bin/env python3
"""Persist stable-id fuse timeline for event gold scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.timeline_player_ids import relink_timeline_players  # noqa: E402

CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"


def build_linked_timeline(clip_dir: Path = CLIP) -> dict:
    raw_path = clip_dir / "timeline.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing fuse timeline: {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    linked = relink_timeline_players(raw)
    out_path = clip_dir / "timeline_linked.json"
    out_path.write_text(json.dumps(linked, indent=2), encoding="utf-8")
    return linked


def main() -> int:
    linked = build_linked_timeline()
    print("WROTE", CLIP / "timeline_linked.json", "frames=", len(linked.get("frames") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
