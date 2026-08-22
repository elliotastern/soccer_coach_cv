#!/usr/bin/env python3
"""Run EventDetector on a gold timeline → emits.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.events.events import EventDetector
from src.state.types import Ball, FrameData, Player


def frame_from_row(row: dict) -> FrameData:
    players = []
    for p in row.get("players") or []:
        pid, x, y = int(p[0]), float(p[1]), float(p[2])
        players.append(
            Player(pid, 0, x, y, (0, 0, 10, 10), int(row["frame_id"]), float(row["t"]))
        )
    ball = None
    if row.get("ball") is not None:
        bx, by = row["ball"]
        ball = Ball(
            float(bx),
            float(by),
            (0, 0, 4, 4),
            int(row["frame_id"]),
            float(row["t"]),
        )
    return FrameData(int(row["frame_id"]), float(row["t"]), players, ball)


def run_timeline(timeline: dict, det: EventDetector | None = None) -> list[dict]:
    det = det or EventDetector()
    frames = [frame_from_row(r) for r in timeline["frames"]]
    emits = []
    prev = None
    for fr in frames:
        for ev in det.detect_events(fr, prev):
            emits.append(
                {
                    "type": ev.type.value,
                    "frame_end": ev.end_frame,
                    "t_start": ev.timestamp_start,
                    "t_end": ev.timestamp_end,
                    "confidence": ev.confidence,
                    "players": list(ev.involved_players),
                    "start_xy": [ev.start_location.x, ev.start_location.y],
                    "end_xy": [ev.end_location.x, ev.end_location.y],
                }
            )
        prev = fr
    return emits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeline", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    emits = run_timeline(timeline)
    out = args.out or args.timeline.with_name("emits.json")
    out.write_text(json.dumps({"emits": emits}, indent=2), encoding="utf-8")
    print("WROTE", out, "n=", len(emits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
