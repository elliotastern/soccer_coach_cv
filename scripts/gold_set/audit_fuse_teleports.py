#!/usr/bin/env python3
"""Count ball / near-ball player teleports on fuse stride-4 timelines."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.timeline_player_ids import relink_timeline_players  # noqa: E402
from src.events.events import MAX_BALL_SPEED_M_S, MAX_PLAYER_STEP_M  # noqa: E402


def audit_timeline(tl: dict, near_m: float = 4.0) -> dict:
    frames = tl.get("frames") or []
    ball_teleports = []
    player_teleports = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        if not a.get("ball") or not b.get("ball"):
            continue
        dt = float(b["t"]) - float(a["t"])
        if dt <= 0:
            continue
        d = math.hypot(b["ball"][0] - a["ball"][0], b["ball"][1] - a["ball"][1])
        speed = d / dt
        if speed > MAX_BALL_SPEED_M_S:
            ball_teleports.append(
                {
                    "t": b["t"],
                    "frame_id": b["frame_id"],
                    "dist_m": round(d, 2),
                    "speed_m_s": round(speed, 1),
                }
            )
        ax, ay = a["ball"][0], a["ball"][1]
        prev_near = {}
        for p in a.get("players") or []:
            pid, x, y = p[0], p[1], p[2]
            if math.hypot(x - ax, y - ay) <= near_m:
                prev_near[pid] = (x, y)
        for p in b.get("players") or []:
            pid, x, y = p[0], p[1], p[2]
            if pid not in prev_near:
                continue
            px, py = prev_near[pid]
            step = math.hypot(x - px, y - py)
            if step > MAX_PLAYER_STEP_M:
                player_teleports.append(
                    {
                        "t": b["t"],
                        "frame_id": b["frame_id"],
                        "player_id": pid,
                        "step_m": round(step, 2),
                    }
                )
    return {
        "ball_teleport_n": len(ball_teleports),
        "player_near_ball_teleport_n": len(player_teleports),
        "ball_teleports": ball_teleports[:20],
        "player_teleports": player_teleports[:20],
    }


def main() -> int:
    clip = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
    raw = json.loads((clip / "timeline.json").read_text(encoding="utf-8"))
    linked = (
        json.loads((clip / "timeline_linked.json").read_text(encoding="utf-8"))
        if (clip / "timeline_linked.json").is_file()
        else None
    )
    out = {
        "raw": audit_timeline(raw),
        "relinked": audit_timeline(relink_timeline_players(raw)),
    }
    if linked:
        out["linked"] = audit_timeline(linked)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
