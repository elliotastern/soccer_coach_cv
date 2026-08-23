"""Relink fuse timeline player ids by sticky pitch xy."""
from __future__ import annotations

import copy

from src.review.team_live import assign_stable_player_ids, STABLE_PID_M


def relink_timeline_players(timeline: dict, sticky_m: float = STABLE_PID_M) -> dict:
  """Return timeline copy with stable fuse player ids (no per-frame j+1 reorder)."""
  tracks: list[dict] = []
  next_id = 1
  out = copy.deepcopy(timeline)
  for row in out.get("frames") or []:
    tuples = []
    for p in row.get("players") or []:
      x, y = float(p[1]), float(p[2])
      team = int(p[3]) if len(p) > 3 else -1
      tuples.append((x, y, team, int(p[0])))
    stable, tracks, next_id = assign_stable_player_ids(
      tuples, tracks, next_id, sticky_m=sticky_m
    )
    row["players"] = [
      [int(pid), float(x), float(y), int(team)]
      for x, y, team, pid in stable
    ]
  return out


def carry_window_id_swaps(
  timeline: dict,
  t_start: float = 10.5,
  t_end: float = 11.5,
  xy_jump_m: float = 2.0,
) -> dict:
  """Count nearest-to-ball id swaps when carrier xy step is small."""
  import math

  prev = None
  swaps = 0
  carriers = []
  for row in timeline.get("frames") or []:
    t = float(row["t"])
    if t < t_start or t > t_end or not row.get("ball"):
      continue
    bx, by = float(row["ball"][0]), float(row["ball"][1])
    best = None
    for p in row.get("players") or []:
      d = math.hypot(float(p[1]) - bx, float(p[2]) - by)
      if best is None or d < best[0]:
        best = (d, int(p[0]), float(p[1]), float(p[2]))
    if best is None:
      continue
    carriers.append(best)
    if prev is not None:
      jump = math.hypot(best[2] - prev[2], best[3] - prev[3])
      if jump < xy_jump_m and best[1] != prev[1]:
        swaps += 1
    prev = best
  ids = [c[1] for c in carriers]
  return {
    "swaps": swaps,
    "unique_ids": len(set(ids)),
    "n_frames": len(carriers),
    "ids": ids,
  }
