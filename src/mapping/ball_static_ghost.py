"""Suppress static high-conf solo ball emits (ballcap / junk on grass).

Multi-cam agree is never suppressed / never decayed.

Solo dwell within STATIC_SOLO_M:
  1) conf fades with dwell (never below DWELL_CONF_FLOOR) — fails product emit
     once under EMIT_CONF (~0.80) without touching the global emit gate
  2) after STATIC_SOLO_FRAMES, lock a ghost zone so hold cannot revive it

Moving solos (> STATIC_SOLO_M) reset dwell. Regular moving / agree balls unaffected.
"""
from __future__ import annotations

import math
from typing import Any

from src.mapping.match3_xy import EMIT_CONF

# Pitch meters: absorb foot-map jitter so streak does not reset.
STATIC_SOLO_M = 3.0
# Hard lock after this source-frame span (~0.2 s @ 60 fps).
STATIC_SOLO_FRAMES = 12
# Soft fade: half-life ~6 source frames → typical 0.95 solo fails emit by ~span 5–7.
DWELL_HALF_LIFE_FR = 6.0
# Never drive conf to 0 (debug / ranking); floor stays below EMIT_CONF so product drops.
DWELL_CONF_FLOOR = 0.40
# Only fade/lock solos from these cams (P7 ballcaps). Other cams keep moving/edge
# solos — A/B 2026-09-04: all-cam ghost killed strip clear_R (P10 1.0→0.60, P8 1.0→0.17).
GHOST_STRICT_CAMS = frozenset({"P7"})
# Ballcaps scrape hull support ~0.43; real P7 balls sit ≥0.50. Kickoff autopsy
# 2026-09-04: ghosting high-support static P7 solos killed ball_frac (0.90 misses).
GHOST_MAX_SUPPORT = 0.50


def _xy(emit: dict) -> tuple[float, float] | None:
    xy = emit.get("xy")
    if xy is None or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1])


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _near(a: tuple[float, float], b: tuple[float, float], m: float = STATIC_SOLO_M) -> bool:
    return _dist(a, b) <= float(m)


def dwell_conf_scale(span: int, half_life_fr: float = DWELL_HALF_LIFE_FR) -> float:
    """Multiplier in (floor_scale, 1]: span 0 → 1; decays toward DWELL_CONF_FLOOR/EMIT ratio."""
    span = max(0, int(span))
    if half_life_fr <= 0:
        return 1.0
    # exp decay of the room above floor: scale = floor + (1-floor)*0.5^(span/hl)
    floor_s = float(DWELL_CONF_FLOOR) / float(EMIT_CONF) if EMIT_CONF > 0 else 0.5
    floor_s = min(max(floor_s, 0.0), 0.99)
    return floor_s + (1.0 - floor_s) * math.pow(0.5, float(span) / float(half_life_fr))


def new_static_ghost_state() -> dict[str, Any]:
    return {
        "solo_xy": None,
        "solo_cam": None,
        "solo_start_fr": None,
        "solo_last_fr": None,
        "ghosts": [],  # list[{"xy": (x,y), "cam": str}]
    }


def is_static_ghost_xy(state: dict | None, xy: tuple[float, float], m: float = STATIC_SOLO_M) -> bool:
    if not state:
        return False
    for g in state.get("ghosts") or []:
        gxy = g.get("xy")
        if gxy is not None and _near(xy, (float(gxy[0]), float(gxy[1])), m):
            return True
    return False


def filter_maps_not_static_ghost(rows: list[dict], state: dict | None) -> list[dict]:
    """Drop mapped ball rows that sit on a locked static-ghost zone."""
    if not state or not rows:
        return rows
    kept = []
    for r in rows:
        xy = r.get("xy")
        if xy is None or len(xy) < 2:
            kept.append(r)
            continue
        if is_static_ghost_xy(state, (float(xy[0]), float(xy[1]))):
            continue
        kept.append(r)
    return kept if kept else rows


def _lock_ghost(st: dict, xy: tuple[float, float], cam: str) -> None:
    st["ghosts"].append({"xy": xy, "cam": cam})
    st["ghosts"] = st["ghosts"][-32:]
    st["solo_xy"] = None
    st["solo_cam"] = None
    st["solo_start_fr"] = None
    st["solo_last_fr"] = None


def _gate_solo_dwell(
    emit: dict,
    st: dict,
    *,
    anchor: tuple[float, float],
    span: int,
    static_frames: int,
) -> tuple[dict | None, dict]:
    """Apply fade + hard lock for a solo (or hold) sitting on a dwell cell."""
    if span >= int(static_frames):
        _lock_ghost(st, anchor, str(emit.get("cam") or ""))
        return None, st
    raw = float(emit.get("conf") or 0.0)
    scale = dwell_conf_scale(span)
    conf_eff = max(float(DWELL_CONF_FLOOR), raw * scale)
    if conf_eff < float(EMIT_CONF):
        # Soft drop this frame; keep dwell so fade / lock continue (no hold revive).
        return None, st
    out = {**emit, "conf": conf_eff, "dwell_span": span, "dwell_scale": round(scale, 4)}
    return out, st


def apply_static_solo_ghost(
    emit: dict | None,
    state: dict | None,
    frame_id: int,
    *,
    static_m: float = STATIC_SOLO_M,
    static_frames: int = STATIC_SOLO_FRAMES,
) -> tuple[dict | None, dict]:
    """Prefer multi-cam agree; fade then ghost static high-conf solos.

    Returns (emit_or_none, updated_state).
    """
    st = dict(state or new_static_ghost_state())
    st["ghosts"] = list(st.get("ghosts") or [])
    fr = int(frame_id)

    if emit is None:
        return None, st

    xy = _xy(emit)
    if xy is None:
        return emit, st

    # Multi-cam agree always wins — never fade / ghost; clear local dwell.
    if bool(emit.get("agree")):
        st["solo_xy"] = None
        st["solo_cam"] = None
        st["solo_start_fr"] = None
        st["solo_last_fr"] = None
        return emit, st

    cam = str(emit.get("cam") or "")
    # Non-strict cams (e.g. P8/P10 edge solos): never fade or lock.
    if GHOST_STRICT_CAMS and cam not in GHOST_STRICT_CAMS:
        st["solo_xy"] = None
        st["solo_cam"] = None
        st["solo_start_fr"] = None
        st["solo_last_fr"] = None
        return emit, st

    # High-hull P7 solos are real balls (often slow/static at kickoff) — never fade.
    support = emit.get("support")
    if support is not None and float(support) >= float(GHOST_MAX_SUPPORT):
        st["solo_xy"] = None
        st["solo_cam"] = None
        st["solo_start_fr"] = None
        st["solo_last_fr"] = None
        return emit, st

    # Already locked ghost zone (incl. hold revival).
    if is_static_ghost_xy(st, xy, static_m):
        return None, st

    prev_xy = st.get("solo_xy")
    if prev_xy is not None:
        prev_xy = (float(prev_xy[0]), float(prev_xy[1]))

    # Same dwell cell (jitter / cam switch within static_m): extend span.
    if prev_xy is not None and _near(xy, prev_xy, static_m):
        start = int(st.get("solo_start_fr") if st.get("solo_start_fr") is not None else fr)
        anchor = prev_xy
        st["solo_last_fr"] = fr
        st["solo_xy"] = anchor
        st["solo_cam"] = cam or str(st.get("solo_cam") or "")
        st["solo_start_fr"] = start
        span = fr - start
        return _gate_solo_dwell(
            emit, st, anchor=anchor, span=span, static_frames=static_frames
        )

    # Far from dwell: real motion (or new false positive elsewhere).
    st["solo_xy"] = xy
    st["solo_cam"] = cam
    st["solo_start_fr"] = fr
    st["solo_last_fr"] = fr
    # span 0 — full conf on first sighting of a new cell
    return _gate_solo_dwell(emit, st, anchor=xy, span=0, static_frames=static_frames)
