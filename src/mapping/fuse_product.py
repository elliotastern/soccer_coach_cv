"""Product ball fuse router: pitch_merge (F0-F3) vs triangulate_3d + optional UKF."""
from __future__ import annotations

from src.mapping.ball_static_ghost import (
    apply_static_solo_ghost,
    filter_maps_not_static_ghost,
    is_static_ghost_xy,
    new_static_ghost_state,
)
from src.mapping.ball_ukf import BallPitchUKF
from src.mapping.fuse3d_ball import fuse_balls_3d
from src.mapping.fuse_config import load_fuse_config
from src.mapping.match3_xy import (
    GHOST_CONF,
    HOLD_MAX_GAP,
    fuse_balls,
    fuse_balls_with_hold,
)


_HOLD_ONLY_KEYS = (
    "soft_hold_renew",
    "soft_hold_min_conf",
    "soft_hold_min_support",
)


def _pitch_merge_kw(reproj: dict | None = None) -> dict:
    return dict(
        soft_dual_fallback=True,
        solo_max_conf=True,
        ghost_prune=True,
        ghost_conf=GHOST_CONF,
        reproj_prune=False,
        soft_hold_renew=True,
        soft_hold_min_conf=0.55,
        soft_hold_min_support=0.50,
    )


def _fuse_only_kw(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if k not in _HOLD_ONLY_KEYS}


def pick_3d_hybrid(fresh_3d: dict | None, fresh_2d: dict | None) -> dict | None:
    """Prefer 3D when it multi-cam agrees; else product F0-F3; else 3D solo."""
    if fresh_3d is None:
        return fresh_2d
    if fresh_2d is None:
        return fresh_3d
    if fresh_3d.get("agree"):
        return fresh_3d
    if fresh_2d.get("agree"):
        return {**fresh_2d, "fuse_mode": "triangulate_3d+f0f3_fallback"}
    return {**fresh_2d, "fuse_mode": "triangulate_3d+f0f3_fallback"}


def fuse_ball_product(
    rows: list[dict],
    prev_emit: dict | None,
    frames_since_emit: int,
    *,
    cfg: dict | None = None,
    ukf: BallPitchUKF | None = None,
    hold_max_gap: int = HOLD_MAX_GAP,
    frame_id: int = 0,
    static_state: dict | None = None,
) -> tuple[dict | None, dict | None, int, BallPitchUKF | None, dict]:
    """Return (emit, prev_emit, frames_since_emit, ukf, static_state).

    Prefer multi-cam agree (existing F0–F3). Static high-conf **P7** solos that do not
    move for STATIC_SOLO_FRAMES are dropped as ghosts (ballcap / junk). Other cams'
    solos are not faded (GHOST_STRICT_CAMS).
    """
    cfg = cfg or load_fuse_config()
    mode = str(cfg.get("mode", "pitch_merge"))
    reproj = cfg.get("reproj_max_px") or {}
    kw = _pitch_merge_kw(reproj)
    fuse_kw = _fuse_only_kw(kw)
    st = static_state if static_state is not None else new_static_ghost_state()
    rows = filter_maps_not_static_ghost(rows, st)

    def _gate(emit: dict | None) -> tuple[dict | None, dict]:
        return apply_static_solo_ghost(emit, st, frame_id)

    def _clear_ghosted_prev(prev: dict | None) -> dict | None:
        """Drop prev when it sits on a locked ghost so other cams can recover."""
        if prev is None:
            return None
        xy = prev.get("xy")
        if xy is None or len(xy) < 2:
            return prev
        if is_static_ghost_xy(st, (float(xy[0]), float(xy[1]))):
            return None
        return prev

    if mode == "triangulate_3d":
        fresh_3d = fuse_balls_3d(rows, reproj_overrides=reproj)
        fresh_2d = fuse_balls(rows, **fuse_kw) if cfg.get("fallback_pitch_merge", True) else None
        fresh = pick_3d_hybrid(fresh_3d, fresh_2d)
        if cfg.get("ukf_enabled"):
            if ukf is None:
                ukf = BallPitchUKF()
            emit = ukf.step(fresh, hold_max_gap=hold_max_gap)
            emit, st = _gate(emit)
            if emit is not None:
                return emit, emit, 0, ukf, st
            return None, _clear_ghosted_prev(prev_emit), frames_since_emit + 1, ukf, st
        if fresh is not None:
            fresh, st = _gate(fresh)
            if fresh is not None:
                return fresh, fresh, 0, ukf, st
        gap = frames_since_emit + 1
        if cfg.get("fallback_pitch_merge", True):
            held = fuse_balls_with_hold(prev_emit, rows, gap, hold_max_gap=hold_max_gap, **kw)
            if held is not None:
                held = {**held, "fuse_mode": "triangulate_3d+f0f3_hold"}
                held, st = _gate(held)
                if held is not None:
                    return held, prev_emit, gap, ukf, st
        if prev_emit is not None and gap <= hold_max_gap:
            held = {**prev_emit, "hold": True, "gap": gap}
            held, st = _gate(held)
            if held is not None:
                return held, prev_emit, gap, ukf, st
        return None, _clear_ghosted_prev(prev_emit), gap, ukf, st

    fresh = fuse_balls(rows, **fuse_kw)
    if fresh is not None:
        fresh, st = _gate(fresh)
        if fresh is not None:
            return fresh, fresh, 0, ukf, st
    gap = frames_since_emit + 1
    # Pass current maps so optional soft_hold_renew can confirm past HOLD_MAX_GAP.
    held = fuse_balls_with_hold(prev_emit, rows, gap, hold_max_gap=hold_max_gap, **kw)
    if held is not None:
        held, st = _gate(held)
        if held is not None:
            return held, prev_emit, gap, ukf, st
    return None, _clear_ghosted_prev(prev_emit), gap, ukf, st
