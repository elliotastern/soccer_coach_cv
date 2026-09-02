"""Product ball fuse router: pitch_merge (F0-F3) vs triangulate_3d + optional UKF."""
from __future__ import annotations

from src.mapping.ball_ukf import BallPitchUKF
from src.mapping.fuse3d_ball import fuse_balls_3d
from src.mapping.fuse_config import load_fuse_config
from src.mapping.match3_xy import (
    GHOST_CONF,
    HOLD_MAX_GAP,
    fuse_balls,
    fuse_balls_with_hold,
)


def _pitch_merge_kw(reproj: dict | None = None) -> dict:
    return dict(
        soft_dual_fallback=True,
        solo_max_conf=True,
        ghost_prune=True,
        ghost_conf=GHOST_CONF,
        reproj_prune=False,
    )


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
) -> tuple[dict | None, dict | None, int, BallPitchUKF | None]:
    """Return (emit, prev_emit, frames_since_emit, ukf)."""
    cfg = cfg or load_fuse_config()
    mode = str(cfg.get("mode", "pitch_merge"))
    reproj = cfg.get("reproj_max_px") or {}
    kw = _pitch_merge_kw(reproj)

    if mode == "triangulate_3d":
        fresh_3d = fuse_balls_3d(rows, reproj_overrides=reproj)
        fresh_2d = fuse_balls(rows, **kw) if cfg.get("fallback_pitch_merge", True) else None
        fresh = pick_3d_hybrid(fresh_3d, fresh_2d)
        if cfg.get("ukf_enabled"):
            if ukf is None:
                ukf = BallPitchUKF()
            emit = ukf.step(fresh, hold_max_gap=hold_max_gap)
            if emit is not None:
                return emit, emit, 0, ukf
            return None, prev_emit, frames_since_emit + 1, ukf
        if fresh is not None:
            return fresh, fresh, 0, ukf
        gap = frames_since_emit + 1
        if cfg.get("fallback_pitch_merge", True):
            held = fuse_balls_with_hold(prev_emit, rows, gap, hold_max_gap=hold_max_gap, **kw)
            if held is not None:
                held = {**held, "fuse_mode": "triangulate_3d+f0f3_hold"}
                return held, prev_emit, gap, ukf
        if prev_emit is not None and gap <= hold_max_gap:
            held = {**prev_emit, "hold": True, "gap": gap}
            return held, prev_emit, gap, ukf
        return None, prev_emit, gap, ukf

    fresh = fuse_balls(rows, **kw)
    if fresh is not None:
        return fresh, fresh, 0, ukf
    gap = frames_since_emit + 1
    held = fuse_balls_with_hold(prev_emit, [], gap, hold_max_gap=hold_max_gap, **kw)
    if held is not None:
        return held, prev_emit, gap, ukf
    return None, prev_emit, gap, ukf
