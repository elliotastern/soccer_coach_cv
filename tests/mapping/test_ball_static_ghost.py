"""Tests for static high-conf solo ball ghost suppression."""
from __future__ import annotations

from src.mapping.ball_static_ghost import (
    DWELL_CONF_FLOOR,
    STATIC_SOLO_FRAMES,
    STATIC_SOLO_M,
    apply_static_solo_ghost,
    dwell_conf_scale,
    new_static_ghost_state,
)
from src.mapping.fuse_product import fuse_ball_product
from src.mapping.match3_xy import EMIT_CONF


def _solo(xy=(5.0, 5.0), conf=0.95, cam="P7", support=0.43):
    return {
        "xy": xy,
        "conf": conf,
        "weight": conf * support,
        "support": support,
        "cam": cam,
        "agree": False,
    }


def _agree(xy=(5.0, 5.0), conf=0.90):
    return {
        "xy": xy,
        "conf": conf,
        "agree": True,
        "cams": ["P10", "P7"],
        "n_agree": 2,
    }


def test_dwell_scale_never_zero():
    assert dwell_conf_scale(0) == 1.0
    s_long = dwell_conf_scale(10_000)
    assert s_long > 0
    assert 0.95 * s_long >= DWELL_CONF_FLOOR - 1e-6 or s_long * EMIT_CONF >= DWELL_CONF_FLOOR - 1e-6


def test_fade_drops_solo_before_hard_lock():
    st = new_static_ghost_state()
    saw_emit = False
    dropped_before_lock = False
    for fr in range(STATIC_SOLO_FRAMES):
        emit, st = apply_static_solo_ghost(_solo(), st, fr)
        if emit is not None:
            saw_emit = True
            assert float(emit["conf"]) >= EMIT_CONF
        else:
            dropped_before_lock = True
            assert not st["ghosts"], "fade drop must not lock yet"
            break
    assert saw_emit, "first frames should still emit"
    assert dropped_before_lock, "fade should fail emit before hard lock"


def test_hard_lock_after_static_frames():
    st = new_static_ghost_state()
    for fr in range(STATIC_SOLO_FRAMES + 1):
        emit, st = apply_static_solo_ghost(_solo(), st, fr)
    assert st["ghosts"], "ghost zone locked"
    emit, st = apply_static_solo_ghost(_solo(), st, STATIC_SOLO_FRAMES + 5)
    assert emit is None


def test_jitter_within_radius_does_not_reset():
    st = new_static_ghost_state()
    _, st = apply_static_solo_ghost(_solo((5.0, 5.0)), st, 0)
    jitter = (5.0 + min(2.0, STATIC_SOLO_M - 0.1), 5.0)
    emit = None
    for fr in range(1, STATIC_SOLO_FRAMES + 1):
        emit, st = apply_static_solo_ghost(_solo(jitter, cam="P7"), st, fr)
    assert emit is None
    assert st["ghosts"]


def test_moving_solo_resets_streak():
    st = new_static_ghost_state()
    for fr in range(4):
        emit, st = apply_static_solo_ghost(_solo((5.0, 5.0)), st, fr)
    emit, st = apply_static_solo_ghost(_solo((5.0 + STATIC_SOLO_M + 1.0, 5.0)), st, 4)
    assert emit is not None
    assert st["solo_start_fr"] == 4
    assert not st["ghosts"]
    assert float(emit["conf"]) >= EMIT_CONF


def test_agree_never_ghosted_or_faded():
    st = new_static_ghost_state()
    for fr in range(STATIC_SOLO_FRAMES + 5):
        emit, st = apply_static_solo_ghost(_agree(conf=0.90), st, fr)
        assert emit is not None
        assert emit.get("agree")
        assert float(emit["conf"]) == 0.90
    assert not st["ghosts"]


def test_fuse_product_prefers_agree_over_static_solo():
    cfg = {"mode": "pitch_merge"}
    rows = [
        {"xy": (0.0, 0.0), "conf": 0.95, "weight": 0.95, "support": 0.95, "cam": "P10"},
        {"xy": (0.4, 0.2), "conf": 0.90, "weight": 0.90, "support": 0.90, "cam": "P7"},
    ]
    emit, _, gap, _, st = fuse_ball_product(rows, None, 0, cfg=cfg, frame_id=0)
    assert emit is not None and gap == 0
    assert emit.get("agree") or emit.get("n_agree", 0) >= 2 or len(emit.get("cams") or []) >= 2


def test_fuse_product_ghosts_static_solo_sequence():
    """Low-support P7 ballcap hold revival still fades/locks (solo cascade skips fresh)."""
    cfg = {"mode": "pitch_merge"}
    row = [{"xy": (3.0, 4.0), "conf": 0.95, "weight": 0.40, "support": 0.43, "cam": "P7"}]
    prev = {
        "xy": (3.0, 4.0),
        "conf": 0.95,
        "cam": "P7",
        "n": 1,
        "agree": False,
        "support": 0.43,
    }
    gap = 1
    st = None
    last = None
    for fr in range(STATIC_SOLO_FRAMES + 3):
        last, prev, gap, _, st = fuse_ball_product(
            row, prev, gap, cfg=cfg, frame_id=fr, static_state=st
        )
    assert last is None, "static P7 ballcap hold must drop after streak"
    assert st and st.get("ghosts")


def test_fuse_product_does_not_ghost_high_support_p7_static():
    """Real P7 balls (hull support ≥0.50) keep emitting when static — kickoff fix."""
    cfg = {"mode": "pitch_merge"}
    row = [{"xy": (3.0, 4.0), "conf": 0.95, "weight": 0.95, "support": 1.0, "cam": "P7"}]
    prev = None
    gap = 0
    st = None
    last = None
    for fr in range(STATIC_SOLO_FRAMES + 5):
        last, prev, gap, _, st = fuse_ball_product(
            row, prev, gap, cfg=cfg, frame_id=fr, static_state=st
        )
        assert last is not None
        assert float(last["conf"]) >= EMIT_CONF
    assert not (st or {}).get("ghosts")


def test_fuse_product_clears_ghosted_prev_for_recovery():
    """Ghost-killed hold must clear prev so a soft other-cam map can soft-renew later."""
    cfg = {"mode": "pitch_merge"}
    st = new_static_ghost_state()
    st["ghosts"] = [{"xy": (-20.1, -16.2), "cam": "P7"}]
    prev = {
        "xy": (-20.1, -16.2),
        "conf": 0.92,
        "cam": "P7",
        "n": 1,
        "agree": False,
        "support": 0.43,
    }
    # Soft P10 near a different place — no fresh emit, hold gated → prev cleared.
    rows = [
        {"xy": (1.5, 5.8), "conf": 0.70, "weight": 0.55, "support": 0.78, "cam": "P10"},
    ]
    emit, prev_out, gap, _, st2 = fuse_ball_product(
        rows, prev, 5, cfg=cfg, frame_id=100, static_state=st
    )
    assert emit is None
    assert prev_out is None, "ghosted prev must clear"
    assert gap == 6


def test_fuse_product_does_not_ghost_p10_static_solo():
    """Non-P7 solos keep emitting (strip clear_R regression fix)."""
    cfg = {"mode": "pitch_merge"}
    row = [{"xy": (3.0, 4.0), "conf": 0.95, "weight": 0.95, "support": 0.95, "cam": "P10"}]
    prev = None
    gap = 0
    st = None
    last = None
    for fr in range(STATIC_SOLO_FRAMES + 5):
        last, prev, gap, _, st = fuse_ball_product(
            row, prev, gap, cfg=cfg, frame_id=fr, static_state=st
        )
        assert last is not None
        assert float(last["conf"]) >= EMIT_CONF
    assert not (st or {}).get("ghosts")


def test_fuse_product_filters_locked_ghost_maps():
    """Locked ghost zone must not seed fuse when a real ball is also mapped."""
    from src.mapping.ball_static_ghost import filter_maps_not_static_ghost

    cfg = {"mode": "pitch_merge"}
    st = new_static_ghost_state()
    st["ghosts"] = [{"xy": (-20.1, -16.2), "cam": "P7"}]
    rows = [
        {"xy": (-20.1, -16.2), "conf": 0.92, "weight": 0.4, "support": 0.43, "cam": "P7"},
        {"xy": (-10.0, 10.0), "conf": 0.88, "weight": 0.88, "support": 1.0, "cam": "P10"},
    ]
    kept = filter_maps_not_static_ghost(rows, st)
    assert [r["cam"] for r in kept] == ["P10"]
    emit, _, _, _, _ = fuse_ball_product(
        rows, None, 0, cfg=cfg, frame_id=0, static_state=st
    )
    assert emit is not None
    assert emit.get("cam") == "P10"
