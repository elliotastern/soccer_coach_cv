#!/usr/bin/env python3
"""Unit checks for Pitch 1 heuristic events (E0)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events.events import EMIT_CONF, PITCH1_HALF_LENGTH_M, EventDetector
from src.state.types import Ball, EventType, FrameData, Player


def _player(pid: int, x: float, y: float, fr: int, t: float) -> Player:
    return Player(pid, 0, x, y, (0, 0, 10, 10), fr, t)


def _ball(x: float, y: float, fr: int, t: float) -> Ball:
    return Ball(x, y, (0, 0, 4, 4), fr, t)


def _pair(p0, b0, p1, b1, t0=0.0, dt=1.0 / 30.0):
    prev = FrameData(0, t0, p0, b0)
    cur = FrameData(1, t0 + dt, p1, b1)
    return prev, cur


def test_no_fifa_52_5():
    assert abs(PITCH1_HALF_LENGTH_M - 26.95) < 1e-6
    det = EventDetector()
    assert abs(det.half_length_m - 26.95) < 1e-6
    assert det.emit_conf >= 0.80


def test_shot_pitch1_goal_band():
    det = EventDetector()
    dt = 0.15
    prev0 = FrameData(
        0, 5.0,
        [_player(1, 18.0, 0.0, 0, 5.0)],
        _ball(18.0, 0.0, 0, 5.0),
    )
    cur0 = FrameData(
        1, 5.0 + dt,
        [_player(1, 18.0, 0.0, 1, 5.0 + dt)],
        _ball(20.0, 0.0, 1, 5.0 + dt),
    )
    assert det.detect_events(cur0, prev0) == []
    prev1 = cur0
    cur1 = FrameData(
        2, 5.0 + 2 * dt,
        [_player(1, 18.0, 0.0, 2, 5.0 + 2 * dt)],
        _ball(23.0, 0.0, 2, 5.0 + 2 * dt),
    )
    evs = det.detect_events(cur1, prev1)
    assert len(evs) == 1 and evs[0].type == EventType.SHOT
    assert evs[0].confidence >= EMIT_CONF


def test_shot_rejects_midfield():
    det = EventDetector()
    dt = 0.2
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(0.0, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(5.0, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_shot(cur, prev) is None


def test_weak_pass_below_emit():
    det = EventDetector()
    # 5.5 m/s pass → conf 5.5/20 < 0.80 → gated
    dt = 1.0
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(0.0, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(5.5, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_pass(cur, prev) is None


def test_strong_pass_emits():
    det = EventDetector()
    dt = 1.0
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(0.0, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(18.0, 0.0, 1, dt),
        dt=dt,
    )
    ev = det.detect_pass(cur, prev)
    assert ev is not None and ev.confidence >= EMIT_CONF


def test_priority_shot_over_pass():
    det = EventDetector()
    dt = 0.15
    prev0 = FrameData(
        0, 5.0,
        [_player(1, 18.0, 0.0, 0, 5.0)],
        _ball(18.0, 0.0, 0, 5.0),
    )
    cur0 = FrameData(
        1, 5.0 + dt,
        [_player(1, 18.0, 0.0, 1, 5.0 + dt)],
        _ball(20.0, 0.0, 1, 5.0 + dt),
    )
    det.detect_events(cur0, prev0)
    prev1 = cur0
    cur1 = FrameData(
        2, 5.0 + 2 * dt,
        [_player(1, 18.0, 0.0, 2, 5.0 + 2 * dt)],
        _ball(23.0, 0.0, 2, 5.0 + 2 * dt),
    )
    evs = det.detect_events(cur1, prev1)
    assert len(evs) == 1 and evs[0].type == EventType.SHOT


def test_kickoff_shot_rejected():
    """Early glitch shot (e.g. mosaic @1.5s) blocked by kickoff floor."""
    det = EventDetector()
    dt = 0.25
    prev0 = FrameData(
        0, 1.0,
        [_player(1, 5.0, 0.0, 0, 1.0)],
        _ball(5.0, 0.0, 0, 1.0),
    )
    cur0 = FrameData(
        1, 1.0 + dt,
        [_player(1, 5.0, 0.0, 1, 1.0 + dt)],
        _ball(8.0, 0.0, 1, 1.0 + dt),
    )
    det.detect_events(cur0, prev0)
    prev1 = cur0
    cur1 = FrameData(
        2, 1.5,
        [_player(1, 5.0, 0.0, 2, 1.5)],
        _ball(24.0, 0.0, 2, 1.5),
    )
    assert det.detect_events(cur1, prev1) == []


def test_shot_rejects_wide_of_goal_mouth():
    det = EventDetector()
    dt = 0.2
    prev0 = FrameData(
        0, 5.0,
        [_player(1, 18.0, 8.0, 0, 5.0)],
        _ball(18.0, 8.0, 0, 5.0),
    )
    cur0 = FrameData(
        1, 5.0 + dt,
        [_player(1, 18.0, 8.0, 1, 5.0 + dt)],
        _ball(21.0, 8.0, 1, 5.0 + dt),
    )
    det.detect_events(cur0, prev0)
    prev1 = cur0
    cur1 = FrameData(
        2, 5.0 + 2 * dt,
        [_player(1, 18.0, 8.0, 2, 5.0 + 2 * dt)],
        _ball(24.0, 8.0, 2, 5.0 + 2 * dt),
    )
    assert det.detect_events(cur1, prev1) == []


def test_teleport_rejected():
    det = EventDetector()
    dt = 0.25
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(-16.0, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(25.0, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_events(cur, prev) == []


def test_weak_dribble_suppressed():
    det = EventDetector(enable_dribble=True)
    dt = 1 / 30
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(0.1, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(0.2, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_dribble(cur, prev) is None


def test_dribble_needs_prev_proximity():
    det = EventDetector(enable_dribble=True)
    dt = 1.0
    prev, cur = _pair(
        [_player(1, 5.0, 0.0, 0, 0.0)],
        _ball(0.5, 0.0, 0, 0.0),
        [_player(1, 0.2, 0.0, 1, dt)],
        _ball(0.9, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_dribble(cur, prev) is None


def test_dribble_emits():
    det = EventDetector(enable_dribble=True)
    dt = 0.33
    steps = [
        ([_player(1, 0.0, 0.0, 0, 0.0)], _ball(0.0, 0.0, 0, 0.0)),
        ([_player(1, 0.18, 0.0, 1, dt)], _ball(0.25, 0.0, 1, dt)),
        ([_player(1, 0.36, 0.0, 2, 2 * dt)], _ball(0.55, 0.0, 2, 2 * dt)),
        ([_player(1, 0.54, 0.0, 3, 3 * dt)], _ball(0.95, 0.0, 3, 3 * dt)),
    ]
    prev = None
    ev = None
    for i, (players, ball) in enumerate(steps):
        cur = FrameData(i, i * dt, players, ball)
        if prev is not None:
            evs = det.detect_events(cur, prev)
            if evs:
                ev = evs[0]
        prev = cur
    assert ev is not None and ev.type == EventType.DRIBBLE
    assert ev.confidence >= EMIT_CONF


def test_movement_emits():
    det = EventDetector(enable_movement=True)
    dt = 1.0
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 5.0)],
        _ball(0.0, 0.0, 0, 5.0),
        [_player(1, 0.5, 0.0, 1, 5.0 + dt)],
        _ball(2.5, 0.0, 1, 5.0 + dt),
        t0=5.0,
        dt=dt,
    )
    ev = det.detect_movement(cur, prev)
    assert ev is not None and ev.type == EventType.MOVEMENT
    assert ev.confidence >= EMIT_CONF


def test_goal_jitter_no_dribble_or_movement():
    """Static player + ball jitter (fusion noise) must not emit E2 types."""
    det = EventDetector(enable_dribble=True, enable_movement=True)
    dt = 0.25
    prev, cur = _pair(
        [_player(1, 24.0, 2.0, 0, 0.0)],
        _ball(25.0, 2.5, 0, 0.0),
        [_player(1, 24.0, 2.0, 1, dt)],
        _ball(25.4, 2.7, 1, dt),
        dt=dt,
    )
    assert det.detect_dribble(cur, prev) is None
    assert det.detect_movement(cur, prev) is None


def test_movement_below_pass_threshold():
    det = EventDetector(enable_movement=True, enable_dribble=False)
    dt = 1.0
    prev, cur = _pair(
        [_player(1, 0.0, 0.0, 0, 0.0)],
        _ball(0.0, 0.0, 0, 0.0),
        [_player(1, 0.0, 0.0, 1, dt)],
        _ball(18.0, 0.0, 1, dt),
        dt=dt,
    )
    assert det.detect_movement(cur, prev) is None


def main() -> int:
    tests = [
        test_no_fifa_52_5,
        test_shot_pitch1_goal_band,
        test_shot_rejects_midfield,
        test_weak_pass_below_emit,
        test_strong_pass_emits,
        test_priority_shot_over_pass,
        test_kickoff_shot_rejected,
        test_shot_rejects_wide_of_goal_mouth,
        test_weak_dribble_suppressed,
        test_dribble_needs_prev_proximity,
        test_dribble_emits,
        test_movement_emits,
        test_goal_jitter_no_dribble_or_movement,
        test_movement_below_pass_threshold,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all_unit_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
