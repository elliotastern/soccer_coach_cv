# Heuristic event detection (Phase 1) — Pitch 1 meters, emit conf ≥ 0.80.
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.state.types import Ball, Event, EventType, FrameData, Location, Player

EMIT_CONF = 0.80
# Pitch 1 length 53.90 m → half from centre origin (not FIFA half-length).
PITCH1_HALF_LENGTH_M = 53.90 / 2.0
SHOT_GOAL_BAND_M = 5.0
# Reject map teleports / cam switches (real Match 3 fuse jumps).
MAX_BALL_SPEED_M_S = 40.0
# Min time between emits (precision-first; kills frame spam).
MIN_EMIT_GAP_S = 1.0

# Mutual exclusion: first match wins.
_EVENT_PRIORITY = (
    EventType.SHOT,
    EventType.PASS,
    EventType.RECOVERY,
    EventType.DRIBBLE,
)


class EventDetector:
    """Physics / rule-based events on Pitch 1 xy. Below EMIT_CONF → do not emit."""

    def __init__(
        self,
        pitch_mapper=None,
        pass_velocity_threshold: float = 5.0,
        dribble_distance_threshold: float = 2.0,
        shot_velocity_threshold: float = 15.0,
        recovery_proximity: float = 1.0,
        emit_conf: float = EMIT_CONF,
        half_length_m: float = PITCH1_HALF_LENGTH_M,
        shot_goal_band_m: float = SHOT_GOAL_BAND_M,
        max_ball_speed_m_s: float = MAX_BALL_SPEED_M_S,
        min_emit_gap_s: float = MIN_EMIT_GAP_S,
        enable_dribble: bool = False,
    ):
        self.pitch_mapper = pitch_mapper
        self.pass_velocity_threshold = pass_velocity_threshold
        self.dribble_distance_threshold = dribble_distance_threshold
        self.shot_velocity_threshold = shot_velocity_threshold
        self.recovery_proximity = recovery_proximity
        self.emit_conf = emit_conf
        self.half_length_m = half_length_m
        self.shot_goal_band_m = shot_goal_band_m
        self.max_ball_speed_m_s = max_ball_speed_m_s
        self.min_emit_gap_s = min_emit_gap_s
        self.enable_dribble = enable_dribble
        self.player_history: Dict[int, List[Player]] = {}
        self.ball_history: List[Ball] = []
        self._last_emit_t: Optional[float] = None
        self._stable_streak: int = 0
        self._need_extra_stable: bool = False

    def _velocity(self, loc1: Location, loc2: Location, dt: float) -> float:
        if dt <= 0:
            return 0.0
        return float(np.hypot(loc2.x - loc1.x, loc2.y - loc1.y) / dt)

    def _distance(self, loc1: Location, loc2: Location) -> float:
        return float(np.hypot(loc2.x - loc1.x, loc2.y - loc1.y))

    def _ball_ok(self, frame_data: FrameData, prev: Optional[FrameData]) -> bool:
        return (
            prev is not None
            and frame_data.ball is not None
            and prev.ball is not None
        )

    def _closest_player(
        self, players: List[Player], loc: Location, max_dist: float
    ) -> Optional[int]:
        best_id = None
        best_d = float("inf")
        for player in players:
            d = self._distance(loc, Location(player.x_pitch, player.y_pitch))
            if d < best_d and d <= max_dist:
                best_d = d
                best_id = player.object_id
        return best_id

    def _gate(self, event: Optional[Event]) -> Optional[Event]:
        if event is None:
            return None
        if event.confidence < self.emit_conf:
            return None
        return event

    def _ball_speed_ok(self, frame_data: FrameData, prev: FrameData) -> bool:
        assert frame_data.ball and prev.ball
        dt = frame_data.timestamp - prev.timestamp
        if dt <= 0:
            return False
        d = self._distance(
            Location(prev.ball.x_pitch, prev.ball.y_pitch),
            Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch),
        )
        return (d / dt) <= self.max_ball_speed_m_s

    def _in_pitch(self, loc: Location) -> bool:
        return abs(loc.x) <= self.half_length_m + 0.05 and abs(loc.y) <= 17.5

    def _cooldown_ok(self, t_end: float) -> bool:
        if self._last_emit_t is None:
            return True
        return (t_end - self._last_emit_t) >= self.min_emit_gap_s

    def detect_pass(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> Optional[Event]:
        if not self._ball_ok(frame_data, prev_frame_data):
            return None
        assert prev_frame_data is not None and frame_data.ball and prev_frame_data.ball
        dt = frame_data.timestamp - prev_frame_data.timestamp
        start = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        end = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        if not (self._in_pitch(start) and self._in_pitch(end)):
            return None
        ball_vel = self._velocity(start, end, dt)
        if ball_vel < self.pass_velocity_threshold:
            return None
        pid = self._closest_player(prev_frame_data.players, start, 3.0)
        if pid is None:
            return None
        # Stronger velocity → higher conf; at thr alone stays below emit.
        conf = min(1.0, ball_vel / 20.0)
        return self._gate(
            Event(
                id=f"pass_{frame_data.frame_id}",
                type=EventType.PASS,
                start_frame=prev_frame_data.frame_id,
                end_frame=frame_data.frame_id,
                start_location=start,
                end_location=end,
                involved_players=[pid],
                confidence=conf,
                timestamp_start=prev_frame_data.timestamp,
                timestamp_end=frame_data.timestamp,
            )
        )

    def detect_dribble(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> Optional[Event]:
        if not self._ball_ok(frame_data, prev_frame_data):
            return None
        assert prev_frame_data is not None and frame_data.ball and prev_frame_data.ball
        ball = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        prev_ball = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        moved = self._distance(prev_ball, ball)
        for player in frame_data.players:
            pl = Location(player.x_pitch, player.y_pitch)
            if self._distance(ball, pl) >= self.dribble_distance_threshold:
                continue
            prev_pl = Location(player.x_pitch, player.y_pitch)
            if self._distance(prev_ball, prev_pl) >= self.dribble_distance_threshold:
                continue
            # Require clear carry distance; weak cling stays below emit.
            conf = 0.85 if moved >= 0.4 else 0.70
            return self._gate(
                Event(
                    id=f"dribble_{frame_data.frame_id}",
                    type=EventType.DRIBBLE,
                    start_frame=prev_frame_data.frame_id,
                    end_frame=frame_data.frame_id,
                    start_location=prev_ball,
                    end_location=ball,
                    involved_players=[player.object_id],
                    confidence=conf,
                    timestamp_start=prev_frame_data.timestamp,
                    timestamp_end=frame_data.timestamp,
                )
            )
        return None

    def detect_shot(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> Optional[Event]:
        if not self._ball_ok(frame_data, prev_frame_data):
            return None
        assert prev_frame_data is not None and frame_data.ball and prev_frame_data.ball
        dt = frame_data.timestamp - prev_frame_data.timestamp
        start = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        end = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        if not (self._in_pitch(start) and self._in_pitch(end)):
            return None
        ball_vel = self._velocity(start, end, dt)
        if ball_vel < self.shot_velocity_threshold:
            return None
        # Pitch 1: near Goal1 (south −x) or Goal2 (north +x).
        goal_line = self.half_length_m
        if abs(end.x) <= goal_line - self.shot_goal_band_m:
            return None
        # Must move toward the nearer goal line (not lateral wiggle / clearance out).
        if abs(end.x) <= abs(start.x) + 0.3:
            return None
        pid = self._closest_player(prev_frame_data.players, start, 4.0)
        if pid is None:
            return None
        conf = min(1.0, ball_vel / 25.0)
        return self._gate(
            Event(
                id=f"shot_{frame_data.frame_id}",
                type=EventType.SHOT,
                start_frame=prev_frame_data.frame_id,
                end_frame=frame_data.frame_id,
                start_location=start,
                end_location=end,
                involved_players=[pid],
                confidence=conf,
                timestamp_start=prev_frame_data.timestamp,
                timestamp_end=frame_data.timestamp,
            )
        )

    def detect_recovery(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> Optional[Event]:
        if not self._ball_ok(frame_data, prev_frame_data):
            return None
        assert prev_frame_data is not None and frame_data.ball and prev_frame_data.ball
        ball = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        prev_ball = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        moved = self._distance(prev_ball, ball)
        if moved < 0.8:
            return None
        for player in frame_data.players:
            pl = Location(player.x_pitch, player.y_pitch)
            if self._distance(ball, pl) >= self.recovery_proximity:
                continue
            prev_d = self._distance(prev_ball, pl)
            if prev_d <= 2.5:
                continue
            # Must be closing on the player (not a random reappear).
            if prev_d - self._distance(ball, pl) < 0.5:
                continue
            return self._gate(
                Event(
                    id=f"recovery_{frame_data.frame_id}",
                    type=EventType.RECOVERY,
                    start_frame=prev_frame_data.frame_id,
                    end_frame=frame_data.frame_id,
                    start_location=prev_ball,
                    end_location=ball,
                    involved_players=[player.object_id],
                    confidence=0.85,
                    timestamp_start=prev_frame_data.timestamp,
                    timestamp_end=frame_data.timestamp,
                )
            )
        return None

    def detect_events(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> List[Event]:
        """Emit at most one event per frame (shot > pass > recovery > dribble)."""
        if not self._ball_ok(frame_data, prev_frame_data):
            self._stable_streak = 0
            return []
        assert prev_frame_data is not None
        if not self._ball_speed_ok(frame_data, prev_frame_data):
            self._stable_streak = 0
            self._need_extra_stable = True
            return []
        self._stable_streak += 1
        # After a teleport, require one extra continuous step before emit.
        if self._need_extra_stable and self._stable_streak < 2:
            return []
        self._need_extra_stable = False
        if not self._cooldown_ok(frame_data.timestamp):
            return []
        detectors = {
            EventType.SHOT: self.detect_shot,
            EventType.PASS: self.detect_pass,
            EventType.RECOVERY: self.detect_recovery,
        }
        if self.enable_dribble:
            detectors[EventType.DRIBBLE] = self.detect_dribble
            order = _EVENT_PRIORITY
        else:
            order = (EventType.SHOT, EventType.PASS, EventType.RECOVERY)
        for et in order:
            ev = detectors[et](frame_data, prev_frame_data)
            if ev is not None:
                self._last_emit_t = ev.timestamp_end
                return [ev]
        return []
