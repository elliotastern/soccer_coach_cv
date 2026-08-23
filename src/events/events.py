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
    EventType.MOVEMENT,
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
        enable_dribble: bool = True,
        enable_movement: bool = True,
        movement_velocity_min: float = 1.0,
        movement_proximity: float = 4.0,
        co_move_min_player_m: float = 0.15,
        co_move_min_cos: float = 0.55,
        dribble_window_frames: int = 2,
        dribble_min_carry_m: float = 0.55,
        dribble_co_move_streak: int = 1,
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
        self.enable_movement = enable_movement
        self.movement_velocity_min = movement_velocity_min
        self.movement_proximity = movement_proximity
        self.co_move_min_player_m = co_move_min_player_m
        self.co_move_min_cos = co_move_min_cos
        self.dribble_window_frames = dribble_window_frames
        self.dribble_min_carry_m = dribble_min_carry_m
        self.dribble_co_move_streak = dribble_co_move_streak
        self.player_history: Dict[int, List[Player]] = {}
        self.ball_history: List[Ball] = []
        self._last_emit_t: Optional[float] = None
        self._stable_streak: int = 0
        self._need_extra_stable: bool = False
        self._dribble_buf: List[tuple] = []

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

    def _player_loc(
        self, players: List[Player], object_id: int
    ) -> Optional[Location]:
        for player in players:
            if player.object_id == object_id:
                return Location(player.x_pitch, player.y_pitch)
        return None

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

    def _co_movement_ok(
        self,
        from_a: Location,
        to_a: Location,
        from_b: Location,
        to_b: Location,
    ) -> bool:
        """Ball and player displacements aligned — rejects map jitter on static players."""
        ax, ay = to_a.x - from_a.x, to_a.y - from_a.y
        bx, by = to_b.x - from_b.x, to_b.y - from_b.y
        player_move = float(np.hypot(bx, by))
        if player_move < self.co_move_min_player_m:
            return False
        ball_move = float(np.hypot(ax, ay))
        if ball_move < 1e-6:
            return False
        cos_sim = (ax * bx + ay * by) / (ball_move * player_move)
        return cos_sim >= self.co_move_min_cos

    def _cooldown_ok(
        self, t_end: float, carry_start_t: Optional[float] = None
    ) -> bool:
        if self._last_emit_t is None:
            return True
        if carry_start_t is not None and carry_start_t < self._last_emit_t:
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

    def _dribble_partial_carry_m(self) -> float:
        if not self._dribble_buf:
            return 0.0
        first_prev, _, _, _ = self._dribble_buf[0]
        _, last_cur, _, _ = self._dribble_buf[-1]
        assert first_prev.ball and last_cur.ball
        start = Location(first_prev.ball.x_pitch, first_prev.ball.y_pitch)
        end = Location(last_cur.ball.x_pitch, last_cur.ball.y_pitch)
        return self._distance(start, end)

    def _dribble_max_speed_m_s(self) -> float:
        return self.pass_velocity_threshold * 0.92

    def _dribble_proximity_ok(
        self,
        prev_ball: Location,
        ball: Location,
        prev_pl: Location,
        cur_pl: Location,
        dt: float,
    ) -> bool:
        """Fuse stride-4: mapped player can lag ball — proximity without strict co-move."""
        cling_m = self.movement_proximity
        d_cur = self._distance(ball, cur_pl)
        d_prev = self._distance(prev_ball, prev_pl)
        if d_cur > min(cling_m, 3.0) or d_prev > cling_m:
            return False
        goal_line = self.half_length_m
        if abs(ball.x) > goal_line - self.shot_goal_band_m:
            return False
        if dt <= 0:
            return False
        ball_vel = self._velocity(prev_ball, ball, dt)
        return ball_vel < self._dribble_max_speed_m_s()

    def _dribble_cling_step(
        self, frame_data: FrameData, prev_frame_data: FrameData
    ) -> Optional[tuple[int, bool]]:
        """Return (player_id, co_move_ok) when ball clings to a co-moving player."""
        assert frame_data.ball and prev_frame_data.ball
        ball = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        prev_ball = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        cling_m = self.movement_proximity
        dt = frame_data.timestamp - prev_frame_data.timestamp
        if dt > 0 and self._velocity(prev_ball, ball, dt) >= self._dribble_max_speed_m_s():
            return None
        pid = self._closest_player(prev_frame_data.players, prev_ball, cling_m)
        if pid is None:
            return None
        best_pid, best_co = pid, False
        best_cos = -1.0
        for player in prev_frame_data.players:
            pl = Location(player.x_pitch, player.y_pitch)
            if self._distance(prev_ball, pl) >= cling_m:
                continue
            cur_pl = self._player_loc(frame_data.players, player.object_id)
            if cur_pl is None:
                continue
            if not self._co_movement_ok(prev_ball, ball, pl, cur_pl):
                continue
            ax, ay = ball.x - prev_ball.x, ball.y - prev_ball.y
            bx, by = cur_pl.x - pl.x, cur_pl.y - pl.y
            pm = float(np.hypot(bx, by))
            bm = float(np.hypot(ax, ay))
            if pm < 1e-6 or bm < 1e-6:
                continue
            cos_sim = (ax * bx + ay * by) / (bm * pm)
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_pid = player.object_id
                best_co = True
        if best_cos < self.co_move_min_cos:
            best_pid, best_d = None, float("inf")
            for player in frame_data.players:
                prev_pl = self._player_loc(prev_frame_data.players, player.object_id)
                if prev_pl is None:
                    continue
                cur_pl = Location(player.x_pitch, player.y_pitch)
                if not self._dribble_proximity_ok(prev_ball, ball, prev_pl, cur_pl, dt):
                    continue
                d_cur = self._distance(ball, cur_pl)
                if d_cur < best_d:
                    best_d = d_cur
                    best_pid = player.object_id
                    best_co = True
            if best_pid is None:
                return None
            return best_pid, best_co
        return best_pid, best_co

    def detect_dribble(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> Optional[Event]:
        """Temporal window dribble — emit once per sustained carry."""
        if not self._ball_ok(frame_data, prev_frame_data):
            self._dribble_buf = []
            return None
        assert prev_frame_data is not None
        return self._detect_dribble_window(frame_data, prev_frame_data)

    def _detect_dribble_window(
        self, frame_data: FrameData, prev_frame_data: FrameData
    ) -> Optional[Event]:
        step = self._dribble_cling_step(frame_data, prev_frame_data)
        if step is None:
            self._dribble_buf = []
            return None
        pid, co_ok = step
        self._dribble_buf.append((prev_frame_data, frame_data, pid, co_ok))
        if len(self._dribble_buf) > self.dribble_window_frames:
            self._dribble_buf = self._dribble_buf[-self.dribble_window_frames:]
        if len(self._dribble_buf) < self.dribble_window_frames:
            return None
        first_prev, _, _, _ = self._dribble_buf[0]
        last_prev, last_cur, last_pid, _ = self._dribble_buf[-1]
        assert first_prev.ball and last_cur.ball
        start = Location(first_prev.ball.x_pitch, first_prev.ball.y_pitch)
        end = Location(last_cur.ball.x_pitch, last_cur.ball.y_pitch)
        if not (self._in_pitch(start) and self._in_pitch(end)):
            return None
        carry = self._distance(start, end)
        if carry + 1e-9 < self.dribble_min_carry_m:
            self._dribble_buf = []
            return None
        co_streak = sum(1 for _, _, _, ok in self._dribble_buf if ok)
        if co_streak < self.dribble_co_move_streak:
            self._dribble_buf = []
            return None
        conf = min(1.0, 0.82 + carry / 4.0)
        ev = self._gate(
            Event(
                id=f"dribble_{last_cur.frame_id}",
                type=EventType.DRIBBLE,
                start_frame=first_prev.frame_id,
                end_frame=last_cur.frame_id,
                start_location=start,
                end_location=end,
                involved_players=[last_pid],
                confidence=conf,
                timestamp_start=first_prev.timestamp,
                timestamp_end=last_cur.timestamp,
            )
        )
        if ev is None:
            return None
        self._dribble_buf = []
        return ev

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

    def detect_movement(
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
        moved = self._distance(start, end)
        if moved < 0.25:
            return None
        ball_vel = self._velocity(start, end, dt)
        if ball_vel < self.movement_velocity_min:
            return None
        if ball_vel >= self.pass_velocity_threshold:
            return None
        pid = self._closest_player(
            prev_frame_data.players, start, self.movement_proximity
        )
        if pid is None:
            return None
        prev_pl = self._player_loc(prev_frame_data.players, pid)
        cur_pl = self._player_loc(frame_data.players, pid)
        if prev_pl is None or cur_pl is None:
            return None
        if not self._co_movement_ok(start, end, prev_pl, cur_pl):
            return None
        conf = min(1.0, 0.80 + moved / 5.0)
        return self._gate(
            Event(
                id=f"movement_{frame_data.frame_id}",
                type=EventType.MOVEMENT,
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

    def detect_events(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData]
    ) -> List[Event]:
        """Emit at most one event per frame (shot > pass > recovery > dribble > movement)."""
        if not self._ball_ok(frame_data, prev_frame_data):
            self._stable_streak = 0
            self._dribble_buf = []
            return []
        assert prev_frame_data is not None
        if not self._ball_speed_ok(frame_data, prev_frame_data):
            self._stable_streak = 0
            self._need_extra_stable = True
            self._dribble_buf = []
            return []
        self._stable_streak += 1
        # After a teleport, require one extra continuous step before emit.
        if self._need_extra_stable and self._stable_streak < 2:
            self._dribble_buf = []
            return []
        self._need_extra_stable = False
        carry_start_t = (
            self._dribble_buf[0][0].timestamp if self._dribble_buf else None
        )
        if not self._cooldown_ok(frame_data.timestamp, carry_start_t):
            if self.enable_dribble and len(self._dribble_buf) >= self.dribble_window_frames:
                ev = self._detect_dribble_window(frame_data, prev_frame_data)
                if ev is not None:
                    self._last_emit_t = ev.timestamp_end
                    return [ev]
            self._dribble_buf = []
            return []
        for et in (EventType.SHOT, EventType.PASS, EventType.RECOVERY):
            ev = {
                EventType.SHOT: self.detect_shot,
                EventType.PASS: self.detect_pass,
                EventType.RECOVERY: self.detect_recovery,
            }[et](frame_data, prev_frame_data)
            if ev is not None:
                self._dribble_buf = []
                self._last_emit_t = ev.timestamp_end
                return [ev]
        if self.enable_dribble:
            ev = self._detect_dribble_window(frame_data, prev_frame_data)
            if ev is not None:
                self._last_emit_t = ev.timestamp_end
                return [ev]
        if self.enable_movement:
            one_from_dribble = (
                len(self._dribble_buf) == self.dribble_window_frames - 1
                and self._dribble_partial_carry_m() + 1e-9 < self.dribble_min_carry_m
            )
            if not one_from_dribble:
                ev = self.detect_movement(frame_data, prev_frame_data)
                if ev is not None:
                    if len(self._dribble_buf) >= self.dribble_window_frames:
                        self._dribble_buf = []
                    self._last_emit_t = ev.timestamp_end
                    return [ev]
        return []
