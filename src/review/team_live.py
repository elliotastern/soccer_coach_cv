"""Live Match 3 team labeling for coach Pitch 1 (precision-first + stable).

Review-only temporal layer (TeamSession) on shared team_core features.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from src.perception.team_core import (
    FEAT_BASE,
    HUE_BINS,
    KIT_DIM,
    KIT_MODE_AUTO,
    KIT_MODE_MATCH3,
    MIN_CROP_STD,
    MIN_JERSEY_FRAC,
    OUTLIER_MEDIAN_MULT,
    STICKY_FLIP_CONF_AUTO,
    TEAM_ASSIGN_CONF,
    TEAM_MIN_CROPS,
    TEAM_MIN_CROPS_AUTO,
    assign_feature,
    assign_from_feature,
    fit_match_centroids,
    fit_team_centroids,
    jersey_feature,
    torso_crop,
    tracklet_median_feature,
    which_goal_box,
)
from src.perception.team_strategy import STRATEGIES, TeamStrategy, production_default, sticky_conf

BOX_DEDUP_M = 2.0
STABLE_PID_M = 2.8
CENTROID_EMA = 0.06
HOLD_MAX_GAP = 2
HOLD_M = 3.0
STICKY_M = 4.0
STICKY_FLIP_CONF = 0.78
VOTE_LEN = 5
VOTE_MIN = 2
HIST_LEN = 4
FEAT_HIST_LEN = 4
TRAJ_GATE_M = 4.5
TRACKLET_COLOR_CONF = 0.55
TRACKLET_VOTE_MIN = 3
PIXEL_NUDGE_MARGIN = 0.12
PIXEL_NUDGE_MIN_CONF = 0.68
AUTO_BALANCE_MIN = 60
AUTO_BALANCE_LO = 0.35
AUTO_BALANCE_HI = 0.65
AUTO_BALANCE_COOLDOWN = 45


def _dist_xy(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _track_vel(hist: list) -> tuple[float, float]:
    if len(hist) < 2:
        return 0.0, 0.0
    n = min(3, len(hist) - 1)
    vx = (hist[-1][0] - hist[-1 - n][0]) / float(n)
    vy = (hist[-1][1] - hist[-1 - n][1]) / float(n)
    return float(vx), float(vy)


def _pred_xy(track: dict) -> tuple[float, float]:
    hist = list(track.get("xy_hist") or [track["xy"]])
    vx, vy = _track_vel(hist)
    x, y = hist[-1]
    return (float(x + vx), float(y + vy))


def _match_cost(obs_xy, track: dict) -> float:
    pred = _pred_xy(track)
    base = _dist_xy(obs_xy, pred)
    hist = list(track.get("xy_hist") or [track["xy"]])
    vx, vy = _track_vel(hist)
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 0.15:
        return base
    dx, dy = obs_xy[0] - hist[-1][0], obs_xy[1] - hist[-1][1]
    step = (dx * dx + dy * dy) ** 0.5
    if step < 1e-6:
        return base
    align = (dx * vx + dy * vy) / (step * speed + 1e-6)
    return base * (1.35 - 0.35 * max(0.0, align))


def _vote_force(votes: list[int], prior: int = -1) -> int:
    labeled = [int(v) for v in votes if int(v) >= 0]
    if labeled:
        tid, _ = Counter(labeled).most_common(1)[0]
        return int(tid)
    return int(prior) if prior >= 0 else -1


def _fill_no_gray(players: list[dict]) -> None:
    labeled = [p for p in players if int(p.get("team", -1)) >= 0]
    if not labeled:
        for p in players:
            p["team"] = 0
        return
    maj = Counter(int(p["team"]) for p in labeled).most_common(1)[0][0]
    for p in players:
        if int(p.get("team", -1)) >= 0:
            continue
        near_t, best = maj, 1e9
        for q in labeled:
            d = _dist_xy(p["xy"], q["xy"])
            if d < best:
                best, near_t = d, int(q["team"])
        p["team"] = near_t if best < 8.0 else int(maj)


def _nudge_frame_skew_flips(players: list[dict], max_ratio: float = 5.0) -> None:
    """Flip low-confidence majority labels instead of graying (no-gray product path)."""
    labeled = [p for p in players if int(p.get("team", -1)) >= 0]
    if len(labeled) < 10:
        return
    n0 = sum(1 for p in labeled if int(p["team"]) == 0)
    n1 = len(labeled) - n0
    if n0 == 0 or n1 == 0:
        return
    lo, hi = min(n0, n1), max(n0, n1)
    if lo < 3 or hi / lo <= max_ratio:
        return
    maj = 0 if n0 >= n1 else 1
    min_team = 1 - maj
    maj_members = [p for p in players if int(p.get("team", -1)) == maj]
    cap = int(lo + 1)
    excess = min(len(maj_members) - cap, 2)
    if excess <= 0:
        return
    maj_members.sort(
        key=lambda p: (float(p.get("team_conf", 0.5)), -int(p.get("age", 0))),
    )
    for p in maj_members[:excess]:
        if float(p.get("team_conf", 0.5)) < 0.58:
            p["team"] = min_team


def assign_stable_player_ids(
    players: list[tuple[float, float, int, int]],
    prev_tracks: list[dict],
    next_id: int,
    sticky_m: float = STABLE_PID_M,
) -> tuple[list[tuple[float, float, int, int]], list[dict], int]:
    cur = [
        {
            "xy": (float(p[0]), float(p[1])),
            "team": int(p[2]),
            "pid": int(p[3]),
            "age": 0,
            "matched": False,
        }
        for p in players
    ]
    for c in cur:
        near = None
        best_d = float(sticky_m)
        for q in prev_tracks:
            d = _dist_xy(c["xy"], q["xy"])
            if d <= best_d:
                best_d = d
                near = q
        if near is not None:
            near["matched"] = True
            c["pid"] = int(near["pid"])
        else:
            c["pid"] = int(next_id)
            next_id += 1
    out_tracks = []
    for c in cur:
        out_tracks.append(
            {
                "xy": c["xy"],
                "team": int(c["team"]),
                "pid": int(c["pid"]),
                "age": 0,
                "matched": True,
            }
        )
    for q in prev_tracks:
        if q.get("matched"):
            continue
        age = int(q.get("age", 0)) + 1
        if age > HOLD_MAX_GAP:
            continue
        out_tracks.append(
            {
                "xy": q["xy"],
                "team": int(q["team"]),
                "pid": int(q["pid"]),
                "age": age,
                "matched": False,
            }
        )
    stable = [(d["xy"][0], d["xy"][1], int(d["team"]), int(d["pid"])) for d in cur]
    return stable, out_tracks, next_id


def _vote_mode(votes: list[int]) -> int:
    labeled = [int(v) for v in votes if int(v) >= 0]
    if not labeled:
        return -1
    counts = Counter(labeled)
    tid, n = counts.most_common(1)[0]
    runner = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    if n >= VOTE_MIN and n > runner:
        return int(tid)
    return -1


def apply_goal_box_prior(players: list[dict]) -> None:
    by_box: dict[str, list[dict]] = {"south": [], "north": []}
    for p in players:
        box = which_goal_box(p["xy"])
        if box is not None:
            by_box[box].append(p)
    for members in by_box.values():
        if not members:
            continue
        labeled = [m for m in members if int(m.get("team", -1)) >= 0]
        if len(members) == 1 and labeled:
            continue
        for m in members:
            if int(m.get("team", -1)) < 0:
                continue
            if len(labeled) > 1:
                teams = {int(x["team"]) for x in labeled}
                if len(teams) > 1:
                    m["team"] = -1


def soft_cap_goal_box_duplicates(players: list[dict]) -> list[dict]:
    outside = [p for p in players if which_goal_box(p["xy"]) is None]
    kept = list(outside)
    for box_name in ("south", "north"):
        members = [p for p in players if which_goal_box(p["xy"]) == box_name]
        members.sort(
            key=lambda p: (
                int(p.get("age", 0)),
                0 if int(p.get("team", -1)) >= 0 else 1,
                -float(p.get("conf", 0.0)),
            )
        )
        chosen: list[dict] = []
        for m in members:
            if any(_dist_xy(m["xy"], c["xy"]) <= BOX_DEDUP_M for c in chosen):
                continue
            chosen.append(m)
        kept.extend(chosen)
    return kept


def soft_cap_frame_team_skew(players: list[dict], max_ratio: float = 4.0) -> None:
    """Gray excess majority labels when one kit dominates a frame (precision-first)."""
    labeled = [p for p in players if int(p.get("team", -1)) >= 0]
    if len(labeled) < 8:
        return
    n0 = sum(1 for p in labeled if int(p["team"]) == 0)
    n1 = len(labeled) - n0
    if n0 == 0 or n1 == 0:
        return
    lo, hi = min(n0, n1), max(n0, n1)
    if lo > 1 and hi / lo <= max_ratio:
        return
    maj = 0 if n0 >= n1 else 1
    cap = max(3, 2 + lo) if lo <= 1 and hi >= 7 else int(max(lo * max_ratio, lo + 1))
    maj_members = [p for p in players if int(p.get("team", -1)) == maj]
    excess = len(maj_members) - cap
    if excess <= 0:
        return
    maj_members.sort(
        key=lambda p: (int(p.get("age", 0)), int(p.get("pid", 0))),
        reverse=True,
    )
    for p in maj_members[:excess]:
        p["team"] = -1


class TeamSession:
    """Session-locked team model: fit once, EMA, sticky, vote buffer, box prior."""

    def __init__(
        self,
        kit_mode: str = KIT_MODE_AUTO,
        strategy: TeamStrategy | None = None,
    ):
        self.strategy = strategy or production_default()
        if strategy is None:
            if kit_mode == KIT_MODE_MATCH3:
                self.strategy = STRATEGIES["match3_session"]
            else:
                self.strategy = STRATEGIES["auto_traj_no_gray"]
        self.kit_mode = self.strategy.kit_mode
        self.min_crops = (
            TEAM_MIN_CROPS_AUTO if self.kit_mode == KIT_MODE_AUTO else TEAM_MIN_CROPS
        )
        self.sticky_flip_conf = sticky_conf(self.strategy)
        self.centroids: np.ndarray | None = None
        self.radius: float | None = None
        self.prev: list[dict] = []
        self.prev_fused: list[dict] = []
        self._next_stable_pid: int = 1
        self._feat_bank: list[np.ndarray] = []
        self._bal_n0 = 0
        self._bal_n1 = 0
        self._bal_cooldown = 0

    def reset(self) -> None:
        self.centroids = None
        self.radius = None
        self.prev = []
        self.prev_fused = []
        self._next_stable_pid = 1
        self._feat_bank = []
        self._bal_n0 = 0
        self._bal_n1 = 0
        self._bal_cooldown = 0

    @staticmethod
    def _flip_team(tid: int) -> int:
        if tid == 0:
            return 1
        if tid == 1:
            return 0
        return tid

    def _swap_centroids(self) -> None:
        if self.centroids is not None:
            self.centroids = self.centroids[[1, 0]]

    def _maybe_rebalance_auto(self, player_pts: list[dict]) -> None:
        if not self.strategy.use_rebalance:
            return
        if self.kit_mode != KIT_MODE_AUTO or self.centroids is None:
            return
        if self._bal_cooldown > 0:
            self._bal_cooldown -= 1
            return
        for p in player_pts:
            t = int(p.get("team", -1))
            if t == 0:
                self._bal_n0 += 1
            elif t == 1:
                self._bal_n1 += 1
        total = self._bal_n0 + self._bal_n1
        if total < AUTO_BALANCE_MIN:
            return
        share = self._bal_n0 / total
        if AUTO_BALANCE_LO <= share <= AUTO_BALANCE_HI:
            return
        self._swap_centroids()
        for p in player_pts:
            t = int(p.get("team", -1))
            if t >= 0:
                p["team"] = self._flip_team(t)
        for q in self.prev:
            t = int(q.get("team", -1))
            if t >= 0:
                q["team"] = self._flip_team(t)
        for q in self.prev_fused:
            t = int(q.get("team", -1))
            if t >= 0:
                q["team"] = self._flip_team(t)
                q["votes"] = [
                    self._flip_team(int(v)) if int(v) >= 0 else int(v)
                    for v in (q.get("votes") or [])
                ]
        self._bal_n0 = 0
        self._bal_n1 = 0
        self._bal_cooldown = AUTO_BALANCE_COOLDOWN

    def _ema(self, feats: list[np.ndarray], labs: list[int]) -> None:
        if self.centroids is None:
            return
        a = CENTROID_EMA
        trial = self.centroids.copy()
        for tid in (0, 1):
            group = [f for f, lab in zip(feats, labs) if lab == tid]
            if not group:
                continue
            mean = np.mean(np.stack(group, axis=0), axis=0)
            trial[tid] = (1.0 - a) * trial[tid] + a * mean
        if self.strategy.use_symmetric_ema and self.kit_mode == KIT_MODE_AUTO:
            self.centroids = trial
            return
        s0 = float(trial[0, 0] - trial[0, 1] - 0.5 * trial[0, 2])
        s1 = float(trial[1, 0] - trial[1, 1] - 0.5 * trial[1, 2])
        if s0 >= s1:
            self.centroids = trial

    def _sticky(self, player_pts: list[dict]) -> None:
        if not self.prev:
            return
        no_gray = bool(self.strategy.no_gray)
        for p in player_pts:
            xy = p.get("xy")
            if xy is None:
                continue
            near = None
            best_d = STICKY_M
            for q in self.prev:
                d = _dist_xy(xy, q["xy"])
                if d <= best_d:
                    best_d = d
                    near = q
            if near is None or int(near.get("team", -1)) < 0:
                continue
            new_t = int(p.get("team", -1))
            conf = float(p.get("team_conf", 0.0))
            prior = int(near["team"])
            if new_t < 0:
                p["team"] = prior
                p["team_conf"] = max(conf, 0.45)
            elif new_t != prior and (no_gray or conf < self.sticky_flip_conf):
                if no_gray and conf >= self.sticky_flip_conf:
                    continue
                p["team"] = prior
                if no_gray:
                    p["team_conf"] = max(conf, 0.55)

    def _tracklet_color_label(
        self,
        player_pts: list[dict],
        idxs: list[int],
    ) -> None:
        if self.centroids is None or self.radius is None:
            return
        if not (self.strategy.use_traj_vote or self.strategy.no_gray):
            return
        for i in idxs:
            p = player_pts[i]
            feat = p.get("feat")
            if feat is None or p.get("xy") is None:
                continue
            near_team_hist: list[int] = []
            near_hist: list[np.ndarray] = []
            for q in self.prev:
                if _dist_xy(p["xy"], q["xy"]) <= STICKY_M:
                    near_hist = list(q.get("feat_hist") or [])
                    near_team_hist = list(q.get("team_hist") or [])
                    break
            obs_tid = int(p.get("team", -1))
            obs_conf = float(p.get("team_conf", 0.0))
            team_hist = (near_team_hist + [obs_tid])[-FEAT_HIST_LEN:]
            labeled = [t for t in team_hist if int(t) >= 0]
            vote_tid = obs_tid
            if len(labeled) >= TRACKLET_VOTE_MIN:
                vote_tid, vote_n = Counter(labeled).most_common(1)[0]
                runner = Counter(labeled).most_common(2)[1][1] if len(Counter(labeled)) > 1 else 0
                vote_tid = int(vote_tid)
                if vote_n >= TRACKLET_VOTE_MIN and vote_n > runner and obs_tid != vote_tid:
                    if obs_conf < TRACKLET_COLOR_CONF or vote_n >= len(labeled):
                        p["team"] = vote_tid
                        p["team_conf"] = max(obs_conf, 0.55 + 0.05 * vote_n)
                        continue
            hist = (near_hist + [feat])[-FEAT_HIST_LEN:]
            if len(hist) < 2:
                continue
            med = tracklet_median_feature(hist)
            if med is None:
                continue
            xy = p.get("xy")
            pos = (float(xy[0]), float(xy[1])) if xy is not None else None
            tid, conf = assign_feature(
                med,
                self.centroids,
                self.radius,
                pos,
                kit_mode=self.kit_mode,
                strategy=self.strategy,
            )
            if int(tid) < 0:
                continue
            obs_tid = int(p.get("team", -1))
            obs_conf = float(p.get("team_conf", 0.0))
            if obs_tid != int(tid) and obs_conf < TRACKLET_COLOR_CONF:
                if len(labeled) < TRACKLET_VOTE_MIN or int(tid) == vote_tid:
                    p["team"] = int(tid)
                    p["team_conf"] = max(obs_conf, float(conf))
            elif obs_tid == int(tid):
                p["team_conf"] = max(obs_conf, float(conf))

    def stabilize_fused(
        self,
        players: list[tuple[float, float, int, int]],
    ) -> list[tuple[float, float, int, int]]:
        no_gray = bool(self.strategy.no_gray)
        use_traj = bool(self.strategy.use_traj_vote) or no_gray
        cur = [
            {
                "xy": (float(p[0]), float(p[1])),
                "team": int(p[2]),
                "pid": int(p[3]),
                "age": 0,
                "matched": False,
                "votes": [int(p[2])],
                "xy_hist": [(float(p[0]), float(p[1]))],
                "team_conf": 0.55,
            }
            for p in players
        ]
        sticky_m = TRAJ_GATE_M if use_traj else min(STICKY_M, 2.8)
        used_prev: set[int] = set()
        for c in cur:
            near = None
            best = sticky_m
            near_i = -1
            for i, q in enumerate(self.prev_fused):
                if i in used_prev:
                    continue
                d = _match_cost(c["xy"], q) if use_traj else _dist_xy(c["xy"], q["xy"])
                if d <= best:
                    best, near, near_i = d, q, i
            if near is None:
                c["pid"] = self._next_stable_pid
                self._next_stable_pid += 1
                continue
            used_prev.add(near_i)
            near["matched"] = True
            c["pid"] = int(near["pid"])
            self._next_stable_pid = max(self._next_stable_pid, c["pid"] + 1)
            prior_votes = list(near.get("votes") or [])
            prior = int(near.get("team", -1))
            obs = int(c["team"])
            if obs < 0 and prior >= 0:
                obs = prior
            votes = (prior_votes + [obs])[-VOTE_LEN:]
            c["votes"] = votes
            voted = _vote_mode(votes)
            if voted >= 0:
                c["team"] = voted
            elif no_gray:
                c["team"] = _vote_force(votes, prior)
            elif prior >= 0 and int(c["team"]) < 0:
                c["team"] = prior
            hist = list(near.get("xy_hist") or [near["xy"]])
            c["xy_hist"] = (hist + [c["xy"]])[-HIST_LEN:]
        held = []
        for q in self.prev_fused:
            if q.get("matched"):
                continue
            age = int(q.get("age", 0)) + 1
            if age > HOLD_MAX_GAP:
                continue
            if int(q.get("team", -1)) < 0 and not no_gray:
                continue
            if which_goal_box(q["xy"]) is not None:
                continue
            if any(_dist_xy(q["xy"], c["xy"]) <= HOLD_M for c in cur):
                continue
            held.append(
                {
                    "xy": q["xy"],
                    "team": int(q["team"]) if int(q.get("team", -1)) >= 0 else 0,
                    "pid": int(q.get("pid", -1)),
                    "age": age,
                    "matched": False,
                    "votes": list(q.get("votes") or [int(q.get("team", 0))]),
                    "xy_hist": list(q.get("xy_hist") or [q["xy"]]),
                    "team_conf": float(q.get("team_conf", 0.6)),
                }
            )
        out_dicts = soft_cap_goal_box_duplicates(cur + held)
        if self.strategy.use_skew_cap and self.kit_mode == KIT_MODE_AUTO and not no_gray:
            soft_cap_frame_team_skew(out_dicts)
        if no_gray:
            _nudge_frame_skew_flips(out_dicts)
        if not no_gray:
            apply_goal_box_prior(out_dicts)
        if no_gray:
            _fill_no_gray(out_dicts)
        self.prev_fused = [
            {
                "xy": d["xy"],
                "team": int(d["team"]),
                "pid": int(d["pid"]),
                "age": int(d.get("age", 0)),
                "votes": list(d.get("votes") or [int(d["team"])]),
                "xy_hist": list(d.get("xy_hist") or [d["xy"]]),
            }
            for d in out_dicts
        ]
        return [
            (d["xy"][0], d["xy"][1], int(d["team"]), int(d["pid"]))
            for d in out_dicts
        ]

    def label(self, player_pts: list[dict], frames_by_cam: dict) -> list[dict]:
        if not player_pts or not frames_by_cam:
            return player_pts
        feats: list[np.ndarray] = []
        idxs: list[int] = []
        for i, p in enumerate(player_pts):
            p["team"] = -1
            p["team_conf"] = 0.0
            cam = p.get("cam")
            bbox = p.get("bbox")
            fr = frames_by_cam.get(cam) if cam else None
            wh = (fr.shape[1], fr.shape[0]) if fr is not None else None
            if fr is None or bbox is None:
                continue
            crop = torso_crop(fr, bbox, cam=cam, frame_wh=wh)
            feat = jersey_feature(crop) if crop is not None else None
            if feat is None:
                continue
            xy = p.get("xy")
            pos_xy = (float(xy[0]), float(xy[1])) if xy is not None else (0.0, 0.0)
            from src.perception.team_core import team_vote_weight

            p["crop_valid"] = True
            p["team_weight"] = team_vote_weight(
                cam, bbox, wh, pos_xy, True
            )
            p["feat"] = feat
            feats.append(feat)
            idxs.append(i)
        for feat in feats:
            self._feat_bank.append(feat)
        if self.centroids is None or self.radius is None or self.strategy.per_frame_only:
            if len(self._feat_bank) < self.min_crops and not self.strategy.per_frame_only:
                return player_pts
            bank = feats if self.strategy.per_frame_only else self._feat_bank
            if len(bank) < (TEAM_MIN_CROPS if self.strategy.per_frame_only else self.min_crops):
                return player_pts
            fit = fit_match_centroids(
                bank,
                min_crops=TEAM_MIN_CROPS if self.strategy.per_frame_only else self.min_crops,
                kit_mode=self.kit_mode,
            )
            if fit is None:
                return player_pts
            self.centroids, self.radius = fit
        labs = []
        for j, i in enumerate(idxs):
            xy = player_pts[i].get("xy")
            pos = (float(xy[0]), float(xy[1])) if xy is not None else None
            tid, conf = assign_feature(
                feats[j],
                self.centroids,
                self.radius,
                pos,
                kit_mode=self.kit_mode,
                strategy=self.strategy,
            )
            player_pts[i]["team"] = int(tid)
            player_pts[i]["team_conf"] = float(conf)
            labs.append(int(tid))
        self._tracklet_color_label(player_pts, idxs)
        self._ema(feats, labs)
        self._sticky(player_pts)
        self._maybe_rebalance_auto(player_pts)
        old_prev = self.prev
        self.prev = []
        for p in player_pts:
            if p.get("xy") is None:
                continue
            feat_hist: list[np.ndarray] = []
            team_hist: list[int] = []
            feat = p.get("feat")
            old_hist: list[np.ndarray] = []
            old_teams: list[int] = []
            for q in old_prev:
                if _dist_xy(p["xy"], q["xy"]) <= STICKY_M:
                    old_hist = list(q.get("feat_hist") or [])
                    old_teams = list(q.get("team_hist") or [])
                    break
            if feat is not None:
                feat_hist = (old_hist + [feat])[-FEAT_HIST_LEN:]
            team_hist = (old_teams + [int(p.get("team", -1))])[-FEAT_HIST_LEN:]
            self.prev.append(
                {
                    "xy": p["xy"],
                    "team": int(p.get("team", -1)),
                    "conf": float(p.get("team_conf", 0.0)),
                    "feat_hist": feat_hist,
                    "team_hist": team_hist,
                }
            )
        return player_pts


def label_player_pts(
    player_pts: list[dict],
    frames_by_cam: dict,
    team_session: TeamSession | None = None,
) -> list[dict]:
    if team_session is not None:
        return team_session.label(player_pts, frames_by_cam)
    sess = TeamSession()
    return sess.label(player_pts, frames_by_cam)
