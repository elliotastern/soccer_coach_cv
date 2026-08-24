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
    MIN_CROP_STD,
    MIN_JERSEY_FRAC,
    OUTLIER_MEDIAN_MULT,
    TEAM_ASSIGN_CONF,
    TEAM_MIN_CROPS,
    assign_feature,
    assign_from_feature,
    fit_match_centroids,
    fit_team_centroids,
    jersey_feature,
    torso_crop,
    which_goal_box,
)

BOX_DEDUP_M = 2.0
STABLE_PID_M = 2.8
CENTROID_EMA = 0.06
HOLD_MAX_GAP = 2
HOLD_M = 3.0
STICKY_M = 4.0
STICKY_FLIP_CONF = 0.78
VOTE_LEN = 5
VOTE_MIN = 2


def _dist_xy(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


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


class TeamSession:
    """Session-locked team model: fit once, EMA, sticky, vote buffer, box prior."""

    def __init__(self):
        self.centroids: np.ndarray | None = None
        self.radius: float | None = None
        self.prev: list[dict] = []
        self.prev_fused: list[dict] = []
        self._next_stable_pid: int = 1

    def reset(self) -> None:
        self.centroids = None
        self.radius = None
        self.prev = []
        self.prev_fused = []
        self._next_stable_pid = 1

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
        s0 = float(trial[0, 0] - trial[0, 1] - 0.5 * trial[0, 2])
        s1 = float(trial[1, 0] - trial[1, 1] - 0.5 * trial[1, 2])
        if s0 >= s1:
            self.centroids = trial

    def _sticky(self, player_pts: list[dict]) -> None:
        if not self.prev:
            return
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
            elif new_t != prior and conf < STICKY_FLIP_CONF:
                p["team"] = prior

    def stabilize_fused(
        self,
        players: list[tuple[float, float, int, int]],
    ) -> list[tuple[float, float, int, int]]:
        cur = [
            {
                "xy": (float(p[0]), float(p[1])),
                "team": int(p[2]),
                "pid": int(p[3]),
                "age": 0,
                "matched": False,
                "votes": [int(p[2])],
            }
            for p in players
        ]
        sticky_m = min(STICKY_M, 2.8)
        for c in cur:
            near = None
            best_d = sticky_m
            for q in self.prev_fused:
                d = _dist_xy(c["xy"], q["xy"])
                if d <= best_d:
                    best_d = d
                    near = q
            if near is not None:
                near["matched"] = True
                c["pid"] = int(near["pid"])
                prior_votes = list(near.get("votes") or [])
                obs = int(c["team"])
                prior = int(near.get("team", -1))
                if obs < 0 and prior >= 0:
                    obs = prior
                votes = (prior_votes + [obs])[-VOTE_LEN:]
                c["votes"] = votes
                voted = _vote_mode(votes)
                if voted >= 0:
                    c["team"] = voted
                elif prior >= 0 and int(c["team"]) < 0:
                    c["team"] = prior
            else:
                c["pid"] = self._next_stable_pid
                self._next_stable_pid += 1
        held = []
        for q in self.prev_fused:
            if q.get("matched"):
                continue
            age = int(q.get("age", 0)) + 1
            if age > HOLD_MAX_GAP:
                continue
            if int(q.get("team", -1)) < 0:
                continue
            if which_goal_box(q["xy"]) is not None:
                continue
            if any(_dist_xy(q["xy"], c["xy"]) <= HOLD_M for c in cur):
                continue
            held.append(
                {
                    "xy": q["xy"],
                    "team": int(q["team"]),
                    "pid": int(q.get("pid", -1)),
                    "age": age,
                    "matched": False,
                    "votes": list(q.get("votes") or [int(q["team"])]),
                }
            )
        out_dicts = soft_cap_goal_box_duplicates(cur + held)
        apply_goal_box_prior(out_dicts)
        self.prev_fused = [
            {
                "xy": d["xy"],
                "team": int(d["team"]),
                "pid": int(d["pid"]),
                "age": int(d.get("age", 0)),
                "votes": list(d.get("votes") or [int(d["team"])]),
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
            feats.append(feat)
            idxs.append(i)
        if self.centroids is None or self.radius is None:
            fit = fit_team_centroids(feats)
            if fit is None:
                return player_pts
            self.centroids, self.radius = fit
        labs = []
        for j, i in enumerate(idxs):
            xy = player_pts[i].get("xy")
            pos = (float(xy[0]), float(xy[1])) if xy is not None else None
            tid, conf = assign_feature(feats[j], self.centroids, self.radius, pos)
            player_pts[i]["team"] = int(tid)
            player_pts[i]["team_conf"] = float(conf)
            labs.append(int(tid))
        self._ema(feats, labs)
        self._sticky(player_pts)
        self.prev = [
            {
                "xy": p["xy"],
                "team": int(p.get("team", -1)),
                "conf": float(p.get("team_conf", 0.0)),
            }
            for p in player_pts
            if p.get("xy") is not None
        ]
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
