"""Live Match 3 team labeling for coach Pitch 1 (precision-first + stable).

Mid-torso crop → adaptive green mask → kit fractions + HSV hue hist →
session-locked centroids, fused vote buffer, Pitch 1 goal-box prior.
Unsure → gray. No FIFA pitch constants. No neural ReID.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_GS = ROOT / "scripts" / "gold_set"
if str(_GS) not in sys.path:
    sys.path.insert(0, str(_GS))

from pitch1 import load_pitch1, pitch1_landmarks  # noqa: E402

TEAM_MIN_CROPS = 5
TEAM_ASSIGN_CONF = 0.48
OUTLIER_MEDIAN_MULT = 2.4
MIN_JERSEY_FRAC = 0.08
MIN_CROP_STD = 4.0
STICKY_M = 4.0
STICKY_FLIP_CONF = 0.78
CENTROID_EMA = 0.06
HOLD_MAX_GAP = 5
HOLD_M = 3.0
HUE_BINS = 10
VOTE_LEN = 5
VOTE_MIN = 2
# Feature: [blue, white, yellow, mean_s, mean_v] + HUE_BINS hist
KIT_DIM = 3
FEAT_BASE = 5


def torso_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Upper-mid torso (skip head + shorts/socks), slight side inset."""
    if frame is None or frame.size == 0:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w < 10 or h < 24:
        return None
    fh, fw = frame.shape[:2]
    y0 = max(0, int(y + 0.15 * h))
    y1 = min(fh, int(y + 0.48 * h))
    x0 = max(0, int(x + 0.12 * w))
    x1 = min(fw, int(x + 0.88 * w))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return frame[y0:y1, x0:x1]


def _adaptive_non_green(hsv: np.ndarray) -> np.ndarray:
    """Drop pitch green using crop-local sat/value percentiles (no fixed grass thr)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Broad green hue window; thresholds adapt to this crop
    hue_g = (h >= 25) & (h <= 105)
    if int(hue_g.sum()) >= 12:
        s_cut = float(np.percentile(s[hue_g], 35))
        v_lo = float(np.percentile(v[hue_g], 20))
        s_cut = max(28.0, min(s_cut, 90.0))
        v_lo = max(25.0, min(v_lo, 80.0))
        green = hue_g & (s >= s_cut) & (v >= v_lo)
    else:
        green = hue_g & (s >= 40) & (v >= 35)
    return (~green) & (v >= 35) & (v <= 250)


def _color_subspace(feat: np.ndarray) -> np.ndarray:
    """Kit fractions + hue hist (skip mean_s / mean_v for distance)."""
    return np.concatenate([feat[:KIT_DIM], feat[FEAT_BASE : FEAT_BASE + HUE_BINS]])


def jersey_feature(crop: np.ndarray) -> Optional[np.ndarray]:
    """[blue, white, yellow, mean_s, mean_v] + L1 hue hist. None if unusable."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = _adaptive_non_green(hsv)
    std = float(crop.std())
    if std < MIN_CROP_STD:
        whiteish = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 125)
        if float(whiteish.mean()) < 0.45:
            return None
    frac = float(keep.mean())
    if frac < MIN_JERSEY_FRAC or int(keep.sum()) < 18:
        return None
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = keep & (h >= 85) & (h <= 145) & (s >= 35)
    purple = keep & (h >= 125) & (h <= 170) & (s >= 30)
    white = keep & (s <= 55) & (v >= 125)
    yellow = keep & (h >= 12) & (h <= 40) & (s >= 45) & (v >= 60)
    n = float(max(int(keep.sum()), 1))
    hues = h[keep].astype(np.float32)
    hist, _ = np.histogram(hues, bins=HUE_BINS, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    hist_sum = float(hist.sum()) + 1e-6
    hist = hist / hist_sum
    base = np.array(
        [
            float((blue | purple).sum()) / n,
            float(white.sum()) / n,
            float(yellow.sum()) / n,
            float(s[keep].mean()),
            float(v[keep].mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, hist])


def _lock_labels(centroids: np.ndarray) -> np.ndarray:
    """Team 0 = bluer kit, Team 1 = whiter/yellower."""
    score = centroids[:, 0] - centroids[:, 1] - 0.5 * centroids[:, 2]
    order = np.argsort(-score)
    return centroids[order]


def fit_team_centroids(features: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    if len(features) < TEAM_MIN_CROPS:
        return None
    x = np.asarray(features, dtype=np.float32)
    xs = np.stack([_color_subspace(f) for f in x], axis=0)
    dmat = ((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2)
    i0, i1 = np.unravel_index(int(np.argmax(dmat)), dmat.shape)
    if i0 == i1:
        return None
    c0, c1 = x[i0].copy(), x[i1].copy()
    for _ in range(20):
        s0, s1 = _color_subspace(c0), _color_subspace(c1)
        d0 = np.linalg.norm(xs - s0, axis=1)
        d1 = np.linalg.norm(xs - s1, axis=1)
        lab = (d1 < d0).astype(np.int32)
        if lab.min() == lab.max():
            c1 = x[np.argmax(np.linalg.norm(xs - s0, axis=1))].copy()
            continue
        c0 = x[lab == 0].mean(axis=0)
        c1 = x[lab == 1].mean(axis=0)
    cents = _lock_labels(np.stack([c0, c1], axis=0))
    cs = np.stack([_color_subspace(cents[0]), _color_subspace(cents[1])], axis=0)
    dmin = np.min(np.linalg.norm(xs[:, None, :] - cs[None, :, :], axis=2), axis=1)
    radius = float(np.median(dmin) * OUTLIER_MEDIAN_MULT + 1e-3)
    sep = float(np.linalg.norm(cents[0, :KIT_DIM] - cents[1, :KIT_DIM]))
    if sep < 0.12:
        return None
    return cents, max(radius, 0.08)


def assign_from_feature(
    feature: np.ndarray,
    centroids: np.ndarray,
    radius: float,
) -> tuple[int, float]:
    blue, white, yellow = float(feature[0]), float(feature[1]), float(feature[2])
    light = white + yellow
    if blue >= 0.38 and blue >= light + 0.12:
        return 0, min(0.95, 0.55 + blue)
    if light >= 0.38 and light >= blue + 0.12:
        return 1, min(0.95, 0.55 + light)
    if float(feature[4]) < 70.0 and blue < 0.25 and light < 0.25:
        return -1, 0.0
    fs = _color_subspace(feature)
    cs = np.stack([_color_subspace(centroids[0]), _color_subspace(centroids[1])], axis=0)
    dists = np.linalg.norm(cs - fs, axis=1)
    tid = int(np.argmin(dists))
    md = float(dists[tid])
    if md > radius:
        return -1, 0.0
    other = float(dists[1 - tid])
    margin = other - md
    conf = float(
        np.clip(0.45 * (1.0 - md / (radius + 1e-3)) + 0.55 * (margin / (other + 1e-3)), 0.0, 1.0)
    )
    if conf < TEAM_ASSIGN_CONF:
        return -1, conf
    return tid, conf


def _dist_xy(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _vote_mode(votes: list[int]) -> int:
    """Majority team if ≥VOTE_MIN agree and beat runner-up; else gray."""
    labeled = [int(v) for v in votes if int(v) >= 0]
    if not labeled:
        return -1
    counts = Counter(labeled)
    tid, n = counts.most_common(1)[0]
    runner = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    if n >= VOTE_MIN and n > runner:
        return int(tid)
    return -1


_BOX_CACHE: dict | None = None


def _goal_boxes() -> dict:
    """Pitch 1 south/north goal-box axis-aligned bounds in meters."""
    global _BOX_CACHE
    if _BOX_CACHE is not None:
        return _BOX_CACHE
    lms = pitch1_landmarks(load_pitch1())
    def _aabb(keys):
        xs = [float(lms[k]["xy"][0]) for k in keys]
        ys = [float(lms[k]["xy"][1]) for k in keys]
        return (min(xs), max(xs), min(ys), max(ys))

    _BOX_CACHE = {
        "south": _aabb(
            ["left_box_goal_near", "left_box_goal_far", "left_box_18_near", "left_box_18_far"]
        ),
        "north": _aabb(
            ["right_box_goal_near", "right_box_goal_far", "right_box_18_near", "right_box_18_far"]
        ),
    }
    return _BOX_CACHE


def which_goal_box(xy) -> str | None:
    x, y = float(xy[0]), float(xy[1])
    for name, (x0, x1, y0, y1) in _goal_boxes().items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def apply_goal_box_prior(players: list[dict]) -> None:
    """Unsure in box stays gray; clear kit only if alone in that Pitch 1 box."""
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
            continue  # alone + clear → keep (cheap GK)
        for m in members:
            if int(m.get("team", -1)) < 0:
                continue  # unsure stays gray
            if len(labeled) > 1:
                teams = {int(x["team"]) for x in labeled}
                if len(teams) > 1:
                    m["team"] = -1


class TeamSession:
    """Session-locked team model: fit once, EMA, sticky, vote buffer, box prior."""

    def __init__(self):
        self.centroids: np.ndarray | None = None
        self.radius: float | None = None
        self.prev: list[dict] = []
        self.prev_fused: list[dict] = []

    def reset(self) -> None:
        self.centroids = None
        self.radius = None
        self.prev = []
        self.prev_fused = []

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
        """Vote buffer + gray-fill sticky + short hold + Pitch 1 goal-box prior.

        Do not force opposite kit labels to the prior before voting — that
        collapses Team 1 into Team 0 when sticky radius is large.
        """
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
            if near is None:
                continue
            near["matched"] = True
            prior_votes = list(near.get("votes") or [])
            obs = int(c["team"])
            prior = int(near.get("team", -1))
            # Gray-fill only — never overwrite a clear opposite kit observation
            if obs < 0 and prior >= 0:
                obs = prior
            votes = (prior_votes + [obs])[-VOTE_LEN:]
            c["votes"] = votes
            voted = _vote_mode(votes)
            if voted >= 0:
                c["team"] = voted
            elif prior >= 0 and int(c["team"]) < 0:
                c["team"] = prior
            # else keep raw obs (allows both kits to appear on first sightings)
        held = []
        for q in self.prev_fused:
            if q.get("matched"):
                continue
            age = int(q.get("age", 0)) + 1
            if age > HOLD_MAX_GAP:
                continue
            if int(q.get("team", -1)) < 0:
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
        out_dicts = cur + held
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
            if fr is None or bbox is None:
                continue
            crop = torso_crop(fr, bbox)
            feat = jersey_feature(crop) if crop is not None else None
            if feat is None:
                continue
            feats.append(feat)
            idxs.append(i)
        if self.centroids is None or self.radius is None:
            fit = fit_team_centroids(feats)
            if fit is None:
                return player_pts
            self.centroids, self.radius = fit
        labs = []
        for j, i in enumerate(idxs):
            tid, conf = assign_from_feature(feats[j], self.centroids, self.radius)
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
