"""Shared team ID core: jersey features, fit, assign, Pitch 1 spatial helpers.

Used by live review (team_live) and batch tracklet export (team_tracklet).
Precision-first: unsure → team_id -1. No FIFA pitch constants.
"""
from __future__ import annotations

import json
import sys
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
TEAM_MIN_TRACKLETS = 15
TEAM_ASSIGN_CONF = 0.55
OUTLIER_MEDIAN_MULT = 2.4
MIN_JERSEY_FRAC = 0.08
MIN_CROP_STD = 4.0
HUE_BINS = 10
KIT_DIM = 3
FEAT_BASE = 5
FISHEYE_CAMS = frozenset({"P7", "P8", "P9", "P10"})
FISHEYE_EDGE_FRAC = 0.20
MAHALANOBIS_CHI2 = 9.21  # p ~ 0.01 for 3 kit dims
USE_CIEDE2000_KIT = False  # A/B via TEAM_USE_CIEDE2000=1 env

_BOX_CACHE: dict | None = None
_CAM_XY_CACHE: dict[str, tuple[float, float]] | None = None


def match3_cam_xy() -> dict[str, tuple[float, float]]:
    """Approximate camera positions on Pitch 1 (meters), diagram chip layout."""
    global _CAM_XY_CACHE
    if _CAM_XY_CACHE is not None:
        return _CAM_XY_CACHE
    rec = load_pitch1()
    hx = float(rec["length_m"]) / 2.0
    hy = float(rec["marks"]["south"]["width_m"]) / 2.0
    _CAM_XY_CACHE = {
        "P1": (-hx - 8.0, 0.0),
        "P6": (hx + 8.0, 0.0),
        "P10": (-hx * 0.45, hy + 8.0),
        "P7": (-hx * 0.45, -hy - 8.0),
        "P8": (hx * 0.45, -hy - 10.0),
        "P9": (hx * 0.4, hy + 8.0),
        "P_Goal1": (-hx - 12.0, hy * 0.55),
        "P_Goal2": (hx + 12.0, -hy * 0.55),
    }
    return _CAM_XY_CACHE


def fisheye_radial_norm(bbox, frame_wh: tuple[int, int]) -> float:
    """0 = image center, 1 = corner (proxy for fisheye distortion severity)."""
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    dx = (cx - fw * 0.5) / max(fw * 0.5, 1.0)
    dy = (cy - fh * 0.5) / max(fh * 0.5, 1.0)
    return float(min(1.0, (dx * dx + dy * dy) ** 0.5))


def fisheye_center_weight(cam: str | None, bbox, frame_wh: tuple[int, int] | None) -> float:
    """Down-weight P7–P10 detections near lens periphery."""
    if cam not in FISHEYE_CAMS or frame_wh is None:
        return 1.0
    r = fisheye_radial_norm(bbox, frame_wh)
    if r <= 1.0 - FISHEYE_EDGE_FRAC:
        return 1.0
    t = (r - (1.0 - FISHEYE_EDGE_FRAC)) / max(FISHEYE_EDGE_FRAC, 1e-3)
    return float(max(0.0, 1.0 - t * t))


def cam_to_player_m(cam: str | None, xy: tuple[float, float]) -> float:
    if not cam:
        return 50.0
    pos = match3_cam_xy().get(cam)
    if pos is None:
        return 50.0
    return float(((xy[0] - pos[0]) ** 2 + (xy[1] - pos[1]) ** 2) ** 0.5)


def team_vote_weight(
    cam: str | None,
    bbox,
    frame_wh: tuple[int, int] | None,
    xy: tuple[float, float],
    crop_valid: bool,
) -> float:
    if not crop_valid:
        return 0.0
    w_fish = fisheye_center_weight(cam, bbox, frame_wh)
    w_dist = 1.0 / (1.0 + cam_to_player_m(cam, xy))
    return w_fish * w_dist


def torso_crop(
    frame: np.ndarray,
    bbox,
    cam: str | None = None,
    frame_wh: tuple[int, int] | None = None,
) -> Optional[np.ndarray]:
    """Upper-mid torso; narrow width on fisheye periphery."""
    if frame is None or frame.size == 0:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w < 10 or h < 24:
        return None
    fh, fw = frame.shape[:2]
    wh = frame_wh or (fw, fh)
    x_inset_lo, x_inset_hi = 0.12, 0.88
    if cam in FISHEYE_CAMS and fisheye_radial_norm(bbox, wh) >= 1.0 - FISHEYE_EDGE_FRAC:
        x_inset_lo, x_inset_hi = 0.30, 0.70
    y0 = max(0, int(y + 0.15 * h))
    y1 = min(fh, int(y + 0.48 * h))
    x0 = max(0, int(x + x_inset_lo * w))
    x1 = min(fw, int(x + x_inset_hi * w))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return frame[y0:y1, x0:x1]


def _adaptive_non_green(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
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
    return np.concatenate([feat[:KIT_DIM], feat[FEAT_BASE : FEAT_BASE + HUE_BINS]])


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(a.astype(np.float64), 1e-8, 1.0)
    b = np.clip(b.astype(np.float64), 1e-8, 1.0)
    bc = float(np.sum(np.sqrt(a * b)))
    return float(-np.log(max(bc, 1e-8)))


def _ciede2000_scalar(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Simplified ΔE00 on mean Lab* (kit proxy — A/B only)."""
    la = lab_a.astype(np.float64)
    lb = lab_b.astype(np.float64)
    dL = la[0] - lb[0]
    C1 = (la[1] ** 2 + la[2] ** 2) ** 0.5
    C2 = (lb[1] ** 2 + lb[2] ** 2) ** 0.5
    dC = C1 - C2
    dH = ((la[1] - lb[1]) ** 2 + (la[2] - lb[2]) ** 2 - dC ** 2) ** 0.5
    return float((dL / 1.0) ** 2 + (dC / (1.0 + 0.045 * (C1 + C2))) ** 2 + (dH / (1.0 + 0.015 * (C1 + C2))) ** 2) ** 0.5


def _kit_lab_proxy(feat: np.ndarray) -> np.ndarray:
    """Rough Lab* from kit fractions (for CIEDE2000 A/B)."""
    b, w, y = float(feat[0]), float(feat[1]), float(feat[2])
    L = 40.0 + 180.0 * (w + 0.5 * y)
    a = -10.0 + 40.0 * y - 20.0 * b
    bb = -30.0 + 80.0 * b - 10.0 * w
    return np.array([L, a, bb], dtype=np.float64)


def feature_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    """Kit fractions L2 or optional CIEDE2000 + Bhattacharyya on hue histogram."""
    import os

    use_ciede = USE_CIEDE2000_KIT or os.environ.get("TEAM_USE_CIEDE2000") == "1"
    if use_ciede:
        kit_d = _ciede2000_scalar(_kit_lab_proxy(fa), _kit_lab_proxy(fb))
    else:
        kit_d = float(np.linalg.norm(fa[:KIT_DIM] - fb[:KIT_DIM]))
    ha = fa[FEAT_BASE : FEAT_BASE + HUE_BINS]
    hb = fb[FEAT_BASE : FEAT_BASE + HUE_BINS]
    hist_d = _bhattacharyya(ha, hb)
    return kit_d + 0.35 * hist_d


def jersey_feature(crop: np.ndarray) -> Optional[np.ndarray]:
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
    hist = hist / (float(hist.sum()) + 1e-6)
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
    score = centroids[:, 0] - centroids[:, 1] - 0.5 * centroids[:, 2]
    order = np.argsort(-score)
    return centroids[order]


def fit_match_centroids(
    features: list[np.ndarray],
    min_crops: int = TEAM_MIN_CROPS,
) -> tuple[np.ndarray, float] | None:
    """K=2 fit with label lock. Alias: fit_team_centroids."""
    if len(features) < min_crops:
        return None
    x = np.asarray(features, dtype=np.float32)
    xs = np.stack([_color_subspace(f) for f in x], axis=0)
    dmat = np.zeros((len(x), len(x)), dtype=np.float32)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = feature_distance(x[i], x[j])
            dmat[i, j] = dmat[j, i] = d
    i0, i1 = np.unravel_index(int(np.argmax(dmat)), dmat.shape)
    if i0 == i1:
        return None
    c0, c1 = x[i0].copy(), x[i1].copy()
    for _ in range(20):
        d0 = np.array([feature_distance(f, c0) for f in x])
        d1 = np.array([feature_distance(f, c1) for f in x])
        lab = (d1 < d0).astype(np.int32)
        if lab.min() == lab.max():
            c1 = x[int(np.argmax(d0))].copy()
            continue
        c0 = x[lab == 0].mean(axis=0)
        c1 = x[lab == 1].mean(axis=0)
    cents = _lock_labels(np.stack([c0, c1], axis=0))
    dmin = np.array(
        [min(feature_distance(f, cents[0]), feature_distance(f, cents[1])) for f in x]
    )
    radius = float(np.median(dmin) * OUTLIER_MEDIAN_MULT + 1e-3)
    sep = float(np.linalg.norm(cents[0, :KIT_DIM] - cents[1, :KIT_DIM]))
    if sep < 0.12:
        return None
    return cents, max(radius, 0.08)


fit_team_centroids = fit_match_centroids


def _mahalanobis_kit(feat: np.ndarray, cents: np.ndarray) -> tuple[float, int]:
    kit = feat[:KIT_DIM].astype(np.float64)
    d0 = float(np.linalg.norm(kit - cents[0, :KIT_DIM]))
    d1 = float(np.linalg.norm(kit - cents[1, :KIT_DIM]))
    tid = 0 if d0 <= d1 else 1
    return min(d0, d1), tid


def is_photometric_outlier(feat: np.ndarray, cents: np.ndarray, radius: float) -> bool:
    md, _ = _mahalanobis_kit(feat, cents)
    return md > radius * 1.35


def assign_feature(
    feature: np.ndarray,
    centroids: np.ndarray,
    radius: float,
    position_xy: tuple[float, float] | None = None,
) -> tuple[int, float]:
    """Return (team_id, confidence). -1 = gray."""
    blue, white, yellow = float(feature[0]), float(feature[1]), float(feature[2])
    light = white + yellow
    if blue >= 0.38 and blue >= light + 0.12:
        return 0, min(0.95, 0.55 + blue)
    if light >= 0.38 and light >= blue + 0.12:
        return 1, min(0.95, 0.55 + light)
    if float(feature[4]) < 70.0 and blue < 0.25 and light < 0.25:
        return -1, 0.0
    if is_photometric_outlier(feature, centroids, radius):
        if position_xy and which_goal_box(position_xy) is not None:
            return -1, 0.0
        return -1, 0.0
    dists = np.array([feature_distance(feature, c) for c in centroids])
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


assign_from_feature = assign_feature


def tracklet_median_feature(feats: list[np.ndarray]) -> np.ndarray | None:
    if not feats:
        return None
    return np.median(np.stack(feats, axis=0), axis=0).astype(np.float32)


def save_centroids(path: Path, centroids: np.ndarray, radius: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "centroids": centroids.tolist(),
        "radius": float(radius),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_centroids(path: Path) -> tuple[np.ndarray, float] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["centroids"], dtype=np.float32), float(data["radius"])


def _goal_boxes() -> dict:
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


def in_pitch1_goal_box(position_xy: tuple[float, float]) -> bool:
    return which_goal_box(position_xy) is not None
