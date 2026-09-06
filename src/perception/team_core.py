"""Shared team ID core: jersey features, fit, assign, Pitch 1 spatial helpers.

Used by live review (team_live) and batch tracklet export (team_tracklet).
Precision-first: unsure → team_id -1. No FIFA pitch constants.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
TEAM_MIN_CROPS_AUTO = 20
TEAM_MIN_TRACKLETS = 15
TEAM_ASSIGN_CONF = 0.55
KIT_MODE_MATCH3 = "match3"
KIT_MODE_AUTO = "auto"
STICKY_FLIP_CONF_AUTO = 0.65
AUTO_BLUE_FRAC = 0.20
AUTO_WHITE_FRAC = 0.22
PIXEL_NUDGE_MARGIN = 0.12
PIXEL_NUDGE_MIN_CONF = 0.68
AUTO_LIGHT_FRAC = 0.35
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
# Undershirt A/B: hard center 50×50 (fallback when annulus off).
JERSEY_CENTER_FRAC = 0.50
# N1 product: zero-weight outer 30% ring (r > 0.70); matches ideas10 annulus30.
JERSEY_ANNULUS_OUTER = 0.30
# Dual-color rescue: both blue+white high → white kit (undershirt).
DUAL_COLOR_FRAC = 0.35
USE_JERSEY_ANNULUS = True
USE_JERSEY_CENTER = True  # used only when USE_JERSEY_ANNULUS is False
USE_DUAL_TO_WHITE = True
# N2: median of last N jersey features before assign (anti-flicker).
MEDIAN_FEAT_LEN = 5
# N4: auto dual-color → white seed bank, then freeze centroids (no human kit-ref).
# Off by default — A/B failed score gate (kit_n4_auto_dual_seed_ab); opt-in for experiments.
USE_AUTO_DUAL_SEED = False
AUTO_DUAL_SEED_MIN = 5
BLUE_SEED_MARGIN = 0.08  # blue must beat white by this for T0 seed
# N5: hard-drop fisheye-periphery crops from centroid *fit* (assign still uses them).
# Off by default — A/B failed score gate (kit_n5_fisheye_drop_ab); opt-in for experiments.
USE_FISHEYE_EDGE_DROP = False
# N6: if outer ring much bluer than core, use core-only feats (undershirt sleeves).
# Off by default — A/B failed flips gate (kit_n6_center_edge_ab); opt-in for experiments.
USE_CENTER_EDGE_VETO = False
EDGE_BLUE_DELTA = 0.18
# N7: dual→white only when white≥blue (avoid false-white when blue dominates dual).
# Off by default — A/B failed score/flips (kit_n7_gated_dual_ab); opt-in for experiments.
USE_GATED_DUAL_TO_WHITE = False
# N9: Match3→Match4 centroid transfer via per-dim HSV/hist affine (no human labels).
# Off by default — A/B kit_n9_centroid_transfer_ab; opt-in for experiments.
USE_MATCH_CENTROID_TRANSFER = False

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


def is_fisheye_edge_crop(
    cam: str | None,
    bbox,
    frame_wh: tuple[int, int] | None,
) -> bool:
    """True if P7–P10 detection sits in the outer FISHEYE_EDGE_FRAC ring."""
    if cam not in FISHEYE_CAMS or frame_wh is None or bbox is None:
        return False
    return fisheye_radial_norm(bbox, frame_wh) >= 1.0 - FISHEYE_EDGE_FRAC


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


def _center_crop_frac(crop: np.ndarray, frac: float) -> np.ndarray:
    """Keep central frac×frac of torso (hard_center_50 undershirt A/B)."""
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return crop
    fx = max(0.15, min(float(frac), 1.0))
    x0 = int(w * (0.5 - fx * 0.5))
    x1 = int(w * (0.5 + fx * 0.5))
    y0 = int(h * (0.5 - fx * 0.5))
    y1 = int(h * (0.5 + fx * 0.5))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return crop
    return crop[y0:y1, x0:x1]


def _xy_norm_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    xn = ((xx - cx) / max(cx, 1.0)).astype(np.float32)
    yn = ((yy - cy) / max(cy, 1.0)).astype(np.float32)
    return xn, yn


def _annulus_keep_weight(h: int, w: int, outer_frac: float = JERSEY_ANNULUS_OUTER) -> np.ndarray:
    """1 inside r<=(1-outer), 0 on outer ring — matches ideas10 annulus30."""
    xn, yn = _xy_norm_grid(h, w)
    r = np.sqrt(xn * xn + yn * yn)
    r_max = max(0.15, min(1.0 - float(outer_frac), 0.95))
    return (r <= r_max).astype(np.float32)


def _feat_from_hsv_keep(
    hsv: np.ndarray, keep: np.ndarray, wgt: np.ndarray | None = None
) -> Optional[np.ndarray]:
    """Weighted jersey feature; unweighted if wgt is None (legacy center path)."""
    if wgt is None:
        ww = keep.astype(np.float32)
    else:
        ww = wgt.astype(np.float32) * keep.astype(np.float32)
    mass = float(ww.sum())
    if mass < 18.0 or float(keep.mean()) < MIN_JERSEY_FRAC:
        return None
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = keep & (h >= 85) & (h <= 145) & (s >= 35)
    purple = keep & (h >= 125) & (h <= 170) & (s >= 30)
    white = keep & (s <= 55) & (v >= 125)
    yellow = keep & (h >= 12) & (h <= 40) & (s >= 45) & (v >= 60)
    n = max(mass, 1e-6)
    hist, _ = np.histogram(
        h.astype(np.float32), bins=HUE_BINS, range=(0.0, 180.0), weights=ww
    )
    hist = hist.astype(np.float32)
    hist = hist / (float(hist.sum()) + 1e-6)
    bp = blue | purple
    base = np.array(
        [
            float((bp.astype(np.float32) * ww).sum() / n),
            float((white.astype(np.float32) * ww).sum() / n),
            float((yellow.astype(np.float32) * ww).sum() / n),
            float((s.astype(np.float32) * ww).sum() / n),
            float((v.astype(np.float32) * ww).sum() / n),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, hist])


def jersey_feature(crop: np.ndarray) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return None
    # Annulus wins over hard center crop (A/B used full torso + r<=0.70 mask).
    if not USE_JERSEY_ANNULUS and USE_JERSEY_CENTER:
        crop = _center_crop_frac(crop, JERSEY_CENTER_FRAC)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = _adaptive_non_green(hsv)
    std = float(crop.std())
    if std < MIN_CROP_STD:
        whiteish = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 125)
        if float(whiteish.mean()) < 0.45:
            return None
    hgt, wdt = crop.shape[:2]
    xn, yn = _xy_norm_grid(hgt, wdt)
    r = np.sqrt(xn * xn + yn * yn)
    # N6: outer much bluer than core → sleeves/undershirt; trust core only.
    if USE_CENTER_EDGE_VETO:
        f_core = _feat_from_hsv_keep(hsv, keep, (r <= 0.45).astype(np.float32))
        f_edge = _feat_from_hsv_keep(hsv, keep, (r >= 0.65).astype(np.float32))
        if (
            f_core is not None
            and f_edge is not None
            and float(f_edge[0] - f_core[0]) > EDGE_BLUE_DELTA
        ):
            return f_core
    if USE_JERSEY_ANNULUS:
        wgt = _annulus_keep_weight(hgt, wdt, JERSEY_ANNULUS_OUTER)
        return _feat_from_hsv_keep(hsv, keep, wgt)
    frac = float(keep.mean())
    if frac < MIN_JERSEY_FRAC or int(keep.sum()) < 18:
        return None
    return _feat_from_hsv_keep(hsv, keep, None)


def _lock_labels(centroids: np.ndarray) -> np.ndarray:
    """Match 3: Team 0 = bluer kit, Team 1 = whiter/yellower."""
    score = centroids[:, 0] - centroids[:, 1] - 0.5 * centroids[:, 2]
    order = np.argsort(-score)
    return centroids[order]


def _lock_labels_auto(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Auto: blue kit = T0 when hue separates; else larger cluster = T0."""
    blue_sep = abs(float(centroids[0, 0]) - float(centroids[1, 0]))
    white_sep = abs(float(centroids[0, 1]) - float(centroids[1, 1]))
    if blue_sep + white_sep >= 0.10:
        return _lock_labels(centroids)
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n1 > n0:
        return centroids[[1, 0]]
    return centroids


def fit_match_centroids(
    features: list[np.ndarray],
    min_crops: int = TEAM_MIN_CROPS,
    kit_mode: str = KIT_MODE_MATCH3,
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
    if kit_mode == KIT_MODE_AUTO:
        cents = _lock_labels_auto(np.stack([c0, c1], axis=0), lab)
    else:
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


def _pixel_team(
    feature: np.ndarray,
    kit_mode: str,
    *,
    use_pixels: bool,
    blue_frac: float,
    white_frac: float,
) -> tuple[int, float] | None:
    if not use_pixels and kit_mode != KIT_MODE_MATCH3:
        return None
    blue, white, yellow = float(feature[0]), float(feature[1]), float(feature[2])
    light = white + yellow
    if kit_mode == KIT_MODE_MATCH3:
        if blue >= 0.38 and blue >= light + 0.12:
            return 0, min(0.95, 0.55 + blue)
        if light >= 0.38 and light >= blue + 0.12:
            return 1, min(0.95, 0.55 + light)
        return None
    if blue >= blue_frac and blue >= light + 0.05:
        return 0, min(0.95, 0.50 + blue)
    if white >= white_frac and white >= blue + 0.05:
        return 1, min(0.95, 0.50 + white)
    if light >= AUTO_LIGHT_FRAC and blue <= 0.18 and light >= blue + 0.10:
        return 1, min(0.95, 0.50 + light)
    return None


def assign_feature(
    feature: np.ndarray,
    centroids: np.ndarray,
    radius: float,
    position_xy: tuple[float, float] | None = None,
    kit_mode: str = KIT_MODE_MATCH3,
    strategy=None,
) -> tuple[int, float]:
    """Return (team_id, confidence). -1 = gray."""
    mode = kit_mode
    use_px = mode in (KIT_MODE_AUTO, KIT_MODE_MATCH3)
    blue_f, white_f = AUTO_BLUE_FRAC, AUTO_WHITE_FRAC
    pixel_agree = False
    no_gray = False
    soft_nudge = False
    if strategy is not None:
        mode = strategy.kit_mode
        use_px = strategy.use_jersey_pixels or mode == KIT_MODE_MATCH3
        blue_f = strategy.auto_blue_frac
        white_f = strategy.auto_white_frac
        pixel_agree = strategy.pixel_cluster_agree
        no_gray = bool(strategy.no_gray)
        soft_nudge = bool(strategy.soft_pixel_nudge)
    blue, white, yellow = float(feature[0]), float(feature[1]), float(feature[2])
    light = white + yellow
    # Dual undershirt hint — soft conf so session sticky/vote can hold identity.
    dual = blue >= DUAL_COLOR_FRAC and white >= DUAL_COLOR_FRAC
    if USE_GATED_DUAL_TO_WHITE:
        dual = dual and white >= blue
    if USE_DUAL_TO_WHITE and dual:
        pixel = (1, 0.58)
    else:
        pixel = _pixel_team(
            feature, mode, use_pixels=use_px, blue_frac=blue_f, white_frac=white_f,
        )
    if pixel is not None and not pixel_agree and not no_gray:
        return pixel
    dists = np.array([feature_distance(feature, c) for c in centroids])
    tid = int(np.argmin(dists))
    md = float(dists[tid])
    other = float(dists[1 - tid])
    margin = other - md
    conf = float(
        np.clip(0.45 * (1.0 - md / (radius + 1e-3)) + 0.55 * (margin / (other + 1e-3)), 0.0, 1.0)
    )
    if no_gray:
        out_tid, out_conf = int(tid), max(conf, 0.45)
        dual_white = blue >= DUAL_COLOR_FRAC and white >= DUAL_COLOR_FRAC
        if USE_GATED_DUAL_TO_WHITE:
            dual_white = dual_white and white >= blue
        if (soft_nudge or use_px) and pixel is not None and not pixel_agree:
            ptid, pconf = int(pixel[0]), float(pixel[1])
            strong_white = white >= white_f and white >= blue + 0.06
            strong_blue = blue >= blue_f and blue >= light + 0.06
            can_nudge = margin < PIXEL_NUDGE_MARGIN and pconf >= PIXEL_NUDGE_MIN_CONF
            white_rescue = ptid == 1 and (strong_white or dual_white) and margin < 0.28
            if ptid != out_tid and (can_nudge or white_rescue):
                if ptid == 1 and (strong_white or dual_white):
                    # Strong enough to win track *birth*; below 0.92 sticky flip gate.
                    out_tid, out_conf = ptid, 0.70 if dual_white else min(0.62, max(pconf, out_conf * 0.85))
                elif ptid == 0 and strong_blue and can_nudge:
                    out_tid, out_conf = ptid, max(pconf, out_conf * 0.9)
            elif ptid == out_tid:
                out_conf = max(out_conf, pconf)
        return out_tid, out_conf
    if float(feature[4]) < 70.0 and blue < 0.25 and light < 0.25:
        return -1, 0.0
    if is_photometric_outlier(feature, centroids, radius):
        if position_xy and which_goal_box(position_xy) is not None:
            return -1, 0.0
        return -1, 0.0
    if md > radius:
        return -1, 0.0
    if conf < TEAM_ASSIGN_CONF:
        return -1, conf
    if pixel_agree:
        if pixel is None or int(pixel[0]) != int(tid):
            return -1, min(conf, 0.4)
        return int(tid), max(conf, float(pixel[1]))
    return tid, conf


assign_from_feature = assign_feature


def tracklet_median_feature(feats: list[np.ndarray]) -> np.ndarray | None:
    if not feats:
        return None
    return np.median(np.stack(feats, axis=0), axis=0).astype(np.float32)


def centroids_from_labeled(
    team_feats: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, float] | None:
    """Fit team centroids from explicit Team 0 / Team 1 jersey features."""
    t0 = team_feats.get(0) or []
    t1 = team_feats.get(1) or []
    if len(t0) < 1 or len(t1) < 1:
        return None
    c0 = np.mean(np.stack(t0, axis=0), axis=0).astype(np.float32)
    c1 = np.mean(np.stack(t1, axis=0), axis=0).astype(np.float32)
    cents = np.stack([c0, c1], axis=0)
    all_feats = t0 + t1
    dmin = [
        min(feature_distance(f, cents[0]), feature_distance(f, cents[1]))
        for f in all_feats
    ]
    radius = float(np.median(dmin) * OUTLIER_MEDIAN_MULT + 1e-3)
    sep = float(np.linalg.norm(cents[0, :KIT_DIM] - cents[1, :KIT_DIM]))
    if sep < 0.12:
        return None
    return cents, max(radius, 0.08)


def is_dual_color_feat(feat: np.ndarray) -> bool:
    """Both blue and white fracs high — undershirt / mixed torso."""
    return float(feat[0]) >= DUAL_COLOR_FRAC and float(feat[1]) >= DUAL_COLOR_FRAC


def is_blue_seed_feat(feat: np.ndarray) -> bool:
    """Clear blue-dominant crop for Team 0 seed (not dual)."""
    b, w = float(feat[0]), float(feat[1])
    if is_dual_color_feat(feat):
        return False
    return b >= DUAL_COLOR_FRAC and b >= w + BLUE_SEED_MARGIN


def is_white_seed_feat(feat: np.ndarray) -> bool:
    """Dual-color or strong white for Team 1 seed."""
    if is_dual_color_feat(feat):
        return True
    b, w = float(feat[0]), float(feat[1])
    return w >= DUAL_COLOR_FRAC and w >= b + BLUE_SEED_MARGIN


def fit_auto_dual_seed_centroids(
    feats: list[np.ndarray],
    min_per_team: int = AUTO_DUAL_SEED_MIN,
) -> tuple[np.ndarray, float] | None:
    """N4: T0=blue-dom seeds, T1=dual+white seeds → freeze-ready centroids."""
    t0 = [f for f in feats if f is not None and is_blue_seed_feat(f)]
    t1 = [f for f in feats if f is not None and is_white_seed_feat(f)]
    if len(t0) < min_per_team or len(t1) < min_per_team:
        return None
    # Cap bags so one side doesn't dominate the mean.
    rng = np.random.RandomState(0)
    if len(t0) > 200:
        t0 = [t0[i] for i in rng.choice(len(t0), 200, replace=False)]
    if len(t1) > 200:
        t1 = [t1[i] for i in rng.choice(len(t1), 200, replace=False)]
    return centroids_from_labeled({0: t0, 1: t1})


def feat_bank_moments(
    feats: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean / std of a jersey-feature bank (per dim)."""
    clean = [np.asarray(f, np.float32) for f in feats if f is not None]
    if len(clean) < 8:
        return None
    x = np.stack(clean, axis=0)
    mu = x.mean(axis=0).astype(np.float32)
    sig = np.maximum(x.std(axis=0).astype(np.float32), 1e-3)
    return mu, sig


def transfer_centroids_affine(
    src_centroids: np.ndarray,
    src_mu: np.ndarray,
    src_sig: np.ndarray,
    dst_mu: np.ndarray,
    dst_sig: np.ndarray,
) -> np.ndarray:
    """Map source-match centroids into dest feature space (HSV + hue hist).

    Per-dim: x' = (x - μ_s) * (σ_d / σ_s) + μ_d, then re-lock Team0=bluer.
    """
    scale = dst_sig / src_sig
    out = (np.asarray(src_centroids, np.float32) - src_mu) * scale + dst_mu
    return _lock_labels(out.astype(np.float32))


def radius_from_feats(
    feats: list[np.ndarray],
    centroids: np.ndarray,
) -> float:
    """Outlier radius from dest bank vs transferred centroids."""
    clean = [np.asarray(f, np.float32) for f in feats if f is not None]
    if not clean:
        return 0.12
    dmin = [
        min(feature_distance(f, centroids[0]), feature_distance(f, centroids[1]))
        for f in clean
    ]
    return max(float(np.median(dmin) * OUTLIER_MEDIAN_MULT + 1e-3), 0.08)


def transfer_match_centroids(
    src_centroids: np.ndarray,
    src_feats: list[np.ndarray],
    dst_feats: list[np.ndarray],
    src_radius: float | None = None,
) -> tuple[np.ndarray, float] | None:
    """N9: adapt source-match centroids to dest lighting via HSV/hist affine."""
    src_m = feat_bank_moments(src_feats)
    dst_m = feat_bank_moments(dst_feats)
    if src_m is None or dst_m is None:
        return None
    cents = transfer_centroids_affine(
        src_centroids, src_m[0], src_m[1], dst_m[0], dst_m[1]
    )
    sep = float(np.linalg.norm(cents[0, :KIT_DIM] - cents[1, :KIT_DIM]))
    if sep < 0.12:
        return None
    rad = radius_from_feats(dst_feats, cents)
    if src_radius is not None:
        # Blend transferred radius with source scale of mean σ ratio.
        scale = float(np.mean(dst_m[1][:KIT_DIM] / src_m[1][:KIT_DIM]))
        rad = max(rad, float(src_radius) * max(scale, 0.5))
    return cents, rad


def kit_feat_preview_bgr(feat: np.ndarray) -> np.ndarray:
    """Approximate kit color swatch from jersey feature fractions."""
    blue = float(feat[0])
    white = float(feat[1])
    yellow = float(feat[2])
    b = int(40 + 180 * blue)
    g = int(40 + 160 * white + 80 * yellow)
    r = int(40 + 160 * white + 180 * yellow)
    swatch = np.zeros((48, 48, 3), dtype=np.uint8)
    swatch[:, :] = (b, g, r)
    return swatch


def save_centroids(path: Path, centroids: np.ndarray, radius: float, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "centroids": centroids.tolist(),
        "radius": float(radius),
    }
    payload.update(meta)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_kit_ref(
    path: Path,
    centroids: np.ndarray,
    radius: float,
    *,
    team_names: tuple[str, str] = ("Team 0", "Team 1"),
    kit_mode: str = KIT_MODE_AUTO,
    n_samples: tuple[int, int] = (0, 0),
    source: str = "kit_label_dashboard",
) -> None:
    save_centroids(
        path,
        centroids,
        radius,
        team_names=list(team_names),
        kit_mode=kit_mode,
        n_samples=list(n_samples),
        source=source,
    )


def kit_samples_bank_path(centroids_path: Path) -> Path:
    return Path(centroids_path).with_name("kit_samples_bank.json")


def backup_kit_ref(path: Path) -> Path | None:
    """Copy existing centroids (+ bank if present) to a timestamped backup."""
    path = Path(path)
    if not path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"team_centroids_backup_{stamp}.json")
    backup.write_bytes(path.read_bytes())
    bank = kit_samples_bank_path(path)
    if bank.is_file():
        bank.with_name(f"kit_samples_bank_backup_{stamp}.json").write_bytes(
            bank.read_bytes()
        )
    return backup


def samples_to_bank_rows(samples: list[dict]) -> list[dict]:
    """Persistable rows: features + ids (no full crop pixels)."""
    rows = []
    for s in samples:
        rows.append(
            {
                "team": int(s["team"]),
                "feat": [float(x) for x in s["feat"]],
                "frame_id": int(s.get("frame_id") or 0),
                "tag": str(s.get("tag") or ""),
                "slot_key": str(s.get("slot_key") or ""),
            }
        )
    return rows


def save_kit_samples_bank(path: Path, samples: list[dict], *, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "kit_label_dashboard",
        "n_samples": [
            sum(1 for s in samples if int(s["team"]) == 0),
            sum(1 for s in samples if int(s["team"]) == 1),
        ],
        "samples": samples_to_bank_rows(samples),
    }
    if meta:
        payload.update(meta)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_kit_samples_bank(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("samples") or []
    out = []
    for s in rows:
        feat = s.get("feat")
        if feat is None:
            continue
        out.append(
            {
                "team": int(s["team"]),
                "feat": [float(x) for x in feat],
                "frame_id": int(s.get("frame_id") or 0),
                "tag": str(s.get("tag") or ""),
                "slot_key": str(s.get("slot_key") or ""),
            }
        )
    return out


def merge_kit_sample_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Union by slot_key (new wins); rows without slot_key are always kept."""
    by_key: dict[str, dict] = {}
    extras: list[dict] = []
    for s in existing + new_rows:
        key = str(s.get("slot_key") or "").strip()
        if not key:
            extras.append(s)
            continue
        by_key[key] = s
    return list(by_key.values()) + extras


def load_centroids(path: Path) -> tuple[np.ndarray, float] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["centroids"], dtype=np.float32), float(data["radius"])


def load_kit_ref_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: data[k]
        for k in ("team_names", "kit_mode", "n_samples", "source")
        if k in data
    }


def is_kit_ref(meta_or_path) -> bool:
    """True when file/meta came from kit label dashboard (or carries kit fields)."""
    if isinstance(meta_or_path, (str, Path)):
        meta = load_kit_ref_meta(Path(meta_or_path))
    else:
        meta = meta_or_path or {}
    if not meta:
        return False
    if meta.get("source") == "kit_label_dashboard":
        return True
    return any(k in meta for k in ("team_names", "kit_mode", "n_samples"))


def resolve_kit_centroids_path(
    cfg: dict | None,
    output_root: Path | str | None = None,
    run_dir: Path | str | None = None,
) -> Path | None:
    """Prefer config path → match-level → run_dir team_centroids.json."""
    ta = (cfg or {}).get("team_assignment") or {}
    raw = (ta.get("kit_centroids_path") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    if output_root is not None:
        match_level = Path(output_root) / "team_centroids.json"
        if match_level.is_file():
            return match_level
    if run_dir is not None:
        run_path = Path(run_dir) / "team_centroids.json"
        if run_path.is_file():
            return run_path
    return None


def find_kit_ref_under(output_root: Path | str) -> Path | None:
    """First cam/match file under output_root with kit_label_dashboard source."""
    root = Path(output_root)
    if not root.is_dir():
        return None
    match_level = root / "team_centroids.json"
    if match_level.is_file() and is_kit_ref(match_level):
        return match_level
    for path in sorted(root.glob("*/team_centroids.json")):
        if is_kit_ref(path):
            return path
    return None


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
