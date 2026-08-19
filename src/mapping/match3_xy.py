"""Match 3: pixel → Pitch 1 meters, then fuse ball (x, y) across cameras."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.mapping.pitch_bounds import in_pitch_bounds

ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
AGREE_M = 4.0
EMIT_CONF = 0.80
MARGIN_M = 1.0
SUPPORT_PX = 180.0
# Soft hull (H1): 0.25 keeps midfield maps; emit gate still EMIT_CONF ≥ 0.80.
MIN_SUPPORT = 0.25
# F3: drop weak maps that disagree with the max-conf anchor (ghost prune).
GHOST_CONF = 0.45
MATCH3_CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]


def calib_path(cam: str) -> Path:
    return CALIB_DIR / f"{cam}_manual.json"


def load_calib(cam: str) -> dict | None:
    path = calib_path(cam)
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    H = rec.get("homography") or rec.get("H")
    if H is None:
        return None
    rec["H"] = np.asarray(H, dtype=float)
    rec["camera"] = cam
    return rec


def load_calib_for_video(path) -> dict | None:
    gold = str(ROOT / "scripts" / "gold_set")
    if gold not in sys.path:
        sys.path.insert(0, gold)
    from raw_cam_id import cam_id_from_raw_name

    try:
        cam = cam_id_from_raw_name(Path(path).name)
    except ValueError:
        return None
    if cam not in MATCH3_CAMS:
        return None
    return load_calib(cam)


def scale_px(x: float, y: float, frame_wh, calib_wh) -> tuple[float, float]:
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    cw, ch = float(calib_wh[0]), float(calib_wh[1])
    if fw < 1 or fh < 1:
        return x, y
    return x * (cw / fw), y * (ch / fh)


def bbox_foot(box) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in box]
    return x + w / 2.0, y + h


def apply_H(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    v = H @ np.array([x, y, 1.0], dtype=float)
    if abs(v[2]) < 1e-8:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def calib_undistort_params(calib: dict) -> dict | None:
    """Brown undistort locked with landmarks (P7–P10). None = raw H path."""
    u = calib.get("undistort")
    if not u:
        return None
    return {
        "k1": float(u.get("k1", 0.0)),
        "k2": float(u.get("k2", 0.0)),
        "p1": float(u.get("p1", 0.0)),
        "p2": float(u.get("p2", 0.0)),
        "alpha": float(u.get("alpha", 0.5)),
    }


def undistort_px(x: float, y: float, w: float, h: float, params: dict) -> tuple[float, float]:
    """Same Brown recipe as match3_landmarks stills (getOptimalNewCameraMatrix + undistort)."""
    w_i, h_i = int(round(w)), int(round(h))
    if w_i < 2 or h_i < 2:
        return float(x), float(y)
    k_mat = np.array(
        [[w_i, 0.0, w_i / 2.0], [0.0, w_i, h_i / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(
        [
            float(params["k1"]),
            float(params["k2"]),
            float(params["p1"]),
            float(params["p2"]),
            0.0,
        ],
        dtype=np.float64,
    )
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        k_mat, dist, (w_i, h_i), float(params["alpha"]), (w_i, h_i)
    )
    pts = cv2.undistortPoints(
        np.array([[[float(x), float(y)]]], dtype=np.float64),
        k_mat,
        dist,
        P=new_k,
    )
    return float(pts[0, 0, 0]), float(pts[0, 0, 1])


def hull_support(px: float, py: float, image_points: list) -> float:
    pts = np.asarray(image_points, dtype=np.float32)
    if len(pts) < 3:
        d = min(float(np.hypot(px - p[0], py - p[1])) for p in pts)
        return max(0.0, 1.0 - d / SUPPORT_PX)
    hull = cv2.convexHull(pts)
    dist = float(cv2.pointPolygonTest(hull, (float(px), float(py)), True))
    if dist >= 0:
        return 1.0
    return max(0.0, 1.0 + dist / SUPPORT_PX)


def combined_conf(confs: list[float]) -> float:
    miss = 1.0
    for c in confs:
        miss *= (1.0 - min(0.999, max(0.0, float(c))))
    return 1.0 - miss


def hull_points(calib: dict) -> list:
    """Image points for support hull. Optional hull_image_points expand FOV without refitting H."""
    extra = calib.get("hull_image_points")
    if extra:
        return list(extra)
    return list(calib.get("image_points") or [])


def map_ball_box(
    calib: dict,
    box,
    conf: float,
    frame_wh=None,
    *,
    apply_undistort: bool | None = None,
) -> dict | None:
    """Map detection box foot to Pitch 1 meters.

    If calib has ``undistort`` (defished landmark H), raw detection pixels are
    undistorted with the same Brown params before H / hull support.
    """
    wh = frame_wh or calib.get("image_wh") or [1920, 1080]
    fx, fy = bbox_foot(box)
    px, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
    params = calib_undistort_params(calib)
    use_u = bool(params) if apply_undistort is None else bool(apply_undistort and params)
    if use_u:
        cw, ch = calib.get("image_wh") or wh
        px, py = undistort_px(px, py, cw, ch, params)
    xy = apply_H(calib["H"], px, py)
    if xy is None:
        return None
    if not in_pitch_bounds(xy[0], xy[1], margin_m=MARGIN_M):
        return None
    support = hull_support(px, py, hull_points(calib))
    if support < MIN_SUPPORT:
        return None
    c = float(conf)
    return {
        "cam": calib["camera"],
        "xy": xy,
        "conf": c,
        "support": support,
        "weight": c * support,
    }


def _median_xy(rows: list[dict]) -> tuple[float, float]:
    xs = sorted(r["xy"][0] for r in rows)
    ys = sorted(r["xy"][1] for r in rows)
    mid = len(rows) // 2
    if len(rows) % 2:
        return xs[mid], ys[mid]
    return (xs[mid - 1] + xs[mid]) / 2, (ys[mid - 1] + ys[mid]) / 2


def _near(a: dict, b: dict) -> bool:
    dx = a["xy"][0] - b["xy"][0]
    dy = a["xy"][1] - b["xy"][1]
    return (dx * dx + dy * dy) ** 0.5 <= AGREE_M


def prune_ghost_maps(
    rows: list[dict],
    *,
    enabled: bool = True,
    ghost_conf: float = GHOST_CONF,
) -> list[dict]:
    """F3: keep maps near max-conf anchor, or far maps with conf ≥ ghost_conf.

    Default ghost_conf=0.45 drops weak far P1/P7 ghosts. A/B may use 0.80
    (far cams only if already emit-eligible).
    """
    if not enabled or len(rows) < 2:
        return rows
    floor = float(ghost_conf)
    anchor = max(rows, key=lambda r: float(r["conf"]))
    kept = [r for r in rows if _near(anchor, r) or float(r["conf"]) >= floor]
    return kept if kept else [anchor]


def _solo_emit(rows: list[dict]) -> dict | None:
    """Emit highest-conf row if it clears EMIT_CONF (never average)."""
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["conf"]))
    if float(best["conf"]) < EMIT_CONF:
        return None
    return {
        "xy": best["xy"],
        "conf": float(best["conf"]),
        "cam": best["cam"],
        "n": 1,
        "agree": False,
    }


def fuse_balls(
    rows: list[dict],
    *,
    soft_dual_fallback: bool = True,
    solo_max_conf: bool = True,
    ghost_prune: bool = True,
    ghost_conf: float = GHOST_CONF,
) -> dict | None:
    """Pitch-space fuse. EMIT_CONF / AGREE_M hard gates.

    F1 soft_dual_fallback: agree cluster with combined conf < emit falls
    through to solo instead of silent drop.
    F2 solo_max_conf: solo uses max conf among candidates, not weight seed only.
    F3 ghost_prune: drop weak maps that disagree with the max-conf anchor.
    """
    valid = [r for r in rows if r]
    if not valid:
        return None
    valid = prune_ghost_maps(valid, enabled=ghost_prune, ghost_conf=ghost_conf)
    valid.sort(key=lambda r: r["weight"], reverse=True)
    seed = valid[0]
    cluster = [r for r in valid if _near(seed, r)]
    if len(cluster) >= 2:
        conf = combined_conf([r["conf"] for r in cluster])
        if conf >= EMIT_CONF:
            win = max(cluster, key=lambda r: r["support"])
            return {
                "xy": _median_xy(cluster),
                "conf": conf,
                "cam": win["cam"],
                "n": len(cluster),
                "agree": True,
            }
        if not soft_dual_fallback:
            return None
    pool = valid if solo_max_conf else [seed]
    return _solo_emit(pool)


# Detect-tick hold (F0): re-emit last good fuse across silent ticks. Not Phase-2 fusion.
HOLD_MAX_GAP = 2


def fuse_balls_with_hold(
    prev_emit: dict | None,
    cur_mapped: list[dict],
    frames_since_emit: int,
    *,
    soft_dual_fallback: bool = True,
    solo_max_conf: bool = True,
    ghost_prune: bool = True,
    ghost_conf: float = GHOST_CONF,
    hold_max_gap: int = HOLD_MAX_GAP,
) -> dict | None:
    """Fuse current maps; if silent, hold prev when conf ≥ EMIT_CONF and gap ≤ hold_max_gap."""
    cur = fuse_balls(
        cur_mapped,
        soft_dual_fallback=soft_dual_fallback,
        solo_max_conf=solo_max_conf,
        ghost_prune=ghost_prune,
        ghost_conf=ghost_conf,
    )
    if cur is not None:
        return cur
    if prev_emit is None:
        return None
    if float(prev_emit.get("conf") or 0.0) < EMIT_CONF:
        return None
    if frames_since_emit > hold_max_gap:
        return None
    return {
        "xy": prev_emit["xy"],
        "conf": float(prev_emit["conf"]),
        "cam": prev_emit["cam"],
        "n": int(prev_emit.get("n") or 1),
        "agree": False,
        "hold": True,
    }
