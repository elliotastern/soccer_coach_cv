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
MIN_SUPPORT = 0.35
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


def map_ball_box(calib: dict, box, conf: float, frame_wh=None) -> dict | None:
    wh = frame_wh or calib.get("image_wh") or [1920, 1080]
    fx, fy = bbox_foot(box)
    px, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
    xy = apply_H(calib["H"], px, py)
    if xy is None:
        return None
    if not in_pitch_bounds(xy[0], xy[1], margin_m=MARGIN_M):
        return None
    support = hull_support(px, py, calib.get("image_points") or [])
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


def fuse_balls(rows: list[dict]) -> dict | None:
    valid = [r for r in rows if r]
    if not valid:
        return None
    valid.sort(key=lambda r: r["weight"], reverse=True)
    seed = valid[0]
    cluster = [r for r in valid if _near(seed, r)]
    if len(cluster) >= 2:
        conf = combined_conf([r["conf"] for r in cluster])
        if conf < EMIT_CONF:
            return None
        win = max(cluster, key=lambda r: r["support"])
        return {
            "xy": _median_xy(cluster),
            "conf": conf,
            "cam": win["cam"],
            "n": len(cluster),
            "agree": True,
        }
    if seed["conf"] < EMIT_CONF:
        return None
    return {
        "xy": seed["xy"],
        "conf": seed["conf"],
        "cam": seed["cam"],
        "n": 1,
        "agree": False,
    }
