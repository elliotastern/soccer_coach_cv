#!/usr/bin/env python3
"""Match 3 landmark dashboard: stills + 4-click homography save."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "reports/eval_match3/landmark_dashboard"
STILL_DIR = DASH / "stills"
CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
MATCH3_RAW = ROOT / "data/raw/Match 3"
# File prefixes were wrong for P1/P9: P9 is the lengthwise view in P1-006.mp4.
RAW_FILE = {
    "P1": MATCH3_RAW / "P9-004.mp4",
    "P6": MATCH3_RAW / "P6-003.mp4",
    "P7": MATCH3_RAW / "P7-001.mp4",
    "P8": MATCH3_RAW / "p8-005.mp4",
    "P9": MATCH3_RAW / "P1-006.mp4",
    "P10": MATCH3_RAW / "P10-002.mp4",
    "P_Goal1": MATCH3_RAW / "P_Goal1-007.mp4",
    "P_Goal2": MATCH3_RAW / "P_Goal2-008.mp4",
}
SRC = ROOT / "reports/eval_match3/pitchmap_gallery/source"
STEM = "rand_t00627.9s"
DETECT_W = 1920
PL, PW, R = 105.0, 68.0, 9.15
PEN_D, PEN_HW, PSPOT = 16.5, 20.16, 11.0

DISPLAY = {
    "halfway_near_touch": "Halfway Left Sideline",
    "halfway_far_touch": "Halfway Right Sideline",
    "left_near_corner": "South Left Corner",
    "right_near_corner": "North Left Corner",
    "left_far_corner": "South Right Corner",
    "right_far_corner": "North Right Corner",
    "center": "Center Spot",
    "circle_near": "Center Circle Left",
    "circle_far": "Center Circle Right",
    "left_box_goal_near": "South Left Goal-Line Corner",
    "left_box_goal_far": "South Right Goal-Line Corner",
    "left_box_18_near": "South Left 18-Yard Corner",
    "left_box_18_far": "South Right 18-Yard Corner",
    "left_post_near": "South Left Goal Post",
    "left_post_far": "South Right Goal Post",
    "left_penalty_spot": "South Penalty Spot",
    "right_box_goal_near": "North Left Goal-Line Corner",
    "right_box_goal_far": "North Right Goal-Line Corner",
    "right_box_18_near": "North Left 18-Yard Corner",
    "right_box_18_far": "North Right 18-Yard Corner",
    "right_post_near": "North Left Goal Post",
    "right_post_far": "North Right Goal Post",
    "right_penalty_spot": "North Penalty Spot",
}
ORDER_TITLES = {
    "both_sides_south": "South · both sidelines",
    "both_sides_north": "North · both sidelines",
    "quad_near_left": "South Left Quarter",
    "quad_near_right": "North Left Quarter",
    "quad_far_left": "South Right Quarter",
    "quad_far_right": "North Right Quarter",
    "goal_left": "South Goal Box",
    "goal_right": "North Goal Box",
}
ORDERS = {
    "both_sides_south": [
        ("halfway_near_touch", (0.0, PW / 2)),
        ("halfway_far_touch", (0.0, -PW / 2)),
        ("left_near_corner", (-PL / 2, PW / 2)),
        ("left_far_corner", (-PL / 2, -PW / 2)),
    ],
    "both_sides_north": [
        ("halfway_near_touch", (0.0, PW / 2)),
        ("halfway_far_touch", (0.0, -PW / 2)),
        ("right_near_corner", (PL / 2, PW / 2)),
        ("right_far_corner", (PL / 2, -PW / 2)),
    ],
    "quad_near_left": [
        ("halfway_near_touch", (0.0, PW / 2)),
        ("left_near_corner", (-PL / 2, PW / 2)),
        ("center", (0.0, 0.0)),
        ("circle_near", (0.0, R)),
    ],
    "quad_near_right": [
        ("halfway_near_touch", (0.0, PW / 2)),
        ("right_near_corner", (PL / 2, PW / 2)),
        ("center", (0.0, 0.0)),
        ("circle_near", (0.0, R)),
    ],
    "quad_far_left": [
        ("halfway_far_touch", (0.0, -PW / 2)),
        ("left_far_corner", (-PL / 2, -PW / 2)),
        ("center", (0.0, 0.0)),
        ("circle_far", (0.0, -R)),
    ],
    "quad_far_right": [
        ("halfway_far_touch", (0.0, -PW / 2)),
        ("right_far_corner", (PL / 2, -PW / 2)),
        ("center", (0.0, 0.0)),
        ("circle_far", (0.0, -R)),
    ],
    "goal_left": [
        ("left_box_goal_near", (-PL / 2, PEN_HW)),
        ("left_box_goal_far", (-PL / 2, -PEN_HW)),
        ("left_box_18_near", (-PL / 2 + PEN_D, PEN_HW)),
        ("left_box_18_far", (-PL / 2 + PEN_D, -PEN_HW)),
    ],
    "goal_right": [
        ("right_box_goal_near", (PL / 2, PEN_HW)),
        ("right_box_goal_far", (PL / 2, -PEN_HW)),
        ("right_box_18_near", (PL / 2 - PEN_D, PEN_HW)),
        ("right_box_18_far", (PL / 2 - PEN_D, -PEN_HW)),
    ],
}

EXTRA_XY = {
    "left_box_18_far": (-PL / 2 + PEN_D, -PEN_HW),
    "left_post_near": (-PL / 2, 3.66),
    "left_post_far": (-PL / 2, -3.66),
    "left_penalty_spot": (-PL / 2 + PSPOT, 0.0),
    "right_box_18_far": (PL / 2 - PEN_D, -PEN_HW),
    "right_post_near": (PL / 2, 3.66),
    "right_post_far": (PL / 2, -3.66),
    "right_penalty_spot": (PL / 2 - PSPOT, 0.0),
}


def all_landmarks():
    out = {n: (float(x), float(y)) for n, (x, y) in EXTRA_XY.items()}
    for pts in ORDERS.values():
        for n, xy in pts:
            out[n] = (float(xy[0]), float(xy[1]))
    return out


def on_pitch_names():
    return list(all_landmarks())


def families():
    names = on_pitch_names()
    return {k: names for k in ORDERS}

CAMS = [
    {"id": "P10", "label": "P10 — South left, both sidelines", "order": "both_sides_south"},
    {"id": "P7", "label": "P7 — South right, both sidelines", "order": "both_sides_south"},
    {"id": "P8", "label": "P8 — North left, both sidelines", "order": "both_sides_north"},
    {"id": "P9", "label": "P9 — North right, toward goal", "order": "both_sides_north"},
    {"id": "P1", "label": "P1 — Goal close", "order": "goal_right"},
    {"id": "P6", "label": "P6 — North", "order": "goal_right"},
    {"id": "P_Goal1", "label": "P_Goal1 — Goal", "order": "goal_right"},
    {"id": "P_Goal2", "label": "P_Goal2 — Goal", "order": "goal_right"},
]


def resize_w(frame, width=DETECT_W):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def extract_still(cam: str) -> Path:
    STILL_DIR.mkdir(parents=True, exist_ok=True)
    dest = STILL_DIR / f"{cam}.jpg"
    src = RAW_FILE[cam]
    if not src.is_file():
        raise FileNotFoundError(src)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(30.0 * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"read failed {src}")
    fr = resize_w(fr)
    cv2.imwrite(str(dest), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return dest


def write_manifest() -> Path:
    DASH.mkdir(parents=True, exist_ok=True)
    cams = []
    allowed = set(on_pitch_names())
    for spec in CAMS:
        still = STILL_DIR / f"{spec['id']}.jpg"
        img = cv2.imread(str(still))
        h, w = (img.shape[0], img.shape[1]) if img is not None else (1080, 1920)
        calib = CALIB_DIR / f"{spec['id']}_manual.json"
        saved = False
        if calib.is_file():
            rec = json.loads(calib.read_text(encoding="utf-8"))
            names = rec.get("landmark_names") or []
            saved = len(names) == 4 and all(n in allowed for n in names)
        cams.append({**spec, "still": f"stills/{spec['id']}.jpg", "image_wh": [w, h], "saved": saved})
    catalog = all_landmarks()
    payload = {
        "title": "landmark_marker",
        "pitch_length_m": PL,
        "pitch_width_m": PW,
        "order_titles": ORDER_TITLES,
        "landmarks": {
            n: {"label": DISPLAY[n], "xy": [xy[0], xy[1]]}
            for n, xy in catalog.items()
        },
        "families": families(),
        "on_pitch": on_pitch_names(),
        "orders": {
            k: [
                {"name": n, "label": DISPLAY[n], "xy": [x, y]}
                for n, (x, y) in pts
            ]
            for k, pts in ORDERS.items()
        },
        "cams": cams,
    }
    path = DASH / "cams.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def pitch_to_img(H, x, y):
    v = np.linalg.inv(H) @ np.array([x, y, 1.0], dtype=float)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def draw_overlay(img, H, clicks, names):
    vis = img.copy()
    for i, (pt, name) in enumerate(zip(clicks, names)):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(vis, p, 8, (0, 255, 0), -1)
        cv2.putText(vis, f"{i + 1}:{name}", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    corners = [(-PL / 2, -PW / 2), (PL / 2, -PW / 2), (PL / 2, PW / 2), (-PL / 2, PW / 2)]
    pts = []
    for xy in corners:
        p = pitch_to_img(H, *xy)
        if p:
            pts.append([int(p[0]), int(p[1])])
    if len(pts) == 4:
        cv2.polylines(vis, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
    circ = []
    for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        p = pitch_to_img(H, R * np.cos(a), R * np.sin(a))
        if p:
            circ.append([int(p[0]), int(p[1])])
    if len(circ) >= 3:
        cv2.polylines(vis, [np.array(circ, np.int32)], True, (255, 0, 0), 2)
    p0 = pitch_to_img(H, 0.0, -PW / 2)
    p1 = pitch_to_img(H, 0.0, PW / 2)
    if p0 and p1:
        cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)
    return vis


def save_clicks(cam: str, order_name: str, image_points: list, landmark_names=None) -> dict:
    if cam not in {c["id"] for c in CAMS}:
        raise ValueError(f"unknown cam {cam}")
    catalog = all_landmarks()
    allowed = set(on_pitch_names())
    if landmark_names:
        names = [str(n) for n in landmark_names]
        if len(names) != 4:
            raise ValueError("need 4 landmarks")
        if len(set(names)) != 4:
            raise ValueError("need 4 different landmarks")
        missing = [n for n in names if n not in catalog]
        if missing:
            raise ValueError(f"unknown landmarks {missing}")
        off = [n for n in names if n not in allowed]
        if off:
            raise ValueError(f"not on this pitch {off}")
        pitch_pts = [catalog[n] for n in names]
    else:
        if order_name not in ORDERS:
            raise ValueError(f"unknown order {order_name}")
        order = ORDERS[order_name]
        names = [n for n, _ in order]
        pitch_pts = [xy for _, xy in order]
    if len(image_points) != 4:
        raise ValueError("need 4 image points")
    still = STILL_DIR / f"{cam}.jpg"
    img = cv2.imread(str(still))
    if img is None:
        raise FileNotFoundError(still)
    h0, w0 = img.shape[:2]
    src = np.float32(image_points)
    dst = np.float32(pitch_pts)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(img, H, image_points, names)
    ov_path = CALIB_DIR / f"{cam}_manual_overlay.jpg"
    cv2.imwrite(str(ov_path), overlay)
    payload = {
        "camera": cam,
        "version": "manual_4click",
        "pass": True,
        "H": H.tolist(),
        "homography": H.tolist(),
        "image_points": [[float(x), float(y)] for x, y in image_points],
        "pitch_points": [[float(x), float(y)] for x, y in pitch_pts],
        "landmark_names": names,
        "order": order_name,
        "source_image": str(still.relative_to(ROOT)),
        "overlay": str(ov_path.relative_to(ROOT)),
        "pitch_length_m": PL,
        "pitch_width_m": PW,
        "image_wh": [int(w0), int(h0)],
    }
    out = CALIB_DIR / f"{cam}_manual.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest()
    return {
        "ok": True,
        "path": str(out.relative_to(ROOT)),
        "overlay": str(ov_path.relative_to(ROOT)),
        "camera": cam,
    }


def extract_all() -> None:
    for spec in CAMS:
        path = extract_still(spec["id"])
        print(f"still {spec['id']} {path.name}", flush=True)
    write_manifest()
    print(f"wrote {DASH / 'cams.json'}", flush=True)


if __name__ == "__main__":
    extract_all()
    raise SystemExit(0)
