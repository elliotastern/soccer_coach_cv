#!/usr/bin/env python3
"""Match 3 landmark dashboard: stills + 4-click homography save."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_cam_id import cam_id_from_raw_name, load_match_raw  # noqa: E402
from pitch1 import load_pitch1, pitch1_landmarks  # noqa: E402

DASH = ROOT / "reports/eval_match3/landmark_dashboard"
STILL_DIR = DASH / "stills"
CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
MATCH3_RAW = ROOT / "data/raw/Match 3"
# Camera id = filename P-code. Never remap by FOV (P1-006.mp4 is P1, not P9).
RAW_FILE = load_match_raw(MATCH3_RAW)
for _cam, _path in RAW_FILE.items():
    if cam_id_from_raw_name(_path.name) != _cam:
        raise ValueError(f"{_path.name} is not camera {_cam}")
SRC = ROOT / "reports/eval_match3/pitchmap_gallery/source"
STEM = "rand_t00627.9s"
DETECT_W = 1920
PITCH1 = load_pitch1()
PL = float(PITCH1["length_m"])
PW = float(PITCH1["width_m"])
LANDMARKS = pitch1_landmarks(PITCH1)
DISPLAY = {n: v["label"] for n, v in LANDMARKS.items()}
R = float(PITCH1["marks"]["center_circle_radius_m"])
PEN_D = float(PITCH1["marks"]["penalty_area_depth_m"])
PEN_HW = float(PITCH1["marks"]["penalty_area_half_width_m"])
PSPOT = float(PITCH1["marks"]["penalty_spot_m"])
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


def _xy(name: str) -> tuple[float, float]:
    x, y = LANDMARKS[name]["xy"]
    return (float(x), float(y))


def _pts(*names: str):
    return [(n, _xy(n)) for n in names]


ORDERS = {
    "both_sides_south": _pts(
        "halfway_near_touch", "halfway_far_touch",
        "left_near_corner", "left_far_corner",
    ),
    "both_sides_north": _pts(
        "halfway_near_touch", "halfway_far_touch",
        "right_near_corner", "right_far_corner",
    ),
    "quad_near_left": _pts(
        "halfway_near_touch", "left_near_corner", "center", "circle_near",
    ),
    "quad_near_right": _pts(
        "halfway_near_touch", "right_near_corner", "center", "circle_near",
    ),
    "quad_far_left": _pts(
        "halfway_far_touch", "left_far_corner", "center", "circle_far",
    ),
    "quad_far_right": _pts(
        "halfway_far_touch", "right_far_corner", "center", "circle_far",
    ),
    "goal_left": _pts(
        "left_box_goal_near", "left_box_goal_far",
        "left_box_18_near", "left_box_18_far",
    ),
    "goal_right": _pts(
        "right_box_goal_near", "right_box_goal_far",
        "right_box_18_near", "right_box_18_far",
    ),
}


def all_landmarks():
    return {n: _xy(n) for n in LANDMARKS}


def on_pitch_names():
    return list(all_landmarks())


def families():
    names = on_pitch_names()
    return {k: names for k in ORDERS}


SWAP_GROUPS = [
    ("Corners", [
        "left_near_corner", "left_far_corner",
        "right_near_corner", "right_far_corner",
    ]),
    ("Halfway / center", [
        "halfway_near_touch", "halfway_far_touch",
        "center", "circle_near", "circle_far",
    ]),
    ("South 18-yard", [
        "left_box_goal_near", "left_box_goal_far",
        "left_box_18_near", "left_box_18_far",
    ]),
    ("South 6-yard / goal", [
        "left_6_goal_near", "left_6_goal_far",
        "left_6_box_near", "left_6_box_far",
        "left_post_near", "left_post_far", "left_penalty_spot",
    ]),
    ("North 18-yard", [
        "right_box_goal_near", "right_box_goal_far",
        "right_box_18_near", "right_box_18_far",
    ]),
    ("North 6-yard / goal", [
        "right_6_goal_near", "right_6_goal_far",
        "right_6_box_near", "right_6_box_far",
        "right_post_near", "right_post_far", "right_penalty_spot",
    ]),
]


def swap_groups():
    return [{"title": title, "names": list(names)} for title, names in SWAP_GROUPS]


def nearby_unused(name: str, used: set[str], n: int = 5) -> list[str]:
    catalog = all_landmarks()
    if name not in catalog:
        raise ValueError(f"unknown landmark {name}")
    x0, y0 = catalog[name]
    ranked = []
    for other, (x, y) in catalog.items():
        if other == name or other in used:
            continue
        ranked.append((math.hypot(x - x0, y - y0), other))
    ranked.sort()
    return [nm for _, nm in ranked[:n]]

CAMS = [
    {"id": "P10", "label": "P10 — South left, both sidelines", "order": "both_sides_south"},
    {"id": "P7", "label": "P7 — South right, both sidelines", "order": "both_sides_south"},
    {"id": "P8", "label": "P8 — North left goal / corner", "order": "goal_right"},
    {"id": "P9", "label": "P9 — North right corner / goal", "order": "both_sides_north"},
    {"id": "P1", "label": "P1 — South, lengthwise toward goal", "order": "both_sides_south"},
    {"id": "P6", "label": "P6 — North", "order": "goal_right"},
    {"id": "P_Goal1", "label": "P_Goal1 — Goal", "order": "goal_right"},
    {"id": "P_Goal2", "label": "P_Goal2 — Goal", "order": "goal_right"},
]

for spec in CAMS:
    if spec["id"] not in RAW_FILE:
        raise FileNotFoundError(f"no video titled {spec['id']} in {MATCH3_RAW}")


def resize_w(frame, width=DETECT_W):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def extract_still(cam: str) -> Path:
    STILL_DIR.mkdir(parents=True, exist_ok=True)
    dest = STILL_DIR / f"{cam}.jpg"
    src = RAW_FILE[cam]
    if cam_id_from_raw_name(src.name) != cam:
        raise ValueError(f"{src.name} is camera {cam_id_from_raw_name(src.name)}, not {cam}")
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
        vid = RAW_FILE[spec["id"]]
        if cam_id_from_raw_name(vid.name) != spec["id"]:
            raise ValueError(f"{vid.name} cannot be camera {spec['id']}")
        cams.append({
            **spec,
            "still": f"stills/{spec['id']}.jpg",
            "video": str(vid.relative_to(ROOT)).replace("\\", "/"),
            "videoName": vid.name,
            "image_wh": [w, h],
            "saved": saved,
        })
    catalog = all_landmarks()
    payload = {
        "title": "landmark_marker",
        "pitch_length_m": PL,
        "pitch_width_m": PW,
        "order_titles": ORDER_TITLES,
        "landmarks": {
            n: {
                "label": LANDMARKS[n]["label"],
                "xy": [xy[0], xy[1]],
                "spec": LANDMARKS[n]["spec"],
            }
            for n, xy in catalog.items()
        },
        "pitch_marks": PITCH1["marks"],
        "pitch1": "docs/product/PITCH1_DIMENSIONS.json",
        "families": families(),
        "swap_groups": swap_groups(),
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


def resolve_landmark_names(order_name: str, landmark_names=None):
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
        return names, [catalog[n] for n in names]
    if order_name not in ORDERS:
        raise ValueError(f"unknown order {order_name}")
    order = ORDERS[order_name]
    names = [n for n, _ in order]
    return names, [xy for _, xy in order]


# P9 still: north-right corner / 18-yard / penalty (all in FOV).
P9_VISIBLE = [
    "right_box_18_far",
    "right_box_goal_far",
    "right_penalty_spot",
    "right_far_corner",
]


def save_clicks(cam: str, order_name: str, image_points: list, landmark_names=None,
                dry_run: bool = False) -> dict:
    if cam not in {c["id"] for c in CAMS}:
        raise ValueError(f"unknown cam {cam}")
    names, pitch_pts = resolve_landmark_names(order_name, landmark_names)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "camera": cam,
            "landmarks": names,
        }
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
