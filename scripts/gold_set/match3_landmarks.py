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
STILL_RAW_DIR = DASH / "stills_raw"
CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
FISH_TAGS = ROOT / "reports/eval_match3/fisheye_dashboard/tags.json"
CALIB_PRESERVE_KEYS = (
    "H_player",
    "player_landmark_names",
    "player_image_points",
    "player_h_note",
    "player_roundtrip_max_m",
    "hull_image_points",
    "hull_note",
)
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
    ("South box", [
        "left_box_goal_near", "left_box_goal_far",
        "left_box_18_near", "left_box_18_mid", "left_box_18_far",
    ]),
    ("South goal", [
        "left_post_near", "left_post_far",
    ]),
    ("North box", [
        "right_box_goal_near", "right_box_goal_far",
        "right_box_18_near", "right_box_18_mid", "right_box_18_far",
    ]),
    ("North goal", [
        "right_post_near", "right_post_far",
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


def load_fisheye_tag(cam: str) -> dict | None:
    if not FISH_TAGS.is_file():
        return None
    data = json.loads(FISH_TAGS.read_text(encoding="utf-8"))
    rec = (data.get("cameras") or {}).get(cam) or {}
    if not rec.get("use_undistort"):
        return None
    return {
        "k1": float(rec.get("k1", -0.2)),
        "k2": float(rec.get("k2", 0.0)),
        "p1": float(rec.get("p1", 0.0)),
        "p2": float(rec.get("p2", 0.0)),
        "alpha": float(rec.get("alpha", 0.5)),
    }


def undistort_bgr(frame, k1, k2, p1, p2, alpha=0.5):
    h, w = frame.shape[:2]
    k_mat = np.array(
        [[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64
    )
    dist = np.array(
        [float(k1), float(k2), float(p1), float(p2), 0.0], dtype=np.float64
    )
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        k_mat, dist, (w, h), float(alpha), (w, h)
    )
    return cv2.undistort(frame, k_mat, dist, None, new_k)


def undistort_fingerprint(tag: dict) -> str:
    return (
        f"k1={tag['k1']:.3f},k2={tag['k2']:.3f},"
        f"p1={tag['p1']:.3f},p2={tag['p2']:.3f},a={tag['alpha']:.3f}"
    )


def extract_still(cam: str) -> Path:
    STILL_DIR.mkdir(parents=True, exist_ok=True)
    STILL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = STILL_DIR / f"{cam}.jpg"
    raw_dest = STILL_RAW_DIR / f"{cam}.jpg"
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
    cv2.imwrite(str(raw_dest), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    tag = load_fisheye_tag(cam)
    if tag is not None:
        fr = undistort_bgr(
            fr, tag["k1"], tag["k2"], tag["p1"], tag["p2"], tag["alpha"]
        )
        print(f"  undistort {cam} {undistort_fingerprint(tag)}", flush=True)
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
        fish = load_fisheye_tag(spec["id"])
        saved = False
        if calib.is_file():
            rec = json.loads(calib.read_text(encoding="utf-8"))
            names = rec.get("landmark_names") or []
            saved = len(names) >= 4 and all(n in allowed for n in names)
            if fish is not None:
                want = undistort_fingerprint(fish)
                if rec.get("undistort_fingerprint") != want:
                    saved = False
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
            "undistort": fish,
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
    H = np.asarray(H, dtype=float)
    if abs(np.linalg.det(H)) < 1e-12:
        return None
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

    def poly(names, color, closed=True):
        pts = []
        for n in names:
            p = pitch_to_img(H, *LANDMARKS[n]["xy"])
            if not p:
                return
            pts.append([int(p[0]), int(p[1])])
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, np.int32)], closed, color, 2)

    poly(
        ["left_far_corner", "left_near_corner", "right_near_corner", "right_far_corner"],
        (0, 255, 255),
    )
    poly(["halfway_far_touch", "halfway_near_touch"], (0, 255, 0), closed=False)
    poly(
        ["left_box_goal_near", "left_box_18_near", "left_box_18_far", "left_box_goal_far"],
        (0, 200, 255),
    )
    poly(
        ["right_box_goal_near", "right_box_18_near", "right_box_18_far", "right_box_goal_far"],
        (0, 200, 255),
    )
    poly(["left_post_near", "left_post_far"], (180, 255, 180), closed=False)
    poly(["right_post_near", "right_post_far"], (180, 255, 180), closed=False)
    circ = []
    for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        p = pitch_to_img(H, R * np.cos(a), R * np.sin(a))
        if p:
            circ.append([int(p[0]), int(p[1])])
    if len(circ) >= 3:
        cv2.polylines(vis, [np.array(circ, np.int32)], True, (255, 0, 0), 2)
    return vis


def _fit_H(src, dst):
    src = np.float32(src)
    dst = np.float32(dst)
    if len(src) >= 4:
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("findHomography failed")
        return H, "manual_clicks"
    if len(src) == 3:
        aff = cv2.getAffineTransform(src, dst)
        H = np.eye(3, dtype=float)
        H[:2] = aff
        return H, "manual_affine_3"
    raise ValueError("need at least 3 image points")


def resolve_landmark_names(order_name: str, landmark_names=None, min_n: int = 4):
    catalog = all_landmarks()
    allowed = set(on_pitch_names())
    if landmark_names:
        names = [str(n) for n in landmark_names]
        if len(names) < min_n:
            raise ValueError(f"need at least {min_n} landmarks")
        if len(set(names)) != len(names):
            raise ValueError("need different landmarks")
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


# P9 still: north-right corner / box / post (all in FOV).
P9_VISIBLE = [
    "right_box_18_far",
    "right_box_goal_far",
    "right_post_far",
    "right_far_corner",
]


def save_clicks(cam: str, order_name: str, image_points: list, landmark_names=None,
                dry_run: bool = False, min_n: int = 4) -> dict:
    if cam not in {c["id"] for c in CAMS}:
        raise ValueError(f"unknown cam {cam}")
    names, pitch_pts = resolve_landmark_names(order_name, landmark_names, min_n=min_n)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "camera": cam,
            "landmarks": names,
        }
    if len(image_points) != len(names):
        raise ValueError("image points must match landmarks")
    if len(image_points) < min_n:
        raise ValueError(f"need at least {min_n} image points")
    still = STILL_DIR / f"{cam}.jpg"
    img = cv2.imread(str(still))
    if img is None:
        raise FileNotFoundError(still)
    h0, w0 = img.shape[:2]
    H, version = _fit_H(image_points, pitch_pts)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(img, H, image_points, names)
    ov_path = CALIB_DIR / f"{cam}_manual_overlay.jpg"
    cv2.imwrite(str(ov_path), overlay)
    fish = load_fisheye_tag(cam)
    payload = {
        "camera": cam,
        "version": version,
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
    if fish is not None:
        payload["undistort"] = fish
        payload["undistort_fingerprint"] = undistort_fingerprint(fish)
    out = CALIB_DIR / f"{cam}_manual.json"
    if out.is_file():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        for key in CALIB_PRESERVE_KEYS:
            if key in prev:
                payload[key] = prev[key]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest()
    return {
        "ok": True,
        "path": str(out.relative_to(ROOT)),
        "overlay": str(ov_path.relative_to(ROOT)),
        "camera": cam,
    }


def refit_saved_calibs() -> None:
    catalog = all_landmarks()
    for spec in CAMS:
        path = CALIB_DIR / f"{spec['id']}_manual.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        names = rec.get("landmark_names") or []
        imgs = rec.get("image_points") or []
        keep_n = [n for n in names if n in catalog]
        keep_p = [p for n, p in zip(names, imgs) if n in catalog]
        dropped = [n for n in names if n not in catalog]
        if len(keep_n) < 3:
            raise ValueError(f"{spec['id']} only {len(keep_n)} measured clicks")
        save_clicks(
            spec["id"], rec.get("order") or spec["order"], keep_p,
            landmark_names=keep_n, min_n=3,
        )
        print(f"refit {spec['id']} n={len(keep_n)} dropped={dropped}", flush=True)
    write_manifest()


def extract_all() -> None:
    for spec in CAMS:
        path = extract_still(spec["id"])
        print(f"still {spec['id']} {path.name}", flush=True)
    write_manifest()
    print(f"wrote {DASH / 'cams.json'}", flush=True)


if __name__ == "__main__":
    extract_all()
    raise SystemExit(0)
