#!/usr/bin/env python3
"""Match 3 fisheye dashboard: tag cams + live undistort preview."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from raw_cam_id import cam_id_from_raw_name, load_match_raw  # noqa: E402

DASH = ROOT / "reports/eval_match3/fisheye_dashboard"
STILL_DIR = DASH / "stills"
TAGS_PATH = DASH / "tags.json"
LANDMARK_STILLS = ROOT / "reports/eval_match3/landmark_dashboard/stills"
MATCH3_RAW = ROOT / "data/raw/Match 3"
CAM_IDS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
DETECT_W = 1280
TAGS = ("none", "mild_barrel", "fisheye", "unknown")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_cam_tag(cam: str) -> dict:
    return {
        "tag": "unknown",
        "use_undistort": False,
        "k1": -0.20,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "alpha": 0.5,
        "notes": "",
        "updated": None,
    }


def load_tags() -> dict:
    if TAGS_PATH.is_file():
        data = json.loads(TAGS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "cameras": {}}
    cams = data.setdefault("cameras", {})
    for cam in CAM_IDS:
        if cam not in cams:
            cams[cam] = default_cam_tag(cam)
    data["version"] = 1
    return data


def save_tags(data: dict) -> Path:
    DASH.mkdir(parents=True, exist_ok=True)
    data["updated"] = utc_now()
    TAGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return TAGS_PATH


def extract_still(cam: str, raw_file: Path) -> Path:
    STILL_DIR.mkdir(parents=True, exist_ok=True)
    dest = STILL_DIR / f"{cam}.jpg"
    src_lm = LANDMARK_STILLS / f"{cam}.jpg"
    if src_lm.is_file():
        shutil.copy2(src_lm, dest)
        return dest
    if cam_id_from_raw_name(raw_file.name) != cam:
        raise ValueError(f"{raw_file.name} is not camera {cam}")
    cap = cv2.VideoCapture(str(raw_file))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {raw_file}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(30.0 * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"read failed {raw_file}")
    h, w = fr.shape[:2]
    if w != DETECT_W:
        fr = cv2.resize(
            fr,
            (DETECT_W, int(round(h * DETECT_W / w))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(dest), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return dest


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


def still_path(cam: str) -> Path:
    return STILL_DIR / f"{cam}.jpg"


def render_preview_jpeg(cam: str, k1, k2, p1, p2, alpha, draw_cross=True) -> bytes:
    path = still_path(cam)
    if not path.is_file():
        raise FileNotFoundError(path)
    fr = cv2.imread(str(path))
    if fr is None:
        raise RuntimeError(f"imread failed {path}")
    out = undistort_bgr(fr, k1, k2, p1, p2, alpha)
    if draw_cross:
        h, w = out.shape[:2]
        cv2.line(out, (0, h // 2), (w, h // 2), (0, 0, 255), 1)
        cv2.line(out, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def write_cams_json(raw: dict) -> Path:
    rows = []
    for cam in CAM_IDS:
        vid = raw[cam]
        rows.append({"id": cam, "file": vid.name, "still": f"stills/{cam}.jpg"})
    path = DASH / "cams.json"
    path.write_text(json.dumps({"cameras": rows}, indent=2), encoding="utf-8")
    return path


def write_index() -> Path:
    src = Path(__file__).with_name("match3_fisheye_dashboard.html")
    path = DASH / "index.html"
    if src.is_file():
        path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise FileNotFoundError(src)
    return path


def build() -> int:
    raw = load_match_raw(MATCH3_RAW)
    for cam in CAM_IDS:
        if cam not in raw:
            raise FileNotFoundError(cam)
        extract_still(cam, raw[cam])
        print(f"still {cam}", flush=True)
    tags = load_tags()
    save_tags(tags)
    write_cams_json(raw)
    write_index()
    print(f"wrote {DASH}")
    print("Open: http://127.0.0.1:8080/match3-fisheye")
    return 0


def apply_tag_updates(payload: dict) -> dict:
    tags = load_tags()
    incoming = payload.get("cameras") or {}
    for cam, rec in incoming.items():
        if cam not in CAM_IDS:
            raise ValueError(f"unknown cam {cam}")
        cur = tags["cameras"].setdefault(cam, default_cam_tag(cam))
        for key in ("tag", "use_undistort", "k1", "k2", "p1", "p2", "alpha", "notes"):
            if key in rec:
                cur[key] = rec[key]
        if cur.get("tag") not in TAGS:
            raise ValueError(f"bad tag {cur.get('tag')}")
        cur["updated"] = utc_now()
    save_tags(tags)
    return tags


if __name__ == "__main__":
    raise SystemExit(build())
