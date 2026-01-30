#!/usr/bin/env python3
"""
Non-interactive homography calibration: use the 4 corners of the defished frame
as source points and map them to the 2D map rectangle. Run this to get a
non-identity calibration so test_homography.html shows a real warp.
For a true bird's-eye from the pitch, run fix_homography.py and click 4 points.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
CALIB_PATH = PROJECT_ROOT / "data/output/homography_calibration.json"
K_VALUE = -0.32
ALPHA = 0.0
W_MAP, H_MAP = 600, 800


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame.astype(np.float64)
    is_black = gray < black_thresh
    y1 = 0
    for y in range(h):
        if (1 - np.mean(is_black[y, :])) >= non_black_min:
            y1 = y
            break
    y2 = h
    for y in range(h - 1, -1, -1):
        if (1 - np.mean(is_black[y, :])) >= non_black_min:
            y2 = y + 1
            break
    band = is_black[y1:y2, :]
    x1 = 0
    for x in range(w):
        if (1 - np.mean(band[:, x])) >= non_black_min:
            x1 = x
            break
    x2 = w
    for x in range(w - 1, -1, -1):
        if (1 - np.mean(band[:, x])) >= non_black_min:
            x2 = x + 1
            break
    if x2 <= x1 or y2 <= y1:
        return frame
    if (x2 - x1) * (y2 - y1) < 0.1 * h * w:
        return frame
    inset = max(1, int(min(w, h) * margin_pct))
    x1 = min(x1 + inset, x2 - 1)
    y1 = min(y1 + inset, y2 - 1)
    x2 = max(x2 - inset, x1 + 1)
    y2 = max(y2 - inset, y1 + 1)
    return frame[y1:y2, x1:x2]


def crop_to_square(frame):
    h, w = frame.shape[:2]
    s = min(w, h)
    x = (w - s) // 2
    y = (h - s) // 2
    return frame[y : y + s, x : x + s]


def defish_frame(frame, k, alpha=0.0):
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def main():
    video_path = DEFAULT_VIDEO
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame = defish_frame(frame, K_VALUE, alpha=ALPHA)
    frame = crop_to_square(frame)
    h, w = frame.shape[:2]

    # Source: 4 corners of defished frame (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    src_points = [[0, 0], [w, 0], [w, h], [0, h]]
    dst_points = np.float32([[0, 0], [W_MAP, 0], [W_MAP, H_MAP], [0, H_MAP]])
    src_points_np = np.float32(src_points)
    H_matrix = cv2.getPerspectiveTransform(src_points_np, dst_points)

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    calib = {
        "homography": H_matrix.tolist(),
        "source_points": src_points,
        "map_size": [W_MAP, H_MAP],
        "k_value": K_VALUE,
        "alpha": ALPHA,
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"Calibration saved: {CALIB_PATH}")
    print(f"Source frame size: {w}x{h} -> Map: {W_MAP}x{H_MAP}")
    print("Run test_homography.py to regenerate the HTML report.")


if __name__ == "__main__":
    main()
