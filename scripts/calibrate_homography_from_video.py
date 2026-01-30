#!/usr/bin/env python3
"""
Calibrate homography using the whole video (static camera).
Samples many frames, runs improved landmark detection on each, fuses keypoints across frames,
then estimates a single homography. Use this for a stable calibration instead of a single frame.
Run from project root. Output: data/output/homography_calibration.json
"""
import argparse
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
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


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


def pitch_to_map_points(pitch_points_meters, w_map, h_map, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """Convert pitch coords in meters to map pixel coords. pitch x in [-52.5,52.5], y in [-34,34]."""
    pts = np.array(pitch_points_meters, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    # map_x along pitch length (105m), map_y along pitch width (68m)
    map_x = (pts[:, 0] + pitch_length / 2) / pitch_length * w_map
    map_y = (pts[:, 1] + pitch_width / 2) / pitch_width * h_map
    return np.column_stack((map_x, map_y)).astype(np.float32)


def fuse_keypoints(all_keypoints, image_grid_step=8):
    """
    Fuse keypoints from many frames: cluster by (landmark_type, quantized image_point), take median.
    all_keypoints: list of (image_point, pitch_point, landmark_type) from all frames.
    """
    from collections import defaultdict
    buckets = defaultdict(list)  # (type, qx, qy) -> [(img_pt, pitch_pt), ...]
    for (img_pt, pitch_pt, _type) in all_keypoints:
        qx = int(round(img_pt[0] / image_grid_step)) * image_grid_step
        qy = int(round(img_pt[1] / image_grid_step)) * image_grid_step
        key = (_type, qx, qy)
        buckets[key].append((list(img_pt), list(pitch_pt)))
    fused_img = []
    fused_pitch = []
    for key, pts in buckets.items():
        if not pts:
            continue
        img_pts = np.array([p[0] for p in pts], dtype=np.float64)
        pitch_pts = np.array([p[1] for p in pts], dtype=np.float64)
        fused_img.append(np.median(img_pts, axis=0))
        fused_pitch.append(np.median(pitch_pts, axis=0))
    if len(fused_img) < 4:
        return None, None
    return np.array(fused_img, dtype=np.float32), np.array(fused_pitch, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate homography from whole video (static camera): fuse landmarks across frames."
    )
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Video path")
    parser.add_argument("--k", type=float, default=K_VALUE, help="Fisheye k")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Fisheye alpha")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=40,
        help="Number of frames to sample across the video (default 40)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Max landmarks per frame (default 50)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(CALIB_PATH),
        help="Output homography JSON path",
    )
    args = parser.parse_args()

    from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto
    from landmark_improve import apply_center_line_and_filter, PITCH_LENGTH, PITCH_WIDTH

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 4:
        print("Error: Video has too few frames")
        cap.release()
        sys.exit(1)

    step = max(1, (total_frames - 1) // max(1, args.sample_frames - 1))
    frame_indices = [min(i * step, total_frames - 1) for i in range(args.sample_frames)]
    frame_indices = sorted(set(frame_indices))

    print(f"Sampling {len(frame_indices)} frames from {total_frames} (static camera)...")
    all_keypoints = []

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        defished = defish_frame(frame, args.k, alpha=args.alpha)
        defished = crop_to_square(defished)
        result = detect_pitch_keypoints_auto(
            defished,
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            min_points=0,
            max_points=args.max_points,
        )
        if result is None or not result.get("keypoints"):
            continue
        raw_kps = result["keypoints"]
        kps = apply_center_line_and_filter(defished, raw_kps, PITCH_LENGTH, PITCH_WIDTH)
        for kp in kps:
            all_keypoints.append((kp.image_point, kp.pitch_point, kp.landmark_type))
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(frame_indices)} frames, {len(all_keypoints)} keypoints so far...")

    cap.release()

    if len(all_keypoints) < 4:
        print("Error: Too few keypoints collected. Need at least 4 after fusion.")
        sys.exit(1)

    image_points, pitch_points = fuse_keypoints(all_keypoints)
    if image_points is None or len(image_points) < 4:
        print("Error: Fused keypoints fewer than 4.")
        sys.exit(1)

    map_points = pitch_to_map_points(pitch_points, W_MAP, H_MAP)
    H, mask = cv2.findHomography(
        image_points,
        map_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
        confidence=0.995,
    )
    if H is None:
        print("Error: Homography estimation failed.")
        sys.exit(1)

    inliers = int(np.sum(mask)) if mask is not None else len(image_points)
    print(f"Homography from {inliers}/{len(image_points)} fused points (RANSAC inliers).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib = {
        "homography": H.tolist(),
        "map_size": [W_MAP, H_MAP],
        "k_value": args.k,
        "alpha": args.alpha,
        "source": "calibrate_homography_from_video",
        "num_frames_used": len(frame_indices),
        "num_fused_points": len(image_points),
    }
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"Calibration saved: {out_path}")
    print("Run test_homography.py to verify, then export_2d_map.py to export the 2D map.")


if __name__ == "__main__":
    main()
