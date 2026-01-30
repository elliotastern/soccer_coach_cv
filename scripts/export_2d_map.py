#!/usr/bin/env python3
"""
Export the 2D (bird's-eye) map from a video using the current homography calibration.
Pipeline: same as fix_fisheye (resize -> defish -> crop_to_square) then warp to 2D map.
Outputs: a 2D map video and/or a folder of 2D map frames.
Run from project root. Requires homography_calibration.json (run fix_homography.py or calibrate_homography_auto.py first).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "data/output/homography_calibration.json"
NET_MASK_HEIGHT = 0.15


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


def get_green_lines(frame):
    """White lines on green only; net mask at bottom 15%."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    field_mask = cv2.dilate(field_mask, kernel, iterations=2)
    field_mask = cv2.erode(field_mask, kernel, iterations=2)
    combined = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)
    mask_h = int(h * (1 - NET_MASK_HEIGHT))
    combined[mask_h:h, :] = 0
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Export 2D bird's-eye map video/frames from video using homography calibration."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Input video path",
    )
    parser.add_argument(
        "--calibration",
        "-c",
        type=str,
        default=str(DEFAULT_CALIB),
        help="Path to homography_calibration.json",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        type=str,
        default=None,
        help="Output 2D map video path (default: data/output/2d_map.mp4)",
    )
    parser.add_argument(
        "--output-frames",
        type=str,
        default=None,
        help="Output folder for 2D map frames (if set, saves each frame as image)",
    )
    parser.add_argument(
        "--lines-only",
        action="store_true",
        help="Export green-mask lines only (white on black) instead of full map",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth frame (default 1 = all frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = all)",
    )
    args = parser.parse_args()

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"Error: Calibration not found: {calib_path}")
        print("Run fix_homography.py (click 4 points) or calibrate_homography_auto.py first.")
        sys.exit(1)

    with open(calib_path, "r") as f:
        calib = json.load(f)

    H = np.array(calib["homography"], dtype=np.float32)
    k_value = calib.get("k_value", -0.32)
    alpha = calib.get("alpha", 0.0)
    w_map, h_map = calib["map_size"][0], calib["map_size"][1]

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.output_video:
        out_video = Path(args.output_video)
    else:
        out_video = PROJECT_ROOT / "data/output/2d_map.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    out_fps = fps / max(1, args.every)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(out_video), fourcc, out_fps, (w_map, h_map))

    out_frames_dir = Path(args.output_frames) if args.output_frames else None
    if out_frames_dir:
        out_frames_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.every > 1 and n % args.every != 0:
            n += 1
            continue
        if args.max_frames > 0 and written >= args.max_frames:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = defish_frame(frame, k_value, alpha=alpha)
        frame = crop_to_square(frame)
        map_frame = cv2.warpPerspective(frame, H, (w_map, h_map))

        if args.lines_only:
            map_frame = get_green_lines(map_frame)
            map_frame = cv2.cvtColor(map_frame, cv2.COLOR_GRAY2BGR)

        writer.write(map_frame)
        if out_frames_dir is not None:
            path = out_frames_dir / f"frame_{n:06d}.jpg"
            cv2.imwrite(str(path), map_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        written += 1
        n += 1
        if written % 100 == 0:
            print(f"  Written {written} frames ...")

    cap.release()
    writer.release()

    print(f"Done. Processed {written} frames.")
    print(f"  2D map video: {out_video}")
    if out_frames_dir is not None:
        print(f"  2D map frames: {out_frames_dir}")


if __name__ == "__main__":
    main()
