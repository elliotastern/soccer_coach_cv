#!/usr/bin/env python3
"""
Batch landmark detection on video using the same fisheye pipeline as test_fisheye.
Output: JSON with per-frame image_points, pitch_points, landmark_types for downstream (e.g. homography).
Run from project root.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/landmark_detection.json"

sys.path.insert(0, str(PROJECT_ROOT))


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    """Same as test_fisheye: crop to rectangle with (almost) no black. x1/x2 from band y1:y2."""
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


def defish_frame(frame, k, alpha=0.0):
    """Same as test_fisheye: remap then crop_black_borders."""
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def crop_to_square(frame):
    """Same as test_fisheye: center-crop to square."""
    h, w = frame.shape[:2]
    s = min(w, h)
    x = (w - s) // 2
    y = (h - s) // 2
    return frame[y : y + s, x : x + s]


def main():
    parser = argparse.ArgumentParser(
        description="Batch landmark detection on video (same fisheye pipeline as test_fisheye). Output JSON."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Video path",
    )
    parser.add_argument("--k", type=float, default=-0.32, help="Fisheye k (default -0.32)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Fisheye alpha (default 0)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = all)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth frame (default 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Max landmarks per frame (default 50)",
    )
    args = parser.parse_args()

    from src.analysis.pitch_keypoint_detector import detect_pitch_keypoints_auto
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from landmark_improve import apply_center_line_and_filter, PITCH_LENGTH, PITCH_WIDTH

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_data = []
    frame_id = 0
    processed = 0
    by_type_total = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.every > 1 and frame_id % args.every != 0:
            frame_id += 1
            continue
        if args.max_frames > 0 and processed >= args.max_frames:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = defish_frame(frame, args.k, alpha=args.alpha)
        frame = crop_to_square(frame)

        result = detect_pitch_keypoints_auto(
            frame,
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            min_points=0,
            max_points=args.max_points,
        )

        if result is not None and len(result["keypoints"]) >= 0:
            raw_kps = result["keypoints"]
            kps = apply_center_line_and_filter(frame, raw_kps, PITCH_LENGTH, PITCH_WIDTH)
            image_points = [list(kp.image_point) for kp in kps]
            pitch_points = [list(kp.pitch_point) for kp in kps]
            landmark_types = [kp.landmark_type for kp in kps]
            confidences = [float(kp.confidence) for kp in kps]
            for t in landmark_types:
                by_type_total[t] = by_type_total.get(t, 0) + 1
        else:
            image_points = []
            pitch_points = []
            landmark_types = []
            confidences = []

        frames_data.append({
            "frame_id": frame_id,
            "num_landmarks": len(image_points),
            "image_points": image_points,
            "pitch_points": pitch_points,
            "landmark_types": landmark_types,
            "confidences": confidences,
        })

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} frames ...")

        frame_id += 1

    cap.release()

    meta = {
        "video": str(video_path),
        "k": args.k,
        "alpha": args.alpha,
        "max_points": args.max_points,
        "num_frames_processed": len(frames_data),
        "every": args.every,
    }

    output = {"meta": meta, "frames": frames_data}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    num_landmarks = [f["num_landmarks"] for f in frames_data]
    mean_lm = float(np.mean(num_landmarks)) if num_landmarks else 0
    median_lm = float(np.median(num_landmarks)) if num_landmarks else 0

    print(f"Done. Processed {len(frames_data)} frames.")
    print(f"  Landmarks per frame: mean={mean_lm:.1f}, median={median_lm:.1f}")
    if by_type_total:
        print("  By type:", ", ".join(f"{k}={v}" for k, v in sorted(by_type_total.items())))
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
