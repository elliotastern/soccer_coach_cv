#!/usr/bin/env python3
"""
Pre-processing pipeline for broadcast soccer footage:
- Blind defishing (radial distortion, tunable k)
- Green-first line extraction (white lines only on turf)
- Net mask (ignore bottom % of frame)

Run from project root. Use '=' / '-' to tune k live; 'q' to quit.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root for default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- CONFIGURATION ---
DISTORTION_K = -0.32  # TUNE THIS: -0.2 (slight curve) to -0.5 (heavy curve)
NET_MASK_HEIGHT = 0.15  # Mask out the bottom 15% of the screen (the net)


def defish_frame(frame, k, alpha=0.0):
    """
    Blindly attempts to undistort the fisheye effect.
    alpha=0: crop to valid region (no black voids). alpha=1: full frame (black edges).
    """
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def isolate_lines(frame):
    """
    1. Creates a mask of the 'Green' field.
    2. Within that green mask, finds 'White' pixels.
    3. Ignores the bottom section (net).
    """
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
    combined_mask = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)
    mask_h = int(h * (1 - NET_MASK_HEIGHT))
    combined_mask[mask_h:h, :] = 0
    return combined_mask


def main():
    parser = argparse.ArgumentParser(
        description="Defish + green-first line extraction. '=' / '-' tune k, 'q' quit."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"),
        help="Video path (default: data/raw/37CAE053-...)",
    )
    parser.add_argument("--k", type=float, default=DISTORTION_K, help="Initial distortion k")
    parser.add_argument(
        "--net-mask",
        type=float,
        default=NET_MASK_HEIGHT,
        help="Fraction of frame height to mask at bottom (default 0.15)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    global NET_MASK_HEIGHT
    NET_MASK_HEIGHT = args.net_mask
    current_k = [args.k]

    print(
        "Controls:\n  'q': Quit\n  '=': Increase distortion fix (straighten more)\n  '-': Decrease distortion fix (straighten less)"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        straight_frame = defish_frame(frame, k=current_k[0])
        line_view = isolate_lines(straight_frame)
        cv2.imshow("1. Defished Reality", straight_frame)
        cv2.imshow("2. Extracted Lines (No Lights/Net)", line_view)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("="):
            current_k[0] -= 0.01
            print(f"k updated: {current_k[0]:.2f}")
        elif key == ord("-"):
            current_k[0] += 0.01
            print(f"k updated: {current_k[0]:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
