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
ALPHA = 0.5  # 0=crop to valid (no black), 1=full frame (black edges), 0.5=compromise (shows more sides)
NET_MASK_HEIGHT = 0.15  # Mask out the bottom 15% of the screen (the net)


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    """Crop to the largest axis-aligned rectangle with (almost) no black pixels.
    Rows/cols need >= non_black_min proportion non-black. x1/x2 from band y1:y2."""
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
    """
    Blindly attempts to undistort the fisheye effect.
    alpha=0: crop to valid region (no black voids). alpha=1: full frame (black edges).
    Uses content-based crop so output has no black borders.
    """
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def crop_to_square(frame):
    """Center-crop to square: bare minimum crop using shorter dimension as side."""
    h, w = frame.shape[:2]
    s = min(w, h)
    x = (w - s) // 2
    y = (h - s) // 2
    return frame[y : y + s, x : x + s]


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
        straight_frame = defish_frame(frame, k=current_k[0], alpha=ALPHA)
        straight_frame = crop_to_square(straight_frame)
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
