#!/usr/bin/env python3
"""
Interactive fisheye fixer: tune k with '=' / '-' until touchlines look straight.
Run from project root. Press 'q' to quit; note final k for downstream use.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"


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
    Undistorts the image based on the distortion coefficient 'k'.
    Uses content-based crop so no black borders.
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


def main():
    parser = argparse.ArgumentParser(
        description="Interactive fisheye fixer. '=' straighten more, '-' less, 'q' quit."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Video path",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=-0.32,
        help="Initial k (default -0.32)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="0=no black (cropped), 1=full frame with black edges (default 0)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    current_k = args.k
    current_alpha = args.alpha
    print("--- FISHEYE FIXER TOOL ---")
    print("Press '=' to STRAIGHTEN MORE (more negative)")
    print("Press '-' to STRAIGHTEN LESS (closer to 0)")
    print("Press 'a' to INCREASE alpha (show more sides, may have black edges)")
    print("Press 'z' to DECREASE alpha (crop more, no black edges)")
    print("Press 'q' to QUIT")
    print("--------------------------")
    print(f"Current k: {current_k:.3f}, alpha: {current_alpha:.2f}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fixed_frame = defish_frame(frame, current_k, alpha=current_alpha)
        fixed_frame = crop_to_square(fixed_frame)
        h, w = fixed_frame.shape[:2]
        cv2.line(fixed_frame, (0, h // 2), (w, h // 2), (0, 0, 255), 1)
        title = f"Fisheye Fixer | k={current_k:.3f} alpha={current_alpha:.2f} | =/-: k, a/z: alpha, q: quit"
        cv2.imshow(title, fixed_frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("="):
            current_k -= 0.01
            print(f"k: {current_k:.3f}, alpha: {current_alpha:.2f}")
        elif key == ord("-"):
            current_k += 0.01
            print(f"k: {current_k:.3f}, alpha: {current_alpha:.2f}")
        elif key == ord("a"):
            current_alpha = min(1.0, current_alpha + 0.1)
            print(f"k: {current_k:.3f}, alpha: {current_alpha:.2f} (more sides visible)")
        elif key == ord("z"):
            current_alpha = max(0.0, current_alpha - 0.1)
            print(f"k: {current_k:.3f}, alpha: {current_alpha:.2f} (more cropped)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFINAL VALUES SAVED: k = {current_k:.3f}, alpha = {current_alpha:.2f}")


if __name__ == "__main__":
    main()
