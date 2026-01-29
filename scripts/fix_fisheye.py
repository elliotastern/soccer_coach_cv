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


def defish_frame(frame, k, alpha=0.0):
    """
    Undistorts the image based on the distortion coefficient 'k'.
    """
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


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
    print("--- FISHEYE FIXER TOOL ---")
    print("Press '=' to STRAIGHTEN MORE (more negative)")
    print("Press '-' to STRAIGHTEN LESS (closer to 0)")
    print("Press 'q' to QUIT")
    print("--------------------------")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fixed_frame = defish_frame(frame, current_k, alpha=args.alpha)
        h, w = fixed_frame.shape[:2]
        cv2.line(fixed_frame, (0, h // 2), (w, h // 2), (0, 0, 255), 1)
        cv2.imshow("Fisheye Fixer (Use + and - to tune)", fixed_frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("="):
            current_k -= 0.01
            print(f"Current k: {current_k:.3f}")
        elif key == ord("-"):
            current_k += 0.01
            print(f"Current k: {current_k:.3f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFINAL VALUES SAVED: k = {current_k:.3f}")


if __name__ == "__main__":
    main()
