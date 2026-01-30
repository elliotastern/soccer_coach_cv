#!/usr/bin/env python3
"""
Interactive homography calibration: click 4 points on the pitch to create a 2D top-down map.
Applies fisheye fix first, then allows point selection for homography calculation.
Run from project root.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"

# Configuration (use same pipeline as fix_fisheye.py)
K_VALUE = -0.32  # Validated fisheye k value
ALPHA = 0.0  # No black edges; crop_black_borders handles trim
NET_MASK_HEIGHT = 0.15  # Bottom 15% masked out (the net)

# Global variables for mouse clicks
src_points = []
calibration_complete = False
current_frame = None


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    """Same as fix_fisheye: crop to rectangle with (almost) no black. x1/x2 from band y1:y2."""
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
    """Center-crop to square (same as fix_fisheye)."""
    h, w = frame.shape[:2]
    s = min(w, h)
    x = (w - s) // 2
    y = (h - s) // 2
    return frame[y : y + s, x : x + s]


def defish_frame(frame, k, alpha=0.0):
    """Same pipeline as fix_fisheye: remap then crop_black_borders (no validPixROI)."""
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def get_green_lines(frame):
    """Extracts clean white lines from the green field only."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green Mask (Turf)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    field_mask = cv2.inRange(hsv, lower_green, upper_green)

    # White Mask (Lines)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine: white lines only on green field
    kernel = np.ones((5, 5), np.uint8)
    field_mask = cv2.dilate(field_mask, kernel, iterations=2)
    field_mask = cv2.erode(field_mask, kernel, iterations=2)
    combined = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)

    # Net Protection (Mask bottom)
    mask_h = int(h * (1 - NET_MASK_HEIGHT))
    combined[mask_h:h, :] = 0

    return combined


def mouse_callback(event, x, y, flags, param):
    """Handles the 4-point calibration clicks."""
    global src_points, calibration_complete

    if event == cv2.EVENT_LBUTTONDOWN and not calibration_complete:
        src_points.append([x, y])
        print(f"Point {len(src_points)} selected: ({x}, {y})")

        if len(src_points) == 4:
            calibration_complete = True
            print("✅ Calibration Complete! Computing Homography...")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive homography calibration: click 4 points to create 2D top-down map"
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
        default=K_VALUE,
        help=f"Fisheye k value (default {K_VALUE})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help=f"Alpha for fisheye (default {ALPHA})",
    )
    parser.add_argument(
        "--map-width",
        type=int,
        default=600,
        help="2D map width in pixels (default 600)",
    )
    parser.add_argument(
        "--map-height",
        type=int,
        default=800,
        help="2D map height in pixels (default 800)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    # Read first frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)

    # 1. Same pipeline as fix_fisheye: resize 0.5 -> defish -> crop_to_square
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    global current_frame
    current_frame = defish_frame(frame, args.k, alpha=args.alpha)
    current_frame = crop_to_square(current_frame)

    # 2. UI Setup
    cv2.namedWindow("Calibration (Click 4 Points)")
    cv2.setMouseCallback("Calibration (Click 4 Points)", mouse_callback)

    print("=" * 60)
    print("🎯 HOMOGRAPHY CALIBRATION")
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("1. Click 4 points in a RECTANGLE pattern")
    print("   Best choice: Penalty box corners")
    print("   Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("2. Press 'q' to quit calibration")
    print("=" * 60)

    # 3. Calibration Loop
    while not calibration_complete:
        display = current_frame.copy()
        # Draw points so far
        for i, pt in enumerate(src_points):
            cv2.circle(display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        # Draw lines connecting points
        if len(src_points) >= 2:
            for i in range(len(src_points) - 1):
                cv2.line(
                    display,
                    tuple(src_points[i]),
                    tuple(src_points[i + 1]),
                    (0, 255, 0),
                    2,
                )
        if len(src_points) == 4:
            # Close the rectangle
            cv2.line(
                display,
                tuple(src_points[3]),
                tuple(src_points[0]),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Calibration (Click 4 Points)", display)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Calibration (Click 4 Points)")

    # 4. Compute Homography Matrix
    w_map, h_map = args.map_width, args.map_height
    dst_points = np.float32(
        [
            [0, 0],  # Top-Left maps to 0,0
            [w_map, 0],  # Top-Right maps to width,0
            [w_map, h_map],  # Bottom-Right
            [0, h_map],  # Bottom-Left
        ]
    )

    src_points_np = np.float32(src_points)
    H_matrix = cv2.getPerspectiveTransform(src_points_np, dst_points)

    print(f"\n✅ Homography matrix computed:")
    print(f"   Map size: {w_map}x{h_map}")
    print(f"   Source points: {src_points}")
    print(f"\nPress 'q' in video window to quit")

    # 5. Process Video Loop (same pipeline as fix_fisheye for each frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # A. Resize, defish, crop to square (same as fix_fisheye output)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        clean_frame = defish_frame(frame, args.k, alpha=args.alpha)
        clean_frame = crop_to_square(clean_frame)

        # B. Warp to 2D Map
        top_down = cv2.warpPerspective(clean_frame, H_matrix, (w_map, h_map))

        # C. Filter Lines (on the map view)
        map_lines = get_green_lines(top_down)

        # Resize for display
        display_orig = cv2.resize(clean_frame, (0, 0), fx=0.5, fy=0.5)
        display_map = cv2.resize(top_down, (0, 0), fx=0.8, fy=0.8)
        display_lines = cv2.resize(map_lines, (0, 0), fx=0.8, fy=0.8)

        cv2.imshow("Original (Defished)", display_orig)
        cv2.imshow("2D Map (Top-Down)", display_map)
        cv2.imshow("2D Map (Lines Only)", display_lines)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save homography matrix
    output_path = PROJECT_ROOT / "data/output/homography_matrix.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), H_matrix)
    print(f"\n💾 Homography matrix saved to: {output_path}")

    # Also save as JSON for easy loading
    import json

    homography_json = {
        "homography": H_matrix.tolist(),
        "source_points": src_points,
        "map_size": [w_map, h_map],
        "k_value": args.k,
        "alpha": args.alpha,
    }
    json_path = PROJECT_ROOT / "data/output/homography_calibration.json"
    with open(json_path, "w") as f:
        json.dump(homography_json, f, indent=2)
    print(f"💾 Homography calibration saved to: {json_path}")


if __name__ == "__main__":
    main()
