#!/usr/bin/env python3
"""Verify 2D map report output: marks file, map dimensions, output files, and (Option B) right goal visible."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKS_PATH = PROJECT_ROOT / "data/output/2dmap_manual_mark/manual_marks.json"
OUT_DIR = PROJECT_ROOT / "data/output/2dmap_manual_mark"
CALIB_PATH = PROJECT_ROOT / "data/output/homography_calibration.json"
DIAGRAM_MARGIN = 25

# Expected diagram sizes (map + 2*margin): half_only 525x680 -> 575x730; left_half/full 1050x680 -> 1100x730
EXPECTED_HALF_ONLY_SIZE = (525 + 2 * DIAGRAM_MARGIN, 680 + 2 * DIAGRAM_MARGIN)
EXPECTED_FULL_SIZE = (1050 + 2 * DIAGRAM_MARGIN, 680 + 2 * DIAGRAM_MARGIN)


def check_right_goal_visible(img):
    """For full-pitch diagram (Option B), right margin should contain goal (light net), not just green."""
    h, w = img.shape[:2]
    if w < EXPECTED_FULL_SIZE[0] - 20:
        return True
    margin_width = 80
    right_roi = img[:, -margin_width:]
    mean_brightness = float(right_roi.mean())
    if mean_brightness < 100:
        return False
    return True


def check_2d_map_accuracy(img, half_pitch_style):
    """Check that the map looks like an accurate 2D pitch: green pitch, landmarks/players present and in expected region."""
    h, w = img.shape[:2]
    margin = DIAGRAM_MARGIN
    # Pitch area (green): expect mean green channel dominant in center
    pitch_left = margin
    pitch_top = margin
    pitch_w = w - 2 * margin
    pitch_h = h - 2 * margin
    if pitch_w <= 0 or pitch_h <= 0:
        return False
    center_roi = img[pitch_top : pitch_top + pitch_h, pitch_left : pitch_left + pitch_w]
    mean_g = float(center_roi[:, :, 1].mean())
    mean_b = float(center_roi[:, :, 0].mean())
    mean_r = float(center_roi[:, :, 2].mean())
    if mean_g < 30 or mean_g < mean_b or mean_g < mean_r:
        print("Accuracy: pitch area does not look green (expected grass color).")
        return False
    # Player/landmark dots: cyan (255,255,0 BGR), white, yellow - high blue and green
    # Count bright non-green-only pixels (dots) in the content area
    if half_pitch_style == "left_half":
        content_width = 525
    else:
        content_width = pitch_w
    content_roi = img[pitch_top : pitch_top + pitch_h, pitch_left : pitch_left + content_width]
    blue = content_roi[:, :, 0]
    green = content_roi[:, :, 1]
    red = content_roi[:, :, 2]
    bright = (blue.astype(float) + green.astype(float) + red.astype(float)) / 3.0
    # Dots are bright and not pure green (so B and R are non-negligible for cyan/white/yellow/red)
    not_dark = bright >= 120
    not_pure_green = (blue.astype(float) + red.astype(float)) >= 80
    dot_like = not_dark & not_pure_green
    dot_count = int(dot_like.sum())
    if dot_count < 20:
        print("Accuracy: too few landmark/player-like pixels (expected dots on pitch).")
        return False
    # Spread: dots should not all be in one corner (players in right place = spread on pitch)
    ys, xs = content_roi.shape[0], content_roi.shape[1]
    yy, xx = np.where(dot_like)
    if len(xx) < 10:
        print("Accuracy: too few dot pixels to check spread.")
        return False
    x_span = float(xx.max() - xx.min()) if xx.max() > xx.min() else 0
    y_span = float(yy.max() - yy.min()) if yy.max() > yy.min() else 0
    min_span_ratio = 0.15
    if x_span < min_span_ratio * xs or y_span < min_span_ratio * ys:
        print("Accuracy: player/landmark dots too clustered (expected spread on pitch).")
        return False
    print("2D map accuracy: pitch green OK, landmarks/players present, spread OK")
    return True


def check_players_match_picture(map_img, frame_path, half_pitch_style, content_width, margin):
    """Check that player positions on the map are consistent with the frame picture (same side / similar distribution)."""
    if not frame_path.exists():
        return True
    try:
        import cv2
    except ImportError:
        return True
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return True
    h_f, w_f = frame.shape[:2]
    # In the frame, player bboxes are green (0, 255, 0) or similar; find green-ish pixels
    g = frame[:, :, 1].astype(float)
    b = frame[:, :, 0].astype(float)
    r = frame[:, :, 2].astype(float)
    green_bbox = (g > 200) & (g > b + 50) & (g > r + 50)
    if green_bbox.sum() < 100:
        return True
    fy, fx = np.where(green_bbox)
    mean_frame_x = float(fx.mean()) / max(1, w_f)
    # Map dots mean x within content area (already computed in check_2d_map_accuracy; we need to pass or recompute)
    pitch_top = margin
    pitch_h = map_img.shape[0] - 2 * margin
    content_roi = map_img[pitch_top : pitch_top + pitch_h, margin : margin + content_width]
    blue = content_roi[:, :, 0].astype(float)
    green = content_roi[:, :, 1].astype(float)
    red = content_roi[:, :, 2].astype(float)
    bright = (blue + green + red) / 3.0
    dot_like = (bright >= 120) & (blue + red >= 80)
    if dot_like.sum() < 20:
        return True
    my, mx = np.where(dot_like)
    mean_map_x = float(mx.mean()) / max(1, content_width)
    # Both normalized 0-1. For correct mapping, frame right (high x) should be map right (high x).
    # Allow some tolerance: if frame mean is on one side of 0.5, map mean should be on same side (or within 0.4)
    same_side = (mean_frame_x > 0.5) == (mean_map_x > 0.5)
    if not same_side and abs(mean_frame_x - mean_map_x) > 0.4:
        print("Accuracy: player positions on map do not match picture (wrong side of pitch).")
        return False
    print("2D map accuracy: player positions consistent with picture OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify 2D map report output.")
    parser.add_argument("--half-pitch-style", choices=["half_only", "left_half"], default=None,
                        help="Expected style; if set, assert diagram size and (for left_half) right goal.")
    args = parser.parse_args()

    # 1. Load manual_marks.json
    if not MARKS_PATH.exists():
        print(f"Missing: {MARKS_PATH}")
        sys.exit(1)
    with open(MARKS_PATH) as f:
        data = json.load(f)
    points = data.get("points", [])
    src_corners = data.get("src_corners_xy", [])
    if len(points) < 2:
        print("manual_marks.json: need at least 2 points (e.g. corners)")
        sys.exit(1)
    if src_corners and len(src_corners) != 4:
        print("manual_marks.json: src_corners_xy must have 4 corners if present")
        sys.exit(1)
    print("manual_marks.json: points (and src_corners_xy) OK")

    # 2. Map image size: must match one of the expected sizes (with margin)
    try:
        import cv2
    except ImportError:
        print("cv2 required for image size check; pip install opencv-python-headless")
        sys.exit(1)
    map_path = OUT_DIR / "frames" / "frame_0_map.jpg"
    if not map_path.exists():
        print(f"Missing: {map_path}")
        sys.exit(1)
    img = cv2.imread(str(map_path))
    if img is None:
        print(f"Cannot read: {map_path}")
        sys.exit(1)
    h_img, w_img = img.shape[:2]
    valid_sizes = [EXPECTED_HALF_ONLY_SIZE, EXPECTED_FULL_SIZE]
    if args.half_pitch_style == "half_only":
        expected = EXPECTED_HALF_ONLY_SIZE
        if (w_img, h_img) != expected:
            print(f"Map size must be {expected[0]}x{expected[1]} (half_only), got {w_img}x{h_img}")
            sys.exit(1)
    elif args.half_pitch_style == "left_half":
        expected = EXPECTED_FULL_SIZE
        if (w_img, h_img) != expected:
            print(f"Map size must be {expected[0]}x{expected[1]} (left_half), got {w_img}x{h_img}")
            sys.exit(1)
        if not check_right_goal_visible(img):
            print("Right goal not visible: right margin of map should show goal (light net).")
            sys.exit(1)
        print("Right goal visible (right margin) OK")
    else:
        if (w_img, h_img) not in valid_sizes:
            print(f"Map size must be one of {valid_sizes}, got {w_img}x{h_img}")
            sys.exit(1)
        if (w_img, h_img) == EXPECTED_FULL_SIZE:
            if not check_right_goal_visible(img):
                print("Right goal not visible: right margin of map should show goal (light net).")
                sys.exit(1)
    print(f"frame_0_map.jpg size: {w_img}x{h_img} OK")

    # 3. Accuracy: looks like a valid 2D map (green pitch, dots in content area, spread)
    style_for_accuracy = args.half_pitch_style if args.half_pitch_style else ("left_half" if (w_img, h_img) == EXPECTED_FULL_SIZE else "half_only")
    if not check_2d_map_accuracy(img, style_for_accuracy):
        sys.exit(1)
    # 4. Players in right place: map positions consistent with frame picture
    content_width = 525 if style_for_accuracy == "left_half" else (w_img - 2 * DIAGRAM_MARGIN)
    frame0_pic = OUT_DIR / "frames" / "frame_0_picture.jpg"
    if not check_players_match_picture(img, frame0_pic, style_for_accuracy, content_width, DIAGRAM_MARGIN):
        sys.exit(1)

    # HTML and frame files exist
    html_path = OUT_DIR / "test_2dmap_manual_mark.html"
    if not html_path.exists():
        print(f"Missing: {html_path}")
        sys.exit(1)
    frame0_pic = OUT_DIR / "frames" / "frame_0_picture.jpg"
    if not frame0_pic.exists():
        print(f"Missing: {frame0_pic}")
        sys.exit(1)
    print("test_2dmap_manual_mark.html and frame_0_picture.jpg exist OK")
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
