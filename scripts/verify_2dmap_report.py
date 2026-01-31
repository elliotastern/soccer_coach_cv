#!/usr/bin/env python3
"""Verify 2D map report output: marks file, map dimensions, and output files exist."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKS_PATH = PROJECT_ROOT / "data/output/2dmap_manual_mark/manual_marks.json"
OUT_DIR = PROJECT_ROOT / "data/output/2dmap_manual_mark"
CALIB_PATH = PROJECT_ROOT / "data/output/homography_calibration.json"


def main():
    # 1. Load manual_marks.json
    if not MARKS_PATH.exists():
        print(f"Missing: {MARKS_PATH}")
        sys.exit(1)
    with open(MARKS_PATH) as f:
        data = json.load(f)
    points = data.get("points", [])
    src_corners = data.get("src_corners_xy", [])
    if len(points) < 5:
        print("manual_marks.json: points must have corner1..4 and center")
        sys.exit(1)
    if len(src_corners) != 4:
        print("manual_marks.json: src_corners_xy must have 4 corners")
        sys.exit(1)
    print("manual_marks.json: points and src_corners_xy shape OK")

    # 2. Expected map size: calibration overrides default 1050x680
    w_map, h_map = 1050, 680
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        if "map_size" in calib:
            w_map, h_map = calib["map_size"][0], calib["map_size"][1]

    # 3. Map image size must match
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
    if w_img != w_map or h_img != h_map:
        print(f"Map size must be {w_map}x{h_map}, got {w_img}x{h_img}")
        sys.exit(1)
    print(f"frame_0_map.jpg size: {w_img}x{h_img} (matches calib/default) OK")

    # 4. HTML and frame files exist
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
