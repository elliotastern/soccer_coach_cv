#!/usr/bin/env python3
"""
Verify homography calibration: sample frames, apply defish + homography, write test_homography.html.
Uses the same pipeline as fix_fisheye (defish + crop_black_borders + crop_to_square).
Run from project root.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not available - cannot run test_homography.py")
    print("   This script requires OpenCV for image processing")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_HOMOGRAPHY = PROJECT_ROOT / "data/output/homography_calibration.json"
NET_MASK_HEIGHT = 0.15


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    """Same as fix_fisheye: crop to rectangle with (almost) no black."""
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
    """Same pipeline as fix_fisheye: remap then crop_black_borders."""
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def get_green_lines(frame):
    """Extracts clean white lines from the green field only (same as fix_homography)."""
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


def save_frame_image(frame_jpg_bytes, output_dir, frame_id, suffix):
    """Save frame as JPEG file and return relative path for HTML."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    filename = f"frame_{frame_id}_{suffix}.jpg"
    filepath = frames_dir / filename
    with open(filepath, "wb") as f:
        f.write(frame_jpg_bytes)
    return f"frames/{filename}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate test_homography.html to verify homography calibration."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Video path",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=str(DEFAULT_HOMOGRAPHY),
        help="Path to homography calibration JSON",
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=5,
        help="Number of sample frames (default 5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for test_homography.html (default: data/output/homography_test/test_homography.html)",
    )
    args = parser.parse_args()

    # Load homography calibration
    if not Path(args.homography).exists():
        print(f"❌ Error: Homography file not found: {args.homography}")
        print(f"   Run fix_homography.py first to calibrate")
        sys.exit(1)

    with open(args.homography, "r") as f:
        calib = json.load(f)

    H_matrix = np.array(calib["homography"], dtype=np.float32)
    k_value = calib.get("k_value", -0.32)
    alpha = calib.get("alpha", 0.5)
    map_size = calib.get("map_size", [600, 800])
    w_map, h_map = map_size[0], map_size[1]

    print(f"✅ Loaded homography calibration:")
    print(f"   k={k_value}, alpha={alpha}")
    print(f"   Map size: {w_map}x{h_map}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not get frame count")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "data/output/homography_test/test_homography.html"
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, (total_frames - 1) // max(1, args.num_frames - 1))
    frame_indices = [min(i * step, total_frames - 1) for i in range(args.num_frames)]

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Homography calibration verification</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #4CAF50; }
        .info { background: #2d2d2d; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .info strong { color: #81c784; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #2d2d2d; }
        th { background: #1a1a1a; padding: 10px; text-align: left; color: #4CAF50; }
        td { padding: 10px; border-top: 1px solid #444; }
        img { max-width: 100%; height: auto; border: 2px solid #444; border-radius: 5px; }
        .col-orig { width: 45%; }
        .col-map { width: 45%; }
        .col-lines { width: 45%; }
    </style>
</head>
<body>
    <h1>Homography Calibration Verification</h1>
    <div class="info">
        <strong>Calibration:</strong> k=""",
        str(k_value),
        """, alpha=""",
        str(alpha),
        """<br>
        <strong>Map size:</strong> """,
        f"{w_map}x{h_map}",
        """ pixels<br>
        <strong>Frames sampled:</strong> """,
        str(args.num_frames),
        """<br>
        <strong>Video:</strong> """,
        Path(args.video).name,
        """
    </div>
    <table>
        <tr>
            <th>Frame #</th>
            <th>Original (Defished)</th>
            <th>2D Top-Down Map</th>
            <th>2D Map (Lines Only)</th>
        </tr>""",
    ]

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # Same pipeline as fix_fisheye: resize 0.5 -> defish -> crop_to_square
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        defished = defish_frame(frame, k_value, alpha=alpha)
        defished = crop_to_square(defished)

        top_down = cv2.warpPerspective(defished, H_matrix, (w_map, h_map))
        map_lines = get_green_lines(top_down)

        _, buf_orig = cv2.imencode(".jpg", defished, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_map = cv2.imencode(".jpg", top_down, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_lines = cv2.imencode(".jpg", map_lines, [cv2.IMWRITE_JPEG_QUALITY, 85])

        orig_path = save_frame_image(buf_orig.tobytes(), out_dir, frame_num, "orig")
        map_path = save_frame_image(buf_map.tobytes(), out_dir, frame_num, "map")
        lines_path = save_frame_image(buf_lines.tobytes(), out_dir, frame_num, "lines")

        html_parts.append(
            f"""
        <tr>
            <td><strong>Frame {frame_num}</strong></td>
            <td class="col-orig"><img src="{orig_path}" alt="Original" loading="lazy" /></td>
            <td class="col-map"><img src="{map_path}" alt="2D map" loading="lazy" /></td>
            <td class="col-lines"><img src="{lines_path}" alt="Lines" loading="lazy" /></td>
        </tr>"""
        )

    html_parts.append(
        """
    </table>
    <div class="info">
        <strong>Instructions:</strong><br>
        • The "2D Top-Down Map" should show a bird's-eye view of the pitch<br>
        • The "2D Map (Lines Only)" should show clean white lines without floodlight noise<br>
        • If the map looks distorted, re-run <code>fix_homography.py</code> and click different points<br>
        • If lines are missing, check the green/white HSV thresholds in the script
    </div>
</body>
</html>"""
    )

    out_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"\n✅ Created: {out_path}")
    print(f"   Open in browser: http://localhost:8080/data/output/homography_test/test_homography.html")

    cap.release()


if __name__ == "__main__":
    main()
