#!/usr/bin/env python3
"""
Visual verification of landmark detection: sample frames, same fisheye pipeline as test_fisheye,
draw landmarks, write test_landmarks.html with Defished | With landmarks | Count by type.
Run from project root.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/landmark_test/test_landmarks.html"

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


def get_landmark_color(landmark_type: str):
    """Same as validate_people_and_landmarks: BGR colors by type."""
    colors = {
        "goal": (0, 255, 255),
        "center_circle": (255, 0, 255),
        "corner": (255, 255, 0),
        "penalty_box": (0, 165, 255),
        "center_line": (255, 255, 255),
        "touchline": (0, 255, 0),
        "penalty_spot": (255, 0, 0),
        "goal_area": (128, 0, 128),
    }
    return colors.get(landmark_type, (128, 128, 128))


def draw_landmarks(frame, keypoints):
    """Draw landmarks on frame: circles and short labels by type."""
    out = frame.copy()
    for kp in keypoints:
        x, y = int(kp.image_point[0]), int(kp.image_point[1])
        color = get_landmark_color(kp.landmark_type)
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.circle(out, (x, y), 12, (255, 255, 255), 2)
        label = kp.landmark_type.replace("_", " ").title()
        cv2.putText(out, label, (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def save_frame_image(frame_jpg_bytes, output_dir, frame_id, suffix):
    """Save frame as JPEG; return relative path for HTML."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    filename = f"frame_{frame_id}_{suffix}.jpg"
    filepath = frames_dir / filename
    with open(filepath, "wb") as f:
        f.write(frame_jpg_bytes)
    return f"frames/{filename}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate test_landmarks.html: defished frames with overlaid landmarks."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Video path",
    )
    parser.add_argument("--k", type=float, default=-0.32, help="Fisheye k")
    parser.add_argument("--alpha", type=float, default=0.0, help="Fisheye alpha")
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=5,
        help="Number of sample frames",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Max landmarks per frame",
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not get frame count")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "data/output/landmark_test/test_landmarks.html"
    out_path = out_path.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, (total_frames - 1) // max(1, args.num_frames - 1))
    frame_indices = [min(i * step, total_frames - 1) for i in range(args.num_frames)]

    results = []
    for frame_num in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        defished = defish_frame(frame, args.k, alpha=args.alpha)
        defished = crop_to_square(defished)

        result = detect_pitch_keypoints_auto(
            defished,
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            min_points=0,
            max_points=args.max_points,
        )

        if result is not None and result.get("keypoints"):
            raw_keypoints = result["keypoints"]
            keypoints = apply_center_line_and_filter(defished, raw_keypoints, PITCH_LENGTH, PITCH_WIDTH)
            frame_landmarks = draw_landmarks(defished, keypoints)
            by_type = {}
            for kp in keypoints:
                t = kp.landmark_type
                by_type[t] = by_type.get(t, 0) + 1
            count_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
        else:
            frame_landmarks = defished.copy()
            keypoints = []
            count_str = "No landmarks"

        _, buf_orig = cv2.imencode(".jpg", defished, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_lm = cv2.imencode(".jpg", frame_landmarks, [cv2.IMWRITE_JPEG_QUALITY, 85])

        orig_path = save_frame_image(buf_orig.tobytes(), out_dir, frame_num, "orig")
        lm_path = save_frame_image(buf_lm.tobytes(), out_dir, frame_num, "landmarks")

        num_lm = len(keypoints)
        results.append({
            "frame_num": frame_num,
            "orig_path": orig_path,
            "landmarks_path": lm_path,
            "count_str": count_str,
            "num_landmarks": num_lm,
        })

    cap.release()

    if not results:
        print("Error: No frames read")
        sys.exit(1)

    html_rows = "".join(
        f"""
        <tr>
            <td><strong>Frame {r['frame_num']}</strong></td>
            <td class="col-orig"><img src="{r['orig_path']}" alt="Defished" loading="lazy" /></td>
            <td class="col-landmarks"><img src="{r['landmarks_path']}" alt="With landmarks" loading="lazy" /></td>
            <td class="col-count">{r['count_str']} ({r['num_landmarks']} total)</td>
        </tr>"""
        for r in results
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Landmark detection verification</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .info {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #4CAF50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border: 1px solid #ddd; vertical-align: top; }}
        td img {{ display: block; max-width: 350px; }}
        .col-count {{ font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Landmark Detection Verification</h1>
        <p class="info"><strong>Pipeline:</strong> same as fix_fisheye (resize 0.5, defish, crop_black_borders, crop_to_square). k={args.k}, alpha={args.alpha}. max_points={args.max_points}. <strong>Center line & circle:</strong> center circle found via HoughCircles (circle closest to image center); center line uses circle center x (or else dominant vertical line). Touchlines filtered (min distance from center, exclude bottom 15%%, cap 4 per side).</p>
        <p class="info"><strong>Video:</strong> {video_path.name}</p>
        <table>
            <thead>
                <tr>
                    <th>Frame #</th>
                    <th>Defished (cropped to square)</th>
                    <th>With landmarks</th>
                    <th>Count by type</th>
                </tr>
            </thead>
            <tbody>
            {html_rows}
            </tbody>
        </table>
        <p class="info">Landmarks are detected on the defished frame. Use <code>find_landmarks.py</code> for batch JSON output.</p>
    </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Created: {out_path}")
    print(f"Open in browser: http://localhost:8080/data/output/landmark_test/test_landmarks.html")


if __name__ == "__main__":
    main()
