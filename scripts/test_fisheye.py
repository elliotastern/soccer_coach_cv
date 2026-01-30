#!/usr/bin/env python3
"""
Verify fisheye fix: sample frames from video, apply defish, write test_fisheye.html
with side-by-side original vs fixed for visual verification.
Run from project root.
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
    Treats pixel as black if mean BGR < black_thresh. Rows/cols need >= non_black_min
    proportion non-black. x1/x2 computed from band y1:y2 so curved black is trimmed.
    If inner rect is empty or < 10% of area, returns frame unchanged."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame.astype(np.float64)
    is_black = gray < black_thresh
    # Top: first row where proportion non-black >= non_black_min
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
    # x1/x2 from band y1:y2 only so curved black at edges is trimmed
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
    """Undistort using radial distortion coefficient k. alpha=0: no black edges.
    Uses content-based crop so no frame has black borders."""
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


def save_frame_image(frame_jpg_bytes, output_dir, frame_id, suffix):
    """Save frame as JPEG file and return relative path."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    filename = f"frame_{frame_id}_{suffix}.jpg"
    filepath = frames_dir / filename
    
    with open(filepath, "wb") as f:
        f.write(frame_jpg_bytes)
    
    # Return relative path from HTML file location
    return f"frames/{filename}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate test_fisheye.html to verify fisheye fix."
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
        help="Distortion k used for defish",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="0=no black (cropped), 1=full frame with black (default 0)",
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
        help="Output path for test_fisheye.html (default: data/output/fisheye_test/test_fisheye.html)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not get frame count")
        sys.exit(1)

    step = max(1, (total_frames - 1) // max(1, args.num_frames - 1))
    frame_indices = [min(i * step, total_frames - 1) for i in range(args.num_frames)]

    # Determine output directory early so we can save images there
    out_path = args.output
    if out_path is None:
        out_dir = PROJECT_ROOT / "data" / "output" / "fisheye_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "test_fisheye.html"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_dir = out_path.parent
    
    results = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fixed = defish_frame(frame, args.k, alpha=args.alpha)
        fixed = crop_to_square(fixed)
        _, buf_orig = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_fixed = cv2.imencode(".jpg", fixed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        frame_id = int(idx)
        orig_path = save_frame_image(buf_orig.tobytes(), out_dir, frame_id, "orig")
        fixed_path = save_frame_image(buf_fixed.tobytes(), out_dir, frame_id, "fixed")
        
        results.append(
            {
                "frame_id": frame_id,
                "orig_path": orig_path,
                "fixed_path": fixed_path,
            }
        )

    cap.release()

    if not results:
        print("Error: No frames read")
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_dir = PROJECT_ROOT / "data" / "output" / "fisheye_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "test_fisheye.html"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    html_rows = []
    for r in results:
        html_rows.append(
            f"""
        <tr>
            <td><strong>Frame {r['frame_id']}</strong></td>
            <td><img src="{r['orig_path']}" alt="original" style="max-width:100%; height:auto;" loading="lazy" /></td>
            <td><img src="{r['fixed_path']}" alt="fixed" style="max-width:100%; height:auto;" loading="lazy" /></td>
        </tr>"""
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fisheye fix verification (k={args.k}, alpha={args.alpha})</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
        th {{ background: #4CAF50; color: white; }}
        td img {{ display: block; max-width: 400px; }}
        .k {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fisheye fix verification</h1>
        <p class="k">Distortion k = {args.k}, alpha = {args.alpha} (0=no black) | Fixed: defished + center-cropped to square | Video: {Path(args.video).name}</p>
        <table>
            <thead>
                <tr>
                    <th>Frame</th>
                    <th>Original</th>
                    <th>Fixed (defished, cropped to square)</th>
                </tr>
            </thead>
            <tbody>
            {"".join(html_rows)}
            </tbody>
        </table>
        <p>Check touchlines: they should look straight in the &quot;Fixed&quot; column. If not, re-run <code>fix_fisheye.py</code> and tune k, then run this test again with <code>--k &lt;value&gt;</code>.</p>
    </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Open in browser to verify fisheye fix (k={args.k}).")


if __name__ == "__main__":
    main()
