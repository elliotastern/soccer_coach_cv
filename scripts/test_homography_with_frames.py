#!/usr/bin/env python3
"""
Generate test_homography.html using existing extracted frames (no cv2 required).
Shows actual frame images with notes about processing limitations.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_HOMOGRAPHY = PROJECT_ROOT / "data/output/homography_calibration.json"
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data/output/37a_20frames/frames"


def frame_to_data_uri(image_path):
    """Encode image file as data URI for embedding in HTML."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        return None


def resize_image_for_display(image_path, max_size=(800, 600)):
    """Resize image using PIL and return as data URI."""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate test_homography.html using existing frames"
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
        "--frames-dir",
        type=str,
        default=str(DEFAULT_FRAMES_DIR),
        help="Directory containing extracted frames",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for test_homography.html",
    )
    args = parser.parse_args()

    # Load homography calibration
    if not Path(args.homography).exists():
        print(f"❌ Error: Homography file not found: {args.homography}")
        sys.exit(1)

    with open(args.homography, "r") as f:
        calib = json.load(f)

    k_value = calib.get("k_value", -0.32)
    alpha = calib.get("alpha", 0.5)
    map_size = calib.get("map_size", [600, 800])
    w_map, h_map = map_size[0], map_size[1]

    # Find available frames
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"❌ Error: Frames directory not found: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print(f"❌ Error: No frame files found in {frames_dir}")
        sys.exit(1)

    # Select frames to display
    num_available = len(frame_files)
    num_to_show = min(args.num_frames, num_available)
    step = max(1, num_available // num_to_show) if num_available > num_to_show else 1
    selected_frames = frame_files[::step][:num_to_show]

    print(f"✅ Found {num_available} frames, showing {len(selected_frames)}")

    # Generate HTML
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
        .warning { background: #ff9800; color: #000; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .warning strong { color: #d32f2f; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #2d2d2d; }
        th { background: #1a1a1a; padding: 10px; text-align: left; color: #4CAF50; }
        td { padding: 10px; border-top: 1px solid #444; }
        img { max-width: 100%; height: auto; border: 2px solid #444; border-radius: 5px; }
        .col-orig { width: 30%; }
        .col-map { width: 30%; }
        .col-lines { width: 30%; }
        .placeholder { background: #444; padding: 80px 20px; border: 2px dashed #666; border-radius: 5px; color: #999; text-align: center; }
    </style>
</head>
<body>
    <h1>Homography Calibration Verification</h1>
    <div class="warning">
        <strong>⚠️ Limited Processing Mode</strong><br><br>
        Showing original extracted frames. Full processing (defish, homography warp, green filtering) requires OpenCV (cv2).<br>
        To see processed images, run: <code>python scripts/test_homography.py</code> (requires cv2)
    </div>
    <div class="info">
        <strong>Calibration:</strong> k=""",
        str(k_value),
        """, alpha=""",
        str(alpha),
        """<br>
        <strong>Map size:</strong> """,
        f"{w_map}x{h_map}",
        """ pixels<br>
        <strong>Frames shown:</strong> """,
        str(len(selected_frames)),
        """ of """,
        str(num_available),
        """ available<br>
        <strong>Source points:</strong> """,
        str(calib.get("source_points", [])),
        """
    </div>
    <table>
        <tr>
            <th>Frame #</th>
            <th>Original Frame</th>
            <th>2D Top-Down Map<br><small>(requires cv2)</small></th>
            <th>2D Map (Lines Only)<br><small>(requires cv2)</small></th>
        </tr>""",
    ]

    # Add frame rows
    for frame_file in selected_frames:
        frame_num = int(frame_file.stem.split("_")[1])
        uri = resize_image_for_display(frame_file)
        
        if uri:
            html_parts.append(
                f"""
        <tr>
            <td><strong>Frame {frame_num}</strong></td>
            <td class="col-orig"><img src="{uri}" alt="Frame {frame_num}"></td>
            <td class="col-map">
                <div class="placeholder">
                    2D Top-Down Map<br>
                    ({w_map}x{h_map}px)<br>
                    <small>Run test_homography.py<br>with cv2 to generate</small>
                </div>
            </td>
            <td class="col-lines">
                <div class="placeholder">
                    Lines Only<br>
                    (Green filtered)<br>
                    <small>Run test_homography.py<br>with cv2 to generate</small>
                </div>
            </td>
        </tr>"""
            )

    html_parts.append(
        """
    </table>
    <div class="info">
        <strong>Instructions:</strong><br>
        • The "Original Frame" column shows actual extracted frames<br>
        • The "2D Top-Down Map" and "Lines Only" columns require cv2 processing<br>
        • To see full processing: Install OpenCV and run <code>python scripts/test_homography.py</code><br>
        • If the map looks distorted, re-run <code>fix_homography.py</code> and click different points
    </div>
</body>
</html>"""
    )

    # Write HTML
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "data/output/homography_test/test_homography.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    print(f"\n✅ Created: {output_path}")
    print(f"   Open in browser: http://localhost:9912/data/output/homography_test/test_homography.html")


if __name__ == "__main__":
    main()
