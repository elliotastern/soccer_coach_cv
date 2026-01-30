#!/usr/bin/env python3
"""
Generate test_homography.html structure without cv2 (for environments without OpenCV).
Creates the HTML report structure with placeholder messages.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_HOMOGRAPHY = PROJECT_ROOT / "data/output/homography_calibration.json"


def main():
    parser = argparse.ArgumentParser(
        description="Generate test_homography.html structure (without cv2 processing)"
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

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Homography calibration verification</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #4CAF50; }}
        .info {{ background: #2d2d2d; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .info strong {{ color: #81c784; }}
        .warning {{ background: #ff9800; color: #000; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .warning strong {{ color: #d32f2f; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #2d2d2d; }}
        th {{ background: #1a1a1a; padding: 10px; text-align: left; color: #4CAF50; }}
        td {{ padding: 10px; border-top: 1px solid #444; text-align: center; }}
        .placeholder {{ background: #444; padding: 100px; border: 2px dashed #666; border-radius: 5px; color: #999; }}
        .col-orig {{ width: 30%; }}
        .col-map {{ width: 30%; }}
        .col-lines {{ width: 30%; }}
    </style>
</head>
<body>
    <h1>Homography Calibration Verification</h1>
    <div class="warning">
        <strong>⚠️ Preview Mode (No cv2 Available)</strong><br><br>
        This is a structure preview. To generate the actual test report with frame images:<br>
        <code>python scripts/test_homography.py</code><br><br>
        <strong>Note:</strong> Requires OpenCV (cv2) to process video frames.
    </div>
    <div class="info">
        <strong>Calibration:</strong> k={k_value}, alpha={alpha}<br>
        <strong>Map size:</strong> {w_map}x{h_map} pixels<br>
        <strong>Frames to sample:</strong> {args.num_frames}<br>
        <strong>Video:</strong> {Path(args.video).name}<br>
        <strong>Source points:</strong> {calib.get('source_points', [])}
    </div>
    <table>
        <tr>
            <th>Frame #</th>
            <th>Original (Defished)</th>
            <th>2D Top-Down Map</th>
            <th>2D Map (Lines Only)</th>
        </tr>"""

    # Add placeholder rows for each frame
    for i in range(args.num_frames):
        frame_num = i * 100  # Estimated frame numbers
        html_content += f"""
        <tr>
            <td><strong>Frame {frame_num}</strong></td>
            <td class="col-orig">
                <div class="placeholder">
                    Original frame {frame_num}<br>
                    (Defished with k={k_value})
                </div>
            </td>
            <td class="col-map">
                <div class="placeholder">
                    2D Top-Down Map<br>
                    ({w_map}x{h_map}px)
                </div>
            </td>
            <td class="col-lines">
                <div class="placeholder">
                    Lines Only<br>
                    (Green filtered)
                </div>
            </td>
        </tr>"""

    html_content += """
    </table>
    <div class="info">
        <strong>Instructions:</strong><br>
        • The "2D Top-Down Map" should show a bird's-eye view of the pitch<br>
        • The "2D Map (Lines Only)" should show clean white lines without floodlight noise<br>
        • If the map looks distorted, re-run <code>fix_homography.py</code> and click different points<br>
        • If lines are missing, check the green/white HSV thresholds in the script<br><br>
        <strong>To generate actual images:</strong> Run <code>python scripts/test_homography.py</code> (requires cv2)
    </div>
</body>
</html>"""

    # Write HTML
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "data/output/homography_test/test_homography.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\n✅ Created: {output_path}")
    print(f"   Open in browser: http://localhost:9912/data/output/homography_test/test_homography.html")
    print(f"\n⚠️  Note: This is a preview structure. To generate actual frame images:")
    print(f"   Run: python scripts/test_homography.py (requires OpenCV/cv2)")


if __name__ == "__main__":
    main()
