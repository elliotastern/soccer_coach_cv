#!/usr/bin/env python3
"""
Test multiple homography estimation strategies and generate comparison HTML files.
Each strategy is tested on the same frames for fair comparison.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
K_VALUE = -0.32
ALPHA = 0.5
MAP_WIDTH = 600
MAP_HEIGHT = 800


def defish_frame(frame, k, alpha=0.5):
    """Undistort using radial distortion coefficient k."""
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


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

    # Combine
    kernel = np.ones((5, 5), np.uint8)
    field_mask = cv2.dilate(field_mask, kernel, iterations=2)
    field_mask = cv2.erode(field_mask, kernel, iterations=2)
    combined = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)

    # Net Protection (Mask bottom 15%)
    mask_h = int(h * 0.85)
    combined[mask_h:h, :] = 0

    return combined


def frame_to_data_uri(frame_jpg_bytes):
    """Encode JPEG bytes as data URI for embedding in HTML."""
    b64 = base64.b64encode(frame_jpg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def strategy1_auto_single(defished_frame: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 1: Automatic single-frame detection."""
    try:
        from src.analysis.homography import estimate_homography_auto
        
        H = estimate_homography_auto(
            defished_frame,
            pitch_length=105.0,
            pitch_width=68.0,
            correct_distortion=False  # Already defished
        )
        return H
    except Exception as e:
        print(f"  ⚠️  Strategy 1 failed: {e}")
        return None


def strategy2_auto_averaged(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """Strategy 2: Multi-frame averaged detection."""
    try:
        from src.analysis.homography import estimate_homography_auto_averaged
        
        H, _ = estimate_homography_auto_averaged(
            frames,
            pitch_length=105.0,
            pitch_width=68.0,
            correct_distortion=False,  # Already defished
            min_frames=min(5, len(frames))
        )
        return H
    except Exception as e:
        print(f"  ⚠️  Strategy 2 failed: {e}")
        return None


def strategy3_estimated(defished_frame: np.ndarray, w_map: int, h_map: int) -> np.ndarray:
    """Strategy 3: Estimated homography from frame dimensions."""
    h, w = defished_frame.shape[:2]
    
    # Estimate source points based on typical pitch view
    # Assume penalty box or center field is visible
    # Use a reasonable rectangle in the center-upper area
    center_x, center_y = w // 2, h // 2
    
    # Estimate a rectangle that might be a penalty box or field area
    # These are educated guesses based on typical camera angles
    box_width = int(w * 0.4)  # 40% of frame width
    box_height = int(h * 0.3)  # 30% of frame height
    
    src_points = np.float32([
        [center_x - box_width // 2, center_y - box_height // 2],  # Top-left
        [center_x + box_width // 2, center_y - box_height // 2],  # Top-right
        [center_x + box_width // 2, center_y + box_height // 2],  # Bottom-right
        [center_x - box_width // 2, center_y + box_height // 2],  # Bottom-left
    ])
    
    # Destination: full map
    dst_points = np.float32([
        [0, 0],
        [w_map, 0],
        [w_map, h_map],
        [0, h_map],
    ])
    
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H


def strategy4_simple_affine(defished_frame: np.ndarray, w_map: int, h_map: int) -> np.ndarray:
    """Strategy 4: Simple affine transformation (scale + translate)."""
    h, w = defished_frame.shape[:2]
    
    # Use affine transformation (6 parameters instead of 8 for homography)
    # Map center region to full map
    scale_x = w_map / w
    scale_y = h_map / h
    
    # Create affine transformation matrix
    # [a b tx]   [sx  0  cx]
    # [c d ty] = [0  sy  cy]
    # [0 0  1]   [0   0   1]
    M = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Convert to homography format (3x3)
    return M


def generate_strategy_html(
    strategy_name: str,
    strategy_desc: str,
    frames: List[np.ndarray],
    frame_indices: List[int],
    H_matrix: Optional[np.ndarray],
    w_map: int,
    h_map: int,
    output_path: Path
) -> bool:
    """Generate HTML for a single strategy."""
    if H_matrix is None:
        # Generate error HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{strategy_name} - Failed</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        .error {{ background: #d32f2f; color: white; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{strategy_name}</h1>
    <div class="error">
        <strong>❌ Strategy Failed</strong><br>
        {strategy_desc}<br><br>
        Could not estimate homography matrix. This strategy may not work for this video.
    </div>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html_content)
        return False
    
    html_parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{strategy_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #4CAF50; }}
        .info {{ background: #2d2d2d; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .info strong {{ color: #81c784; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #2d2d2d; }}
        th {{ background: #1a1a1a; padding: 10px; text-align: left; color: #4CAF50; }}
        td {{ padding: 10px; border-top: 1px solid #444; }}
        img {{ max-width: 100%; height: auto; border: 2px solid #444; border-radius: 5px; }}
        .col-orig {{ width: 30%; }}
        .col-map {{ width: 30%; }}
        .col-lines {{ width: 30%; }}
    </style>
</head>
<body>
    <h1>{strategy_name}</h1>
    <div class="info">
        <strong>Method:</strong> {strategy_desc}<br>
        <strong>Map size:</strong> {w_map}x{h_map} pixels<br>
        <strong>Frames:</strong> {len(frame_indices)}<br>
        <strong>Status:</strong> ✅ Homography estimated successfully
    </div>
    <table>
        <tr>
            <th>Frame #</th>
            <th>Original (Defished)</th>
            <th>2D Top-Down Map</th>
            <th>2D Map (Lines Only)</th>
        </tr>""",
    ]
    
    for idx, frame in enumerate(frames):
        frame_num = frame_indices[idx]
        
        # Defish (already done, but keep for consistency)
        defished = frame
        
        # Warp to 2D map
        top_down = cv2.warpPerspective(defished, H_matrix, (w_map, h_map))
        
        # Extract lines
        map_lines = get_green_lines(top_down)
        
        # Convert to JPEG
        _, buf_orig = cv2.imencode(".jpg", defished, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_map = cv2.imencode(".jpg", top_down, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_lines = cv2.imencode(".jpg", map_lines, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        uri_orig = frame_to_data_uri(buf_orig.tobytes())
        uri_map = frame_to_data_uri(buf_map.tobytes())
        uri_lines = frame_to_data_uri(buf_lines.tobytes())
        
        html_parts.append(
            f"""
        <tr>
            <td><strong>Frame {frame_num}</strong></td>
            <td class="col-orig"><img src="{uri_orig}" alt="Original frame {frame_num}"></td>
            <td class="col-map"><img src="{uri_map}" alt="2D map frame {frame_num}"></td>
            <td class="col-lines"><img src="{uri_lines}" alt="Lines frame {frame_num}"></td>
        </tr>"""
        )
    
    html_parts.append(
        """
    </table>
    <div class="info">
        <strong>Instructions:</strong><br>
        • Compare this strategy with others using the comparison index<br>
        • The "2D Top-Down Map" should show a bird's-eye view of the pitch<br>
        • The "2D Map (Lines Only)" should show clean white lines without floodlight noise
    </div>
</body>
</html>"""
    )
    
    with open(output_path, "w") as f:
        f.write("".join(html_parts))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple homography strategies and generate comparison HTML"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Video path",
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=5,
        help="Number of sample frames (default 5)",
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
        default=MAP_WIDTH,
        help=f"2D map width (default {MAP_WIDTH})",
    )
    parser.add_argument(
        "--map-height",
        type=int,
        default=MAP_HEIGHT,
        help=f"2D map height (default {MAP_HEIGHT})",
    )
    args = parser.parse_args()
    
    # Load video and sample frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not get frame count")
        sys.exit(1)
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
    
    print("=" * 60)
    print("🎯 MULTI-STRATEGY HOMOGRAPHY TEST")
    print("=" * 60)
    print(f"Video: {Path(args.video).name}")
    print(f"Frames to test: {args.num_frames}")
    print(f"Map size: {args.map_width}x{args.map_height}")
    print()
    
    # Load and defish frames
    print("Loading and defishing frames...")
    frames = []
    defished_frames = []
    
    for frame_num in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        defished = defish_frame(frame, args.k, alpha=args.alpha)
        defished_frames.append(defished)
    
    cap.release()
    
    if not defished_frames:
        print("Error: No frames loaded")
        sys.exit(1)
    
    print(f"✅ Loaded {len(defished_frames)} frames")
    print()
    
    # Output directory
    output_dir = PROJECT_ROOT / "data/output/homography_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each strategy
    strategies = []
    
    # Strategy 1: Auto single-frame
    print("Testing Strategy 1: Automatic Single-Frame Detection...")
    H1 = strategy1_auto_single(defished_frames[0])
    success1 = generate_strategy_html(
        "Strategy 1: Auto Single-Frame",
        "Automatic landmark detection from first defished frame using pitch_keypoint_detector",
        defished_frames,
        frame_indices.tolist(),
        H1,
        args.map_width,
        args.map_height,
        output_dir / "test_homography_strategy1_auto.html"
    )
    strategies.append(("Strategy 1: Auto Single-Frame", "test_homography_strategy1_auto.html", success1))
    print(f"  {'✅ Success' if success1 else '❌ Failed'}")
    print()
    
    # Strategy 2: Auto averaged
    print("Testing Strategy 2: Multi-Frame Averaged Detection...")
    H2 = strategy2_auto_averaged(defished_frames[:min(10, len(defished_frames))])
    success2 = generate_strategy_html(
        "Strategy 2: Auto Averaged",
        "Averaged landmark detection across multiple frames for improved stability",
        defished_frames,
        frame_indices.tolist(),
        H2,
        args.map_width,
        args.map_height,
        output_dir / "test_homography_strategy2_averaged.html"
    )
    strategies.append(("Strategy 2: Auto Averaged", "test_homography_strategy2_averaged.html", success2))
    print(f"  {'✅ Success' if success2 else '❌ Failed'}")
    print()
    
    # Strategy 3: Estimated
    print("Testing Strategy 3: Estimated Homography...")
    H3 = strategy3_estimated(defished_frames[0], args.map_width, args.map_height)
    success3 = generate_strategy_html(
        "Strategy 3: Estimated",
        "Estimated homography from frame dimensions and typical pitch view assumptions",
        defished_frames,
        frame_indices.tolist(),
        H3,
        args.map_width,
        args.map_height,
        output_dir / "test_homography_strategy3_estimated.html"
    )
    strategies.append(("Strategy 3: Estimated", "test_homography_strategy3_estimated.html", success3))
    print(f"  {'✅ Success' if success3 else '❌ Failed'}")
    print()
    
    # Strategy 4: Simple affine
    print("Testing Strategy 4: Simple Affine Transformation...")
    H4 = strategy4_simple_affine(defished_frames[0], args.map_width, args.map_height)
    success4 = generate_strategy_html(
        "Strategy 4: Simple Affine",
        "Simple scale and translate transformation (affine, not perspective)",
        defished_frames,
        frame_indices.tolist(),
        H4,
        args.map_width,
        args.map_height,
        output_dir / "test_homography_strategy4_simple.html"
    )
    strategies.append(("Strategy 4: Simple Affine", "test_homography_strategy4_simple.html", success4))
    print(f"  {'✅ Success' if success4 else '❌ Failed'}")
    print()
    
    # Generate comparison index
    print("Generating comparison index...")
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Homography Strategy Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #4CAF50; }}
        .info {{ background: #2d2d2d; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .strategy {{ background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .strategy.failed {{ border-left-color: #d32f2f; }}
        .strategy h3 {{ margin-top: 0; color: #81c784; }}
        .strategy.failed h3 {{ color: #f44336; }}
        a {{ color: #4CAF50; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .status {{ display: inline-block; padding: 5px 10px; border-radius: 3px; margin-left: 10px; }}
        .status.success {{ background: #4CAF50; color: white; }}
        .status.failed {{ background: #d32f2f; color: white; }}
    </style>
</head>
<body>
    <h1>Homography Strategy Comparison</h1>
    <div class="info">
        <strong>Test Configuration:</strong><br>
        Video: {Path(args.video).name}<br>
        Frames tested: {args.num_frames}<br>
        Map size: {args.map_width}x{args.map_height} pixels<br>
        Fisheye: k={args.k}, alpha={args.alpha}
    </div>
"""
    
    for name, filename, success in strategies:
        status_class = "success" if success else "failed"
        status_text = "✅ Success" if success else "❌ Failed"
        strategy_class = "" if success else "failed"
        
        index_html += f"""
    <div class="strategy {strategy_class}">
        <h3>{name} <span class="status {status_class}">{status_text}</span></h3>
        <p><a href="{filename}" target="_blank">View Results →</a></p>
    </div>
"""
    
    index_html += """
    <div class="info">
        <strong>How to Compare:</strong><br>
        1. Click each strategy link to view its results<br>
        2. Check the "2D Top-Down Map" column - it should show a bird's-eye view<br>
        3. Check the "2D Map (Lines Only)" column - it should show clean white lines<br>
        4. The best strategy will show a clear top-down pitch with visible field lines<br>
        5. Once you find the best strategy, you can use its homography matrix for future processing
    </div>
</body>
</html>"""
    
    index_path = output_dir / "test_homography_comparison_index.html"
    with open(index_path, "w") as f:
        f.write(index_html)
    
    print(f"✅ Comparison index: {index_path}")
    print()
    print("=" * 60)
    print("✅ All strategies tested!")
    print("=" * 60)
    print(f"View comparison at: http://localhost:9912/data/output/homography_test/test_homography_comparison_index.html")
    print()
    print("Results:")
    for name, filename, success in strategies:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()
