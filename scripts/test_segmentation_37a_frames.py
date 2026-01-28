#!/usr/bin/env python3
"""
Test semantic segmentation on 5 random frames from 37a video.
Creates detailed visualizations for each frame.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector, detect_pitch_keypoints_auto
from src.analysis.pitch_line_segmentation import PitchLineSegmenter


def create_detailed_visualization(image, mask, lines, keypoints, output_path):
    """Create a detailed visualization showing all detection results."""
    h, w = image.shape[:2]
    
    # Create a large canvas for detailed view
    canvas_h = h * 2
    canvas_w = w * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas.fill(40)  # Dark gray background
    
    # Panel 1: Original image (top left)
    canvas[0:h, 0:w] = image
    cv2.putText(canvas, "1. Original Frame", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 2: Segmentation mask (top right)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    canvas[0:h, w:2*w] = mask_colored
    cv2.putText(canvas, "2. Segmentation Mask", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 3: Detected lines (bottom left)
    lines_vis = image.copy()
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    canvas[h:2*h, 0:w] = lines_vis
    num_lines = len(lines) if lines is not None and len(lines) > 0 else 0
    cv2.putText(canvas, f"3. Detected Lines ({num_lines})", (10, h + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 4: Keypoints overlay (bottom right)
    keypoints_vis = image.copy()
    
    # Draw lines first
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(keypoints_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    if keypoints:
        from collections import defaultdict
        type_counts = defaultdict(int)
        
        for kp in keypoints:
            x, y = int(kp.image_point[0]), int(kp.image_point[1])
            landmark_type = kp.landmark_type
            confidence = kp.confidence
            type_counts[landmark_type] += 1
            
            # Color by type
            if landmark_type == "goal":
                color = (0, 0, 255)  # Red
            elif landmark_type == "corner":
                color = (255, 0, 0)  # Blue
            elif landmark_type == "penalty_box":
                color = (255, 255, 0)  # Cyan
            elif landmark_type == "center_line":
                color = (0, 255, 255)  # Yellow
            elif landmark_type == "touchline":
                color = (255, 0, 255)  # Magenta
            elif landmark_type == "center_circle":
                color = (128, 255, 128)  # Light green
            else:
                color = (255, 255, 255)  # White
            
            # Draw keypoint
            cv2.circle(keypoints_vis, (x, y), 8, color, -1)
            cv2.circle(keypoints_vis, (x, y), 12, color, 2)
            
            # Label (only show for important keypoints to avoid clutter)
            if landmark_type in ["goal", "corner", "penalty_box"]:
                label = f"{landmark_type[:6]}:{confidence:.2f}"
                cv2.putText(keypoints_vis, label, (x + 15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add legend
        y_offset = 40
        for kp_type, count in sorted(type_counts.items()):
            if kp_type == "goal":
                color = (0, 0, 255)
            elif kp_type == "corner":
                color = (255, 0, 0)
            elif kp_type == "penalty_box":
                color = (255, 255, 0)
            elif kp_type == "center_line":
                color = (0, 255, 255)
            elif kp_type == "touchline":
                color = (255, 0, 255)
            elif kp_type == "center_circle":
                color = (128, 255, 128)
            else:
                color = (255, 255, 255)
            
            cv2.circle(keypoints_vis, (w - 200, y_offset), 6, color, -1)
            cv2.putText(keypoints_vis, f"{kp_type}: {count}", (w - 180, y_offset + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    canvas[h:2*h, w:2*w] = keypoints_vis
    num_keypoints = len(keypoints) if keypoints else 0
    cv2.putText(canvas, f"4. Lines + Keypoints ({num_keypoints})", 
               (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), canvas)
    return output_path


def test_frame(video_path, frame_number, output_dir):
    """Test segmentation on a single frame."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None
    
    h, w = frame.shape[:2]
    print(f"\n  Frame {frame_number}/{total_frames}: {w}x{h} pixels")
    
    # Test segmentation
    segmenter = PitchLineSegmenter(model_path=None)  # Use color-based fallback
    mask = segmenter.segment_pitch_lines(frame)
    line_pixels = np.sum(mask > 0)
    
    # Test line detection
    detector = PitchKeypointDetector(
        pitch_length=105.0,
        pitch_width=68.0,
        use_semantic_segmentation=False
    )
    detector.enable_zero_shot = False  # Disable for faster testing
    lines = detector._detect_field_lines(frame)
    
    # Test keypoint detection
    keypoints = detector.detect_all_keypoints(frame)
    
    # Create visualization
    output_path = output_dir / f"frame_{frame_number:05d}_segmentation.jpg"
    create_detailed_visualization(frame, mask, lines, keypoints, output_path)
    
    # Return results
    from collections import Counter
    type_counts = Counter(kp.landmark_type for kp in keypoints) if keypoints else {}
    
    return {
        'frame': frame_number,
        'lines': len(lines) if lines is not None else 0,
        'keypoints': len(keypoints),
        'line_pixels': line_pixels,
        'keypoint_types': dict(type_counts),
        'output': str(output_path)
    }


def main():
    video_path = "/workspace/soccer_coach_cv/data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        print("Please check the video path.")
        return
    
    print("=" * 70)
    print("Semantic Segmentation Test on 37a Video - 5 Random Frames")
    print("=" * 70)
    print(f"Video: {Path(video_path).name}")
    
    # Open video to get frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    # Select 5 random frames
    random_frames = sorted(random.sample(range(0, total_frames), min(5, total_frames)))
    print(f"\nSelected frames: {random_frames}")
    
    # Create output directory
    output_dir = Path("output/segmentation_37a_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each frame
    print("\n" + "-" * 70)
    print("Testing Frames")
    print("-" * 70)
    
    results = []
    for i, frame_num in enumerate(random_frames, 1):
        print(f"\n[{i}/5] Processing frame {frame_num}...")
        result = test_frame(video_path, frame_num, output_dir)
        if result:
            results.append(result)
            print(f"  ✓ Lines: {result['lines']}, Keypoints: {result['keypoints']}, "
                  f"Line pixels: {result['line_pixels']:,}")
            if result['keypoint_types']:
                print(f"  Keypoint types: {', '.join(f'{k}({v})' for k, v in result['keypoint_types'].items())}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if results:
        avg_lines = np.mean([r['lines'] for r in results])
        avg_keypoints = np.mean([r['keypoints'] for r in results])
        avg_line_pixels = np.mean([r['line_pixels'] for r in results])
        
        print(f"\nAverage across {len(results)} frames:")
        print(f"  Lines detected: {avg_lines:.1f}")
        print(f"  Keypoints detected: {avg_keypoints:.1f}")
        print(f"  Line pixels: {avg_line_pixels:,.0f}")
        
        print(f"\nPer-frame results:")
        for r in results:
            print(f"  Frame {r['frame']:5d}: {r['lines']:3d} lines, {r['keypoints']:3d} keypoints, "
                  f"{r['line_pixels']:6,} line pixels")
        
        print(f"\nVisualizations saved to: {output_dir}/")
        print(f"  Files: frame_XXXXX_segmentation.jpg")
        
        # Overall keypoint type summary
        all_types = {}
        for r in results:
            for kp_type, count in r['keypoint_types'].items():
                all_types[kp_type] = all_types.get(kp_type, 0) + count
        
        if all_types:
            print(f"\nTotal keypoints by type across all frames:")
            for kp_type, count in sorted(all_types.items()):
                print(f"  {kp_type:20s}: {count:3d}")
        
        # Write HTML report (how it did)
        html_path = Path("output") / "segmentation_37a_results.html"
        write_html_report(results, output_dir, video_path, width, height, html_path, all_types)
        print(f"\nHTML report: {html_path}")
    
    print("\n✓ Test complete!")


def write_html_report(results, output_dir, video_path, width, height, html_path, all_types=None):
    """Write HTML report showing how the pipeline did (stats + pipeline description)."""
    from pathlib import Path
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    n = len(results)
    avg_lines = np.mean([r['lines'] for r in results]) if results else 0
    avg_keypoints = np.mean([r['keypoints'] for r in results]) if results else 0
    avg_line_pixels = np.mean([r['line_pixels'] for r in results]) if results else 0
    all_types = all_types or {}
    
    frames_html = ""
    for r in results:
        fn = r['frame']
        img_src = f"segmentation_37a_frames/frame_{fn:05d}_segmentation.jpg"
        badges = "".join(
            f'<span class="keypoint-badge">{kp_type}: {count}</span>'
            for kp_type, count in sorted(r.get('keypoint_types', {}).items())
        )
        frames_html += f"""
        <div class="frame-section">
            <div class="frame-header">
                <div class="frame-title">Frame {fn}</div>
                <div class="frame-stats">
                    <div class="frame-stat">{r['lines']} Lines</div>
                    <div class="frame-stat">{r['keypoints']} Keypoints</div>
                    <div class="frame-stat">{r['line_pixels']:,} Line Pixels</div>
                </div>
            </div>
            <img src="{img_src}" alt="Frame {fn}" class="frame-image" onclick="window.open(this.src, '_blank')">
            <div class="keypoint-types">{badges}</div>
        </div>
"""
    
    kp_items = ""
    for kp_type, count in sorted(all_types.items()):
        kp_items += f'<div class="keypoint-item"><div class="keypoint-name">{kp_type}</div><div class="keypoint-count">{count} detections</div></div>\n'
    if not kp_items:
        kp_items = '<div class="keypoint-item"><div class="keypoint-name">(none)</div><div class="keypoint-count">No keypoints</div></div>'
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Segmentation Test Results - 37a Video</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); padding: 30px; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; text-align: center; }}
        .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 1.1em; }}
        .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin-bottom: 30px; }}
        .summary h2 {{ margin-bottom: 15px; font-size: 1.8em; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .stat-box {{ background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .keypoint-breakdown {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px; color: #333; }}
        .keypoint-breakdown h3 {{ margin-bottom: 15px; color: #2c3e50; }}
        .keypoint-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .keypoint-item {{ background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #667eea; color: #333; }}
        .keypoint-name {{ font-weight: bold; color: #2c3e50; }}
        .keypoint-count {{ color: #667eea; font-size: 1.2em; }}
        .frame-section {{ margin-bottom: 40px; border: 2px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white; }}
        .frame-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }}
        .frame-title {{ font-size: 1.5em; font-weight: bold; }}
        .frame-stats {{ display: flex; gap: 20px; font-size: 0.9em; }}
        .frame-stat {{ background: rgba(255, 255, 255, 0.2); padding: 5px 15px; border-radius: 20px; }}
        .frame-image {{ width: 100%; display: block; cursor: pointer; }}
        .keypoint-types {{ padding: 15px 20px; background: #f8f9fa; display: flex; flex-wrap: wrap; gap: 10px; }}
        .keypoint-badge {{ background: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85em; border: 1px solid #ddd; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Semantic Segmentation Test Results</h1>
        <div class="subtitle">37a Video - {n} Frames | How it did</div>
        
        <div class="summary">
            <h2>Overall Performance</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;">
                <strong>Pipeline:</strong> Mask → closing + skeletonization → Hough (rho=1, minLineLength, maxLineGap=30) → VP + DBSCAN merge → virtual keypoints (L_long ∩ L_trans). Center circle: HoughCircles on mask; temporal averaging when static.
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;">
                <strong>How it did:</strong> Avg {avg_lines:.0f} lines/frame, {avg_keypoints:.1f} keypoints/frame, {avg_line_pixels:,.0f} line pixels. White = any shade (HSV+LAB+gray). Virtual keypoints from merged lines when VP/clustering succeed.
            </div>
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-value">{avg_lines:.0f}</div><div class="stat-label">Avg Lines/Frame</div></div>
                <div class="stat-box"><div class="stat-value">{avg_keypoints:.1f}</div><div class="stat-label">Avg Keypoints/Frame</div></div>
                <div class="stat-box"><div class="stat-value">{avg_line_pixels:,.0f}</div><div class="stat-label">Avg Line Pixels</div></div>
                <div class="stat-box"><div class="stat-value">{width}×{height}</div><div class="stat-label">Video Resolution</div></div>
            </div>
            <div class="keypoint-breakdown">
                <h3>Keypoint Detection Summary</h3>
                <div class="keypoint-list">{kp_items}</div>
            </div>
        </div>
        {frames_html}
        <div class="footer">
            <p>Generated by test_segmentation_37a_frames.py | Framework-based pipeline</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Re-run script to refresh. Use test_segmentation_with_averaging.py for temporal averaging.</p>
        </div>
    </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
