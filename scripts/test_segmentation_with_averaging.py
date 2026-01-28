#!/usr/bin/env python3
"""
Test semantic segmentation with temporal averaging for static camera.
Averages masks across multiple frames to improve detection of faint off-white elements.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector
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
    cv2.putText(canvas, "2. Averaged Segmentation Mask", (w + 10, 30), 
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
    
    # Panel 4: Lines + Keypoints (bottom right)
    keypoints_vis = lines_vis.copy()
    center_circle_kps = [kp for kp in keypoints if kp.landmark_type == 'center_circle']
    other_kps = [kp for kp in keypoints if kp.landmark_type != 'center_circle']
    
    # Draw other keypoints in blue
    for kp in other_kps:
        x, y = int(kp.image_point[0]), int(kp.image_point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(keypoints_vis, (x, y), 8, (255, 0, 0), -1)
    
    # Draw center circle keypoints in green
    for kp in center_circle_kps:
        x, y = int(kp.image_point[0]), int(kp.image_point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(keypoints_vis, (x, y), 8, (0, 255, 0), -1)
    
    canvas[h:2*h, w:2*w] = keypoints_vis
    cv2.putText(canvas, f"4. Lines + Keypoints ({len(keypoints)})", (w + 10, h + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add legend
    legend_y = h + 70
    cv2.circle(canvas, (w + 20, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(canvas, f"Center Circle ({len(center_circle_kps)})", (w + 40, legend_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(canvas, (w + 20, legend_y + 30), 8, (255, 0, 0), -1)
    cv2.putText(canvas, f"Other ({len(other_kps)})", (w + 40, legend_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved visualization: {output_path}")


def test_with_averaging(video_path: str, frame_id: int, num_frames: int = 10):
    """
    Test segmentation with temporal averaging on a specific frame.
    
    Args:
        video_path: Path to video file
        frame_id: Center frame to test
        num_frames: Number of frames to average (should be odd, centered on frame_id)
    """
    print(f"Testing frame {frame_id} with {num_frames} frame averaging...")
    
    # Load frames around the target frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_buffer = []
    start_frame = max(0, frame_id - num_frames // 2)
    end_frame = frame_id + num_frames // 2 + 1
    
    for f in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
        else:
            break
    
    cap.release()
    
    if len(frame_buffer) < 3:
        print(f"Error: Could not load enough frames (got {len(frame_buffer)})")
        return
    
    print(f"Loaded {len(frame_buffer)} frames for averaging")
    
    # Use center frame as the main image
    center_image = frame_buffer[len(frame_buffer) // 2]
    
    # Create segmenter and detector
    segmenter = PitchLineSegmenter(model_path=None)
    detector = PitchKeypointDetector(pitch_length=105.0, pitch_width=68.0)
    detector.enable_zero_shot = False
    detector.use_semantic_segmentation = True
    detector._line_segmenter = segmenter
    
    # Get averaged segmentation mask
    averaged_mask = segmenter.segment_pitch_lines_averaged(frame_buffer)
    
    # Detect lines and keypoints with frame averaging
    lines = detector._detect_field_lines(center_image, frame_buffer=frame_buffer)
    keypoints = detector.detect_all_keypoints(center_image, frame_buffer=frame_buffer)
    
    # Create visualization
    output_dir = Path('output/segmentation_37a_frames')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'frame_{frame_id:05d}_averaged_segmentation.jpg'
    
    create_detailed_visualization(center_image, averaged_mask, lines, keypoints, output_path)
    
    # Print statistics
    center_circle_kps = [kp for kp in keypoints if kp.landmark_type == 'center_circle']
    print(f"\nResults:")
    print(f"  Lines detected: {len(lines) if lines is not None and len(lines) > 0 else 0}")
    print(f"  Total keypoints: {len(keypoints)}")
    print(f"  Center circle keypoints: {len(center_circle_kps)}")
    if center_circle_kps:
        print(f"  ✅ Center circle detected!")
    else:
        print(f"  ❌ Center circle NOT detected")
    
    return output_path


if __name__ == "__main__":
    video_path = "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
    
    # Test on frame 253 with averaging
    test_with_averaging(video_path, frame_id=253, num_frames=10)
    
    print("\nDone! Check output/segmentation_37a_frames/")
