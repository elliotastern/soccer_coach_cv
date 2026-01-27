#!/usr/bin/env python3
"""
Validation tool to compare video processing results against ground truth or visual validation.
Creates side-by-side comparisons and validation metrics.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add rf-detr directory to path if it exists
rfdetr_path = Path("/workspace/rf-detr")
if rfdetr_path.exists():
    sys.path.insert(0, str(rfdetr_path))

from src.analysis.mapping import PitchMapper
from src.analysis.homography import HomographyEstimator

# Optional RF-DETR import for re-running detection
try:
    from rfdetr import RFDETRMedium
    RFDETR_AVAILABLE = True
except ImportError as e:
    RFDETR_AVAILABLE = False
    RFDETRMedium = None
    # Note: debug not available at module level, will print in function


def draw_pitch_diagram(image, pitch_length=105.0, pitch_width=68.0, scale=10):
    """
    Draw a standard soccer pitch diagram on an image.
    
    Args:
        image: Image to draw on
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        scale: Pixels per meter
    """
    h, w = image.shape[:2]
    
    # Calculate pitch dimensions in pixels
    pitch_w_px = int(pitch_width * scale)
    pitch_h_px = int(pitch_length * scale)
    
    # Center the pitch
    start_x = (w - pitch_w_px) // 2
    start_y = (h - pitch_h_px) // 2
    
    # Draw pitch outline
    cv2.rectangle(image, (start_x, start_y), 
                  (start_x + pitch_w_px, start_y + pitch_h_px), 
                  (255, 255, 255), 2)
    
    # Draw center line
    center_x = start_x + pitch_w_px // 2
    cv2.line(image, (center_x, start_y), (center_x, start_y + pitch_h_px), 
             (255, 255, 255), 1)
    
    # Draw center circle
    center_y = start_y + pitch_h_px // 2
    cv2.circle(image, (center_x, center_y), int(pitch_w_px * 0.15), 
               (255, 255, 255), 1)
    
    # Draw penalty boxes (16.5m from goal line, 40.32m wide)
    penalty_depth = int(16.5 * scale)
    penalty_width = int(40.32 * scale)
    penalty_start_y = (pitch_w_px - penalty_width) // 2
    
    # Left penalty box
    cv2.rectangle(image, 
                  (start_x, start_y + penalty_start_y),
                  (start_x + penalty_depth, start_y + penalty_start_y + penalty_width),
                  (255, 255, 255), 1)
    
    # Right penalty box
    cv2.rectangle(image,
                  (start_x + pitch_w_px - penalty_depth, start_y + penalty_start_y),
                  (start_x + pitch_w_px, start_y + penalty_start_y + penalty_width),
                  (255, 255, 255), 1)
    
    # Draw goals
    goal_width = int(7.32 * scale)
    goal_start_y = (pitch_w_px - goal_width) // 2
    
    # Left goal
    cv2.rectangle(image,
                  (start_x - 2, start_y + goal_start_y),
                  (start_x, start_y + goal_start_y + goal_width),
                  (255, 255, 0), 2)
    
    # Right goal
    cv2.rectangle(image,
                  (start_x + pitch_w_px, start_y + goal_start_y),
                  (start_x + pitch_w_px + 2, start_y + goal_start_y + goal_width),
                  (255, 255, 0), 2)
    
    return image


def create_validation_visualization(results_path: str, video_path: str, 
                                   output_dir: str, num_frames: int = 10,
                                   model_path: str = None, re_run_detection: bool = False,
                                   debug: bool = True):
    """
    Create validation visualization comparing predictions with pitch diagram.
    
    Args:
        results_path: Path to results JSON
        video_path: Path to source video
        output_dir: Output directory for validation images
        num_frames: Number of frames to validate
        model_path: Optional path to RF-DETR model to re-run detection
        re_run_detection: If True, re-run detection and compare with stored bboxes
        debug: Enable extensive debugging output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if debug:
        print("="*70)
        print("🔍 DEBUG MODE ENABLED")
        print("="*70)
        print(f"Results path: {results_path}")
        print(f"Video path: {video_path}")
        print(f"Output dir: {output_dir}")
        print()
    
    # Load results
    if debug:
        print("📂 Loading results.json...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if debug:
        print(f"   ✅ Loaded {len(results)} frame results")
        if results:
            print(f"   First frame keys: {list(results[0].keys())}")
            if results[0].get('players'):
                print(f"   First frame has {len(results[0]['players'])} players")
                if results[0]['players']:
                    print(f"   First player keys: {list(results[0]['players'][0].keys())}")
                    print(f"   First player bbox: {results[0]['players'][0].get('bbox')}")
        print()
    
    # Optionally load RF-DETR model to re-run detection
    detector = None
    if re_run_detection and model_path:
        if debug:
            print(f"🔍 Attempting to load RF-DETR model from: {model_path}")
        if not RFDETR_AVAILABLE:
            print("⚠️  RF-DETR not available (module not installed)")
            print("   Install with: pip install rfdetr")
            print("   Continuing with stored bboxes only")
            re_run_detection = False
        else:
            print("🔍 Loading RF-DETR model to re-run detection...")
            try:
                detector = RFDETRMedium(pretrain_weights=model_path)
                # RF-DETR doesn't have eval() method, it's always in eval mode for inference
                if hasattr(detector, 'eval'):
                    detector.eval()
                print("✅ Model loaded - will compare stored bboxes with fresh detections")
                if debug:
                    print(f"   Model type: {type(detector)}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                print("   Continuing with stored bboxes only")
                re_run_detection = False
    
    # Open video
    if debug:
        print(f"📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if debug:
        print(f"   ✅ Video opened successfully")
        print(f"   Video properties:")
        print(f"      Width: {width}px")
        print(f"      Height: {height}px")
        print(f"      FPS: {fps:.2f}")
        print(f"      Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print()
    
    # Initialize homography for pitch mapping (simple 4-point initialization)
    # Note: We don't actually need homography for validation since bounding boxes
    # are pixel coordinates, but we initialize it for pitch position validation
    homography_estimator = HomographyEstimator()
    manual_points = {
        'image_points': [[0, 0], [width, 0], [width, height], [0, height]],
        'pitch_points': [[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]]
    }
    
    # Read first frame to initialize
    ret, first_frame = cap.read()
    if ret:
        homography_estimator.estimate(first_frame, manual_points)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    # Create pitch diagram template
    diagram_height = 800
    diagram_width = 1200
    pitch_diagram = np.zeros((diagram_height, diagram_width, 3), dtype=np.uint8)
    pitch_diagram[:, :] = [34, 139, 34]  # Green background
    pitch_diagram = draw_pitch_diagram(pitch_diagram, scale=8)
    
    validation_results = []
    frame_idx = 0
    
    print("="*70)
    print("VALIDATION: Comparing Predictions vs Pitch Diagram")
    print("="*70)
    print(f"Processing {num_frames} frames for validation...")
    print()
    
    while frame_idx < num_frames and cap.isOpened():
        if debug:
            print("="*70)
            print(f"🔍 PROCESSING FRAME {frame_idx}")
            print("="*70)
        
        # Use set() to ensure we read the exact frame (cap.read() might be sequential)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            if debug:
                print(f"   ❌ Failed to read frame {frame_idx}")
            break
        
        # Verify frame dimensions match
        frame_h, frame_w = frame.shape[:2]
        if debug:
            print(f"   Frame read: {frame_w}x{frame_h} (HxW from shape)")
            print(f"   Expected: {width}x{height} (from video properties)")
            print(f"   Video position set to: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
        
        if frame_w != width or frame_h != height:
            print(f"⚠️  Warning: Frame {frame_idx} size mismatch: video={frame_w}x{frame_h}, expected={width}x{height}")
            if debug:
                print(f"   ⚠️  DIMENSION MISMATCH - This could cause bbox misalignment!")
        
        # Find corresponding result
        if debug:
            print(f"   Searching for frame_id={frame_idx} in results...")
        result = None
        for r in results:
            if r.get('frame_id') == frame_idx:
                result = r
                break
        
        if not result:
            if debug:
                print(f"   ⚠️  No result found for frame {frame_idx}")
            frame_idx += 1
            continue
        
        if 'players' not in result:
            if debug:
                print(f"   ⚠️  Result found but no 'players' key")
            frame_idx += 1
            continue
        
        # Verify frame_id matches
        result_frame_id = result.get('frame_id')
        if result_frame_id != frame_idx:
            print(f"⚠️  Warning: Frame index mismatch! Video frame={frame_idx}, Result frame_id={result_frame_id}")
        
        if debug:
            print(f"   ✅ Found result with {len(result['players'])} players")
            print(f"   Result frame_id: {result_frame_id}")
            print(f"   Result keys: {list(result.keys())}")
            
            # Verify frame hash for first frame
            if frame_idx == 0:
                import hashlib
                h, w = frame.shape[:2]
                center_region = frame[h//4:3*h//4, w//4:3*w//4]
                frame_hash = hashlib.md5(center_region.tobytes()).hexdigest()[:8]
                print(f"   Frame hash (center region): {frame_hash}")
            print()
        
        # Create side-by-side visualization
        vis_width = width + diagram_width + 40
        vis_height = max(height, diagram_height) + 100
        if debug:
            print(f"   📐 Creating visualization canvas:")
            print(f"      Canvas size: {vis_width}x{vis_height}")
            print(f"      Frame size: {width}x{height}")
            print(f"      Diagram size: {diagram_width}x{diagram_height}")
        
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        visualization[:, :] = [240, 240, 240]  # Light gray background
        
        if debug:
            print(f"      ✅ Canvas created: {visualization.shape}")
        
        # Left side: Original frame with annotations
        frame_annotated = frame.copy()
        
        # Add visual verification markers (corners and grid) to verify coordinate system
        if debug:
            # Draw corner markers
            cv2.circle(frame_annotated, (0, 0), 10, (0, 255, 0), -1)  # Top-left (green)
            cv2.circle(frame_annotated, (width-1, 0), 10, (0, 255, 0), -1)  # Top-right
            cv2.circle(frame_annotated, (width-1, height-1), 10, (0, 255, 0), -1)  # Bottom-right
            cv2.circle(frame_annotated, (0, height-1), 10, (0, 255, 0), -1)  # Bottom-left
            cv2.circle(frame_annotated, (width//2, height//2), 10, (255, 0, 255), -1)  # Center (magenta)
            
            # Draw grid lines every 1000 pixels (for 6500x1000 frame)
            grid_spacing_x = 1000
            grid_spacing_y = 200
            for x in range(0, width, grid_spacing_x):
                cv2.line(frame_annotated, (x, 0), (x, height), (128, 128, 128), 1)
            for y in range(0, height, grid_spacing_y):
                cv2.line(frame_annotated, (0, y), (width, y), (128, 128, 128), 1)
            
            print(f"   📐 Added visual verification markers:")
            print(f"      Corner markers: (0,0), ({width-1},0), ({width-1},{height-1}), (0,{height-1})")
            print(f"      Center marker: ({width//2}, {height//2})")
            print(f"      Grid: X every {grid_spacing_x}px, Y every {grid_spacing_y}px")
        
        # Add visual verification markers (corners and grid) to verify coordinate system
        if debug:
            # Draw corner markers
            cv2.circle(frame_annotated, (0, 0), 10, (0, 255, 0), -1)  # Top-left (green)
            cv2.circle(frame_annotated, (width-1, 0), 10, (0, 255, 0), -1)  # Top-right
            cv2.circle(frame_annotated, (width-1, height-1), 10, (0, 255, 0), -1)  # Bottom-right
            cv2.circle(frame_annotated, (0, height-1), 10, (0, 255, 0), -1)  # Bottom-left
            cv2.circle(frame_annotated, (width//2, height//2), 10, (255, 0, 255), -1)  # Center (magenta)
            
            # Draw grid lines every 1000 pixels (for 6500x1000 frame)
            grid_spacing_x = 1000
            grid_spacing_y = 200
            for x in range(0, width, grid_spacing_x):
                cv2.line(frame_annotated, (x, 0), (x, height), (128, 128, 128), 1)
            for y in range(0, height, grid_spacing_y):
                cv2.line(frame_annotated, (0, y), (width, y), (128, 128, 128), 1)
            
            print(f"   📐 Added visual verification markers:")
            print(f"      Corner markers: (0,0), ({width-1},0), ({width-1},{height-1}), (0,{height-1})")
            print(f"      Center marker: ({width//2}, {height//2})")
            print(f"      Grid: X every {grid_spacing_x}px, Y every {grid_spacing_y}px")
        
        # Optionally re-run detection to compare
        fresh_detections = None
        if re_run_detection and detector is not None:
            if debug:
                print(f"   🔄 Re-running RF-DETR detection on frame {frame_idx}...")
                print(f"      Frame shape: {frame.shape} (BGR)")
            
            # Convert BGR to RGB for RF-DETR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if debug:
                print(f"      Converted to RGB: {frame_rgb.shape}")
            
            detections_raw = detector.predict(frame_rgb, threshold=0.3)
            
            if debug:
                print(f"      Raw detections type: {type(detections_raw)}")
                if hasattr(detections_raw, 'class_id'):
                    print(f"      Raw detections count: {len(detections_raw.class_id)}")
            
            # Convert to same format as stored
            fresh_detections = []
            if hasattr(detections_raw, 'class_id'):
                for i in range(len(detections_raw.class_id)):
                    if detections_raw.class_id[i] == 1:  # Person class
                        bbox_xyxy = detections_raw.xyxy[i]
                        x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
                        width_fresh = x_max - x_min
                        height_fresh = y_max - y_min
                        fresh_detections.append({
                            'bbox': [x_min, y_min, width_fresh, height_fresh],
                            'confidence': float(detections_raw.confidence[i])
                        })
            
            if debug:
                print(f"      ✅ Fresh detections: {len(fresh_detections)} players")
                if fresh_detections:
                    print(f"      First fresh bbox: {fresh_detections[0]['bbox']}")
                print()
        
        # Draw detections on frame
        if debug:
            print(f"   🎨 Drawing {len(result['players'])} players on frame...")
        
        for player_idx, player in enumerate(result['players']):
            if debug:
                print(f"      Player {player_idx + 1}/{len(result['players'])}:")
            
            # Bbox format is [x, y, w, h] (top-left corner + width/height)
            bbox = player['bbox']
            if debug:
                print(f"         Raw bbox from JSON: {bbox} (type: {type(bbox)}, len: {len(bbox) if hasattr(bbox, '__len__') else 'N/A'})")
            
            if len(bbox) != 4:
                if debug:
                    print(f"         ⚠️  Skipping: bbox length is {len(bbox)}, expected 4")
                continue
                
            # Bbox format is [x, y, w, h] (top-left corner + width/height)
            # Use exact coordinates as stored (they should be correct from RF-DETR)
            x, y, w, h = [float(v) for v in bbox]
            
            if debug:
                print(f"         Parsed bbox: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")
                print(f"         Bbox center: ({x + w/2:.2f}, {y + h/2:.2f})")
                print(f"         Bbox right edge: {x + w:.2f} (frame width: {width})")
                print(f"         Bbox bottom edge: {y + h:.2f} (frame height: {height})")
                print(f"         Bbox in bounds: {0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height}")
            
            # Convert to integers for drawing (same as diagnostic script)
            x_int = int(x)
            y_int = int(y)
            w_int = int(w)
            h_int = int(h)
            
            if debug:
                print(f"         Integer conversion: x_int={x_int}, y_int={y_int}, w_int={w_int}, h_int={h_int}")
            
            # Calculate bottom-right corner
            x2_int = x_int + w_int
            y2_int = y_int + h_int
            
            if debug:
                print(f"         Bottom-right corner: ({x2_int}, {y2_int})")
            
            # Only clamp if absolutely necessary (outside frame bounds)
            needs_clamping = x_int < 0 or y_int < 0 or x2_int > width or y2_int > height
            if needs_clamping:
                if debug:
                    print(f"         ⚠️  NEEDS CLAMPING:")
                    print(f"            Before: x={x_int}, y={y_int}, x2={x2_int}, y2={y2_int}")
                    print(f"            Frame bounds: 0-{width-1} x 0-{height-1}")
                
                # Clamp to bounds
                x_int_old = x_int
                y_int_old = y_int
                x2_int_old = x2_int
                y2_int_old = y2_int
                
                x_int = max(0, min(x_int, width - 1))
                y_int = max(0, min(y_int, height - 1))
                x2_int = min(x2_int, width - 1)
                y2_int = min(y2_int, height - 1)
                w_int = x2_int - x_int
                h_int = y2_int - y_int
                
                if debug:
                    print(f"            After: x={x_int}, y={y_int}, x2={x2_int}, y2={y2_int}")
                    print(f"            Changes: x={x_int_old}->{x_int}, y={y_int_old}->{y_int}, x2={x2_int_old}->{x2_int}, y2={y2_int_old}->{y2_int}")
            else:
                if debug:
                    print(f"         ✅ No clamping needed")
            
            team_id = player.get('team_id')
            
            # Color by team
            if team_id == 0:
                color = (0, 0, 255)  # Red
                label = "Team 0"
            elif team_id == 1:
                color = (255, 0, 0)  # Blue
                label = "Team 1"
            else:
                color = (0, 255, 255)  # Yellow
                label = "Unassigned"
            
            # Draw bounding box (using integer coordinates)
            # Use EXACT same approach as debug_bbox_alignment.py which was working
            if debug and player_idx == 0:  # Detailed debug for first player
                print(f"         🖼️  Drawing rectangle:")
                print(f"            Top-left: ({x_int}, {y_int})")
                print(f"            Bottom-right: ({x2_int}, {y2_int})")
                print(f"            Color (BGR): {color}")
                print(f"            Thickness: 3 (matching debug_bbox_alignment.py)")
                print(f"            Frame bounds: 0-{width-1} x 0-{height-1}")
                # Check pixel before drawing
                check_y = max(0, min(y_int, height-1))
                check_x = max(0, min(x_int, width-1))
                pixel_before = frame_annotated[check_y, check_x].copy()
                print(f"            Pixel before at ({check_x}, {check_y}): {pixel_before}")
            
            # Draw rectangle - EXACT same as debug_bbox_alignment.py (thickness 3)
            cv2.rectangle(frame_annotated, (x_int, y_int), (x2_int, y2_int), color, 3)
            
            # Also draw center point like debug script (for verification)
            if debug and player_idx < 3:  # Draw center for first 3 players
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                cv2.circle(frame_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
            
            if debug and player_idx == 0:  # Verify drawing for first player
                check_y = max(0, min(y_int, height-1))
                check_x = max(0, min(x_int, width-1))
                pixel_after = frame_annotated[check_y, check_x]
                print(f"            Pixel after at ({check_x}, {check_y}): {pixel_after}")
                if np.array_equal(pixel_after, pixel_before):
                    print(f"            ⚠️  WARNING: Pixel unchanged - rectangle may not have been drawn!")
                else:
                    print(f"            ✅ Pixel changed - rectangle drawn successfully")
                print(f"            Rectangle valid: {0 <= x_int < width and 0 <= y_int < height and x2_int <= width and y2_int <= height}")
            
            # If re-running detection, compare with fresh detection
            if re_run_detection and fresh_detections:
                if debug:
                    print(f"         🔄 Comparing with fresh detections...")
                
                # Find closest fresh detection
                stored_center = (x + w/2, y + h/2)
                if debug:
                    print(f"            Stored center: ({stored_center[0]:.2f}, {stored_center[1]:.2f})")
                
                min_dist = float('inf')
                closest_fresh = None
                closest_idx = -1
                
                for fresh_idx, fresh in enumerate(fresh_detections):
                    fx, fy, fw, fh = fresh['bbox']
                    fresh_center = (fx + fw/2, fy + fh/2)
                    dist = np.sqrt((stored_center[0] - fresh_center[0])**2 + 
                                 (stored_center[1] - fresh_center[1])**2)
                    if debug and fresh_idx < 3:  # Debug first 3
                        print(f"            Fresh {fresh_idx}: center=({fresh_center[0]:.2f}, {fresh_center[1]:.2f}), dist={dist:.2f}")
                    if dist < min_dist:
                        min_dist = dist
                        closest_fresh = fresh
                        closest_idx = fresh_idx
                
                if debug:
                    if closest_fresh:
                        fx, fy, fw, fh = closest_fresh['bbox']
                        fresh_center = (fx + fw/2, fy + fh/2)
                        print(f"            ✅ Closest match: fresh #{closest_idx}, distance={min_dist:.2f}px")
                        print(f"               Stored: ({stored_center[0]:.2f}, {stored_center[1]:.2f})")
                        print(f"               Fresh:  ({fresh_center[0]:.2f}, {fresh_center[1]:.2f})")
                    else:
                        print(f"            ⚠️  No closest match found!")
                
                # Draw fresh detection in different color if mismatch
                if closest_fresh and min_dist > 10:  # Significant mismatch
                    fx, fy, fw, fh = closest_fresh['bbox']
                    fx_int, fy_int = int(fx), int(fy)
                    fx2_int, fy2_int = int(fx + fw), int(fy + fh)
                    # Draw in green to show mismatch
                    cv2.rectangle(frame_annotated, (fx_int, fy_int), (fx2_int, fy2_int), (0, 255, 0), 1)
                    print(f"  ⚠️  Frame {frame_idx} Player {player_idx+1}: BBOX MISMATCH!")
                    print(f"      Stored center: ({stored_center[0]:.1f}, {stored_center[1]:.1f})")
                    print(f"      Fresh center: ({fx + fw/2:.1f}, {fy + fh/2:.1f})")
                    print(f"      Distance: {min_dist:.1f}px")
                    print(f"      Stored bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
                    print(f"      Fresh bbox:  [{fx:.1f}, {fy:.1f}, {fw:.1f}, {fh:.1f}]")
                elif debug:
                    if closest_fresh:
                        print(f"            ✅ Match is good (distance={min_dist:.2f}px <= 10px)")
            
            # Draw label
            label_y = max(15, y_int - 10)  # Ensure label is visible
            cv2.putText(frame_annotated, label, (x_int, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw pitch position
            pitch_pos = player.get('pitch_position', [0, 0])
            if isinstance(pitch_pos, (list, tuple)) and len(pitch_pos) >= 2:
                pitch_x, pitch_y = float(pitch_pos[0]), float(pitch_pos[1])
            else:
                pitch_x, pitch_y = 0.0, 0.0
                
            info = f"({pitch_x:.1f}, {pitch_y:.1f})m"
            info_y = min(height - 5, y2_int + 20)
            cv2.putText(frame_annotated, info, (x_int, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Place annotated frame on left
        # CRITICAL: Bboxes are drawn on frame_annotated, which is then placed in canvas
        # The bbox coordinates are relative to frame_annotated, NOT the canvas
        # So placing frame at (20, 20) offset is correct - bboxes stay in same position
        if debug:
            print(f"   📍 Placing annotated frame:")
            print(f"      Source: frame_annotated {frame_annotated.shape} (H, W, C)")
            print(f"      Destination: visualization[20:{20+height}, 20:{20+width}]")
            print(f"      Visualization shape: {visualization.shape} (H, W, C)")
            print(f"      Note: OpenCV uses (y, x) indexing for arrays")
            print(f"      Note: Bboxes are drawn on frame_annotated BEFORE placement")
            print(f"      Note: Canvas offset (20, 20) does NOT affect bbox coordinates")
        
        # Verify frame dimensions match before placement
        if frame_annotated.shape[0] != height or frame_annotated.shape[1] != width:
            print(f"      ⚠️  CRITICAL: Frame dimension mismatch!")
            print(f"         frame_annotated.shape = {frame_annotated.shape}")
            print(f"         Expected (height, width) = ({height}, {width})")
            print(f"         This WILL cause bbox misalignment!")
        
        visualization[20:20+height, 20:20+width] = frame_annotated
        
        if debug:
            # Verify placement
            check_y, check_x = 20, 20
            placed_pixel = visualization[check_y, check_x]
            original_pixel = frame_annotated[0, 0]
            print(f"      ✅ Frame placed")
            print(f"         Verification: visualization[{check_y}, {check_x}] = {placed_pixel}")
            print(f"         Original: frame_annotated[0, 0] = {original_pixel}")
            if not np.array_equal(placed_pixel, original_pixel):
                print(f"         ⚠️  PIXEL MISMATCH!")
            
            # Check bbox pixel location - verify it's correctly placed
            if result['players']:
                first_player = result['players'][0]
                if len(first_player.get('bbox', [])) == 4:
                    px, py, pw, ph = [float(v) for v in first_player['bbox']]
                    px_int, py_int = int(px), int(py)
                    # Bbox coordinates are relative to frame, so in visualization they're at:
                    vis_bbox_x = 20 + px_int  # X offset + bbox x
                    vis_bbox_y = 20 + py_int  # Y offset + bbox y
                    if 0 <= vis_bbox_x < vis_width and 0 <= vis_bbox_y < vis_height:
                        vis_bbox_pixel = visualization[vis_bbox_y, vis_bbox_x]
                        frame_bbox_pixel = frame_annotated[py_int, px_int]
                        print(f"         Bbox pixel check:")
                        print(f"            Bbox in frame: ({px_int}, {py_int})")
                        print(f"            Bbox in visualization: ({vis_bbox_x}, {vis_bbox_y})")
                        print(f"            Visualization[{vis_bbox_y}, {vis_bbox_x}] = {vis_bbox_pixel}")
                        print(f"            Frame[{py_int}, {px_int}] = {frame_bbox_pixel}")
                        if not np.array_equal(vis_bbox_pixel, frame_bbox_pixel):
                            print(f"            ⚠️  BBOX PIXEL MISMATCH!")
                        else:
                            print(f"            ✅ Bbox pixel matches - canvas placement correct")
            print()
        
        # Right side: Pitch diagram with player positions
        diagram_copy = pitch_diagram.copy()
        
        # First, collect all pitch coordinates to determine range and center
        pitch_coords = []
        for player in result['players']:
            if 'pitch_position' in player:
                pitch_x, pitch_y = player['pitch_position']
            elif 'x_pitch' in player and 'y_pitch' in player:
                pitch_x = player['x_pitch']
                pitch_y = player['y_pitch']
            else:
                continue
            pitch_coords.append((pitch_x, pitch_y))
        
        # Calculate auto-scaling if we have coordinates
        if pitch_coords:
            x_coords = [p[0] for p in pitch_coords]
            y_coords = [p[1] for p in pitch_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range = x_max - x_min if x_max > x_min else 105.0
            y_range = y_max - y_min if y_max > y_min else 68.0
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Scale to fit diagram (with some margin)
            scale_x = (diagram_width * 0.9) / max(x_range, 105.0)
            scale_y = (diagram_height * 0.9) / max(y_range, 68.0)
            scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
            
            diagram_center_x = diagram_width // 2
            diagram_center_y = diagram_height // 2
        else:
            # Fallback to standard scaling
            scale = 8  # pixels per meter
            x_center, y_center = 0.0, 0.0
            diagram_center_x = diagram_width // 2
            diagram_center_y = diagram_height // 2
        
        for player in result['players']:
            # Handle both pitch_position (tuple) and x_pitch/y_pitch formats
            if 'pitch_position' in player:
                pitch_x, pitch_y = player['pitch_position']
            elif 'x_pitch' in player and 'y_pitch' in player:
                pitch_x = player['x_pitch']
                pitch_y = player['y_pitch']
            else:
                continue  # Skip players without pitch coordinates
            team_id = player.get('team_id')
            
            # Convert pitch coordinates to diagram coordinates
            # Center coordinates relative to actual center, then scale
            diagram_x = int(diagram_center_x + (pitch_x - x_center) * scale)
            diagram_y = int(diagram_center_y - (pitch_y - y_center) * scale)  # Flip Y axis
            
            # Ensure within bounds
            if 0 <= diagram_x < diagram_width and 0 <= diagram_y < diagram_height:
                # Color by team
                if team_id == 0:
                    color = (0, 0, 255)  # Red
                elif team_id == 1:
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Draw player position
                cv2.circle(diagram_copy, (diagram_x, diagram_y), 5, color, -1)
                cv2.circle(diagram_copy, (diagram_x, diagram_y), 8, (255, 255, 255), 1)
        
        visualization[20:20+diagram_height, 20+width+20:20+width+20+diagram_width] = diagram_copy
        
        # Add labels
        cv2.putText(visualization, f"Frame {frame_idx} (t={result.get('timestamp', 0):.2f}s)", 
                   (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(visualization, "Original Frame + Detections", 
                   (20, height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(visualization, "Pitch Diagram + Player Positions", 
                   (20+width+20, diagram_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        output_path = output_dir / f"validation_frame_{frame_idx:03d}.jpg"
        if debug:
            print(f"   💾 Saving visualization to: {output_path}")
            print(f"      Visualization size: {visualization.shape}")
            print(f"      Frame annotated size: {frame_annotated.shape}")
            print(f"      Frame annotated placed at: (0, 0) to ({width}, {height})")
        
        cv2.imwrite(str(output_path), visualization)
        
        if debug:
            # Verify the saved image
            saved_img = cv2.imread(str(output_path))
            if saved_img is not None:
                print(f"      ✅ Image saved successfully: {saved_img.shape}")
            else:
                print(f"      ⚠️  Failed to verify saved image")
            print()
        
        # Calculate validation metrics
        num_players = len(result['players'])
        num_assigned = sum(1 for p in result['players'] if p.get('team_id') is not None)
        assignment_rate = num_assigned / num_players if num_players > 0 else 0
        
        # Check if positions are within valid pitch bounds
        valid_positions = 0
        for player in result['players']:
            # Handle both pitch_position (tuple) and x_pitch/y_pitch formats
            if 'pitch_position' in player:
                pitch_x, pitch_y = player['pitch_position']
            elif 'x_pitch' in player and 'y_pitch' in player:
                pitch_x = player['x_pitch']
                pitch_y = player['y_pitch']
            else:
                continue
            # Standard pitch is 105m x 68m, with some margin
            # After calibration, coordinates may be offset, so use wider range
            # Check if within reasonable field bounds (allowing for offset)
            if -100 <= pitch_x <= 100 and -250 <= pitch_y <= 50:
                valid_positions += 1
        
        position_validity = valid_positions / num_players if num_players > 0 else 0
        
        validation_results.append({
            'frame_id': frame_idx,
            'num_players': num_players,
            'num_assigned': num_assigned,
            'assignment_rate': assignment_rate,
            'valid_positions': valid_positions,
            'position_validity': position_validity
        })
        
        print(f"Frame {frame_idx:3d}: {num_players} players | "
              f"Assignment: {assignment_rate*100:.1f}% | "
              f"Valid positions: {position_validity*100:.1f}%")
        
        frame_idx += 1
    
    cap.release()
    
    # Save validation summary
    summary = {
        'total_frames_validated': len(validation_results),
        'average_assignment_rate': np.mean([r['assignment_rate'] for r in validation_results]),
        'average_position_validity': np.mean([r['position_validity'] for r in validation_results]),
        'frame_results': validation_results
    }
    
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Frames validated: {summary['total_frames_validated']}")
    print(f"Average assignment rate: {summary['average_assignment_rate']*100:.1f}%")
    print(f"Average position validity: {summary['average_position_validity']*100:.1f}%")
    print()
    print(f"✅ Validation images saved to: {output_dir}")
    print(f"✅ Summary saved to: {summary_path}")
    print("="*70)
    
    return summary


def create_validation_html(output_dir: str):
    """Create HTML viewer for validation results"""
    output_dir = Path(output_dir)
    
    # Find validation images
    validation_images = sorted(output_dir.glob("validation_frame_*.jpg"))
    
    if not validation_images:
        print("No validation images found")
        return
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Results - Predictions vs Pitch Diagram</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .validation-grid {{
            display: grid;
            gap: 20px;
            margin: 20px 0;
        }}
        .validation-item {{
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background: #f9f9f9;
        }}
        .validation-item img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        .validation-item h3 {{
            margin: 10px 0 5px 0;
            color: #2196F3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Results: Predictions vs Pitch Diagram</h1>
        <p>Side-by-side comparison of detected player positions (left) with pitch diagram visualization (right)</p>
        
        <div class="validation-grid">
"""
    
    for img_path in validation_images:
        frame_num = img_path.stem.split('_')[-1]
        html_content += f"""
            <div class="validation-item">
                <h3>Frame {frame_num}</h3>
                <img src="{img_path.name}" alt="Validation frame {frame_num}">
            </div>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    html_path = output_dir / "validation_viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Validation HTML viewer created: {html_path}")
    return html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate video processing results")
    parser.add_argument("results", type=str, help="Path to results JSON file")
    parser.add_argument("video", type=str, help="Path to source video file")
    parser.add_argument("--output", "-o", type=str, default="output/validation",
                       help="Output directory for validation images")
    parser.add_argument("--num-frames", "-n", type=int, default=10,
                       help="Number of frames to validate")
    parser.add_argument("--model", "-m", type=str, 
                       default="models/rf_detr_soccertrack/checkpoint0099.pth",
                       help="Path to RF-DETR model to re-run detection and compare (default: checkpoint0099 - 99 epoch trained model)")
    parser.add_argument("--re-run-detection", action="store_true",
                       help="Re-run detection with RF-DETR model and compare with stored bboxes")
    parser.add_argument("--debug", action="store_true", default=True,
                       help="Enable extensive debugging output (default: True)")
    
    args = parser.parse_args()
    
    summary = create_validation_visualization(
        args.results,
        args.video,
        args.output,
        args.num_frames,
        model_path=args.model,
        re_run_detection=args.re_run_detection,
        debug=args.debug
    )
    
    create_validation_html(args.output)
