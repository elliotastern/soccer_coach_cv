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

from src.analysis.mapping import PitchMapper
from src.analysis.homography import HomographyEstimator


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
                                   output_dir: str, num_frames: int = 10):
    """
    Create validation visualization comparing predictions with pitch diagram.
    
    Args:
        results_path: Path to results JSON
        video_path: Path to source video
        output_dir: Output directory for validation images
        num_frames: Number of frames to validate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize homography for pitch mapping
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
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find corresponding result
        result = None
        for r in results:
            if r.get('frame_id') == frame_idx:
                result = r
                break
        
        if not result or 'players' not in result:
            frame_idx += 1
            continue
        
        # Create side-by-side visualization
        vis_width = width + diagram_width + 40
        vis_height = max(height, diagram_height) + 100
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        visualization[:, :] = [240, 240, 240]  # Light gray background
        
        # Left side: Original frame with annotations
        frame_annotated = frame.copy()
        
        # Draw detections on frame
        for player in result['players']:
            x, y, w, h = [int(v) for v in player['bbox']]
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
            
            # Draw bounding box
            cv2.rectangle(frame_annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(frame_annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw pitch position
            pitch_x, pitch_y = player['pitch_position']
            info = f"({pitch_x:.1f}, {pitch_y:.1f})m"
            cv2.putText(frame_annotated, info, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        visualization[20:20+height, 20:20+width] = frame_annotated
        
        # Right side: Pitch diagram with player positions
        diagram_copy = pitch_diagram.copy()
        
        # Map pitch positions to diagram coordinates
        # Pitch is 105m x 68m, diagram is scaled
        scale = 8  # pixels per meter
        diagram_center_x = diagram_width // 2
        diagram_center_y = diagram_height // 2
        
        for player in result['players']:
            pitch_x, pitch_y = player['pitch_position']
            team_id = player.get('team_id')
            
            # Convert pitch coordinates to diagram coordinates
            # Pitch origin is at center (0, 0), diagram origin is at top-left
            diagram_x = int(diagram_center_x + pitch_x * scale)
            diagram_y = int(diagram_center_y - pitch_y * scale)  # Flip Y axis
            
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
        cv2.imwrite(str(output_path), visualization)
        
        # Calculate validation metrics
        num_players = len(result['players'])
        num_assigned = sum(1 for p in result['players'] if p.get('team_id') is not None)
        assignment_rate = num_assigned / num_players if num_players > 0 else 0
        
        # Check if positions are within valid pitch bounds
        valid_positions = 0
        for player in result['players']:
            pitch_x, pitch_y = player['pitch_position']
            # Standard pitch is 105m x 68m, with some margin
            if -60 <= pitch_x <= 60 and -40 <= pitch_y <= 40:
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
    
    args = parser.parse_args()
    
    summary = create_validation_visualization(
        args.results,
        args.video,
        args.output,
        args.num_frames
    )
    
    create_validation_html(args.output)
