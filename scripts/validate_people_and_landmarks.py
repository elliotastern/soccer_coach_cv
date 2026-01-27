#!/usr/bin/env python3
"""
Validation tool that highlights detected landmarks, all people, and 2D representation.
Creates a comprehensive visualization showing what the AI can see.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_keypoint_detector import PitchKeypointDetector
from src.analysis.mapping import PitchMapper


def draw_pitch_diagram(image, pitch_length=105.0, pitch_width=68.0, scale=10):
    """Draw a standard soccer pitch diagram on an image."""
    h, w = image.shape[:2]
    pitch_w_px = int(pitch_width * scale)
    pitch_h_px = int(pitch_length * scale)
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
    
    return image


def get_landmark_color(landmark_type: str):
    """Get color for different landmark types."""
    colors = {
        'goal': (0, 255, 255),  # Yellow
        'center_circle': (255, 0, 255),  # Magenta
        'corner': (255, 255, 0),  # Cyan
        'penalty_box': (0, 165, 255),  # Orange
        'center_line': (255, 255, 255),  # White
        'touchline': (0, 255, 0),  # Green
        'penalty_spot': (255, 0, 0),  # Blue
        'goal_area': (128, 0, 128),  # Purple
    }
    return colors.get(landmark_type, (128, 128, 128))  # Gray default


def visualize_landmarks(frame, keypoints):
    """Draw detected landmarks on frame."""
    frame_with_landmarks = frame.copy()
    
    # Group by type for legend
    by_type = {}
    for kp in keypoints:
        kp_type = kp.landmark_type
        if kp_type not in by_type:
            by_type[kp_type] = []
        by_type[kp_type].append(kp)
    
    # Draw landmarks
    for kp_type, kps in by_type.items():
        color = get_landmark_color(kp_type)
        for kp in kps:
            x, y = int(kp.image_point[0]), int(kp.image_point[1])
            # Draw circle for landmark
            cv2.circle(frame_with_landmarks, (x, y), 8, color, -1)
            cv2.circle(frame_with_landmarks, (x, y), 12, (255, 255, 255), 2)
            # Draw label
            label = kp_type.replace('_', ' ').title()
            cv2.putText(frame_with_landmarks, label, (x + 15, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame_with_landmarks, by_type


def main():
    parser = argparse.ArgumentParser(description='Validate people and landmarks')
    parser.add_argument('results_path', type=str, help='Path to frame_data.json')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='validation_landmarks',
                       help='Output directory')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames to process')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_path) as f:
        results = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize landmark detector
    landmark_detector = PitchKeypointDetector()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pitch diagram template
    diagram_height = 800
    diagram_width = 1200
    pitch_diagram = np.zeros((diagram_height, diagram_width, 3), dtype=np.uint8)
    pitch_diagram[:, :] = [34, 139, 34]  # Green background
    pitch_diagram = draw_pitch_diagram(pitch_diagram, scale=8)
    
    # HTML output
    html_content = []
    html_content.append("""
<!DOCTYPE html>
<html>
<head>
    <title>People and Landmarks Validation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .frame-container { margin: 20px 0; background: white; padding: 20px; border-radius: 8px; }
        .frame-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .frame-image { max-width: 100%; border: 2px solid #333; }
        .stats { margin-top: 10px; font-size: 14px; color: #666; }
        .legend { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
        .legend-item { display: inline-block; margin: 5px 15px; }
        .legend-color { display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
    </style>
</head>
<body>
    <h1>People and Landmarks Validation</h1>
    <div class="legend">
        <strong>Landmark Colors:</strong>
        <span class="legend-item"><span class="legend-color" style="background: yellow;"></span>Goal</span>
        <span class="legend-item"><span class="legend-color" style="background: magenta;"></span>Center Circle</span>
        <span class="legend-item"><span class="legend-color" style="background: cyan;"></span>Corner</span>
        <span class="legend-item"><span class="legend-color" style="background: orange;"></span>Penalty Box</span>
        <span class="legend-item"><span class="legend-color" style="background: white; border: 1px solid black;"></span>Center Line</span>
        <span class="legend-item"><span class="legend-color" style="background: green;"></span>Touchline</span>
        <span class="legend-item"><span class="legend-color" style="background: blue;"></span>Penalty Spot</span>
        <span class="legend-item"><span class="legend-color" style="background: purple;"></span>Goal Area</span>
    </div>
""")
    
    frame_idx = 0
    num_frames = min(args.num_frames, len(results))
    
    print("="*70)
    print("VALIDATION: People and Landmarks")
    print("="*70)
    print(f"Processing {num_frames} frames...")
    print()
    
    while frame_idx < num_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find corresponding result
        result = None
        for r in results:
            if r.get('frame_id') == frame_idx:
                result = r
                break
        
        if not result:
            frame_idx += 1
            continue
        
        # Detect landmarks
        keypoints = landmark_detector.detect_all_keypoints(frame)
        
        # Visualize landmarks on frame
        frame_with_landmarks, landmarks_by_type = visualize_landmarks(frame, keypoints)
        
        # Draw all people
        for player in result.get('players', []):
            bbox = player.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = [int(b) for b in bbox]
                team_id = player.get('team_id', -1)
                
                # Color by team
                if team_id == 0:
                    color = (0, 0, 255)  # Red
                elif team_id == 1:
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Draw bounding box
                cv2.rectangle(frame_with_landmarks, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"P{player.get('object_id', '?')} T{team_id}"
                cv2.putText(frame_with_landmarks, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Create visualization canvas
        vis_width = width + diagram_width + 40
        vis_height = max(height, diagram_height) + 100
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        visualization[:, :] = [240, 240, 240]  # Light gray
        
        # Left side: Frame with landmarks and people
        visualization[20:20+height, 20:20+width] = frame_with_landmarks
        
        # Right side: Pitch diagram with player positions
        diagram_copy = pitch_diagram.copy()
        scale = 8
        diagram_center_x = diagram_width // 2
        diagram_center_y = diagram_height // 2
        
        # Collect pitch coordinates for auto-scaling
        pitch_coords = []
        for player in result.get('players', []):
            if 'x_pitch' in player and 'y_pitch' in player:
                pitch_coords.append((player['x_pitch'], player['y_pitch']))
        
        # Auto-scale if we have coordinates
        if pitch_coords:
            x_coords = [p[0] for p in pitch_coords]
            y_coords = [p[1] for p in pitch_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range = x_max - x_min if x_max > x_min else 105.0
            y_range = y_max - y_min if y_max > y_min else 68.0
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            scale_x = (diagram_width * 0.9) / max(x_range, 105.0)
            scale_y = (diagram_height * 0.9) / max(y_range, 68.0)
            scale = min(scale_x, scale_y)
        else:
            x_center, y_center = 0.0, 0.0
        
        # Draw player positions on pitch
        for player in result.get('players', []):
            if 'x_pitch' in player and 'y_pitch' in player:
                pitch_x = player['x_pitch']
                pitch_y = player['y_pitch']
                team_id = player.get('team_id', -1)
                
                diagram_x = int(diagram_center_x + (pitch_x - x_center) * scale)
                diagram_y = int(diagram_center_y - (pitch_y - y_center) * scale)
                
                if 0 <= diagram_x < diagram_width and 0 <= diagram_y < diagram_height:
                    if team_id == 0:
                        color = (0, 0, 255)  # Red
                    elif team_id == 1:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 255, 255)  # Yellow
                    
                    cv2.circle(diagram_copy, (diagram_x, diagram_y), 5, color, -1)
                    cv2.circle(diagram_copy, (diagram_x, diagram_y), 8, (255, 255, 255), 1)
        
        # Place diagram
        visualization[20:20+diagram_height, 20+width+20:20+width+20+diagram_width] = diagram_copy
        
        # Add labels
        cv2.putText(visualization, f"Frame {frame_idx} (t={result.get('timestamp', 0):.2f}s)", 
                   (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(visualization, "Frame + Landmarks + People", 
                   (20, height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(visualization, "2D Pitch Representation", 
                   (20+width+20, diagram_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save image
        img_path = output_dir / f"validation_frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(img_path), visualization)
        
        # Add to HTML
        landmark_counts = {k: len(v) for k, v in landmarks_by_type.items()}
        landmark_summary = ", ".join([f"{k}: {v}" for k, v in landmark_counts.items()])
        
        html_content.append(f"""
    <div class="frame-container">
        <div class="frame-header">Frame {frame_idx} (t={result.get('timestamp', 0):.2f}s)</div>
        <img src="{img_path.name}" class="frame-image" alt="Frame {frame_idx}">
        <div class="stats">
            <strong>People:</strong> {len(result.get('players', []))} | 
            <strong>Landmarks:</strong> {len(keypoints)} ({landmark_summary})
        </div>
    </div>
""")
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    # Close HTML
    html_content.append("""
</body>
</html>
""")
    
    # Save HTML
    html_path = output_dir / "validate_people_and_landmark.html"
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_content))
    
    print()
    print("="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"✅ Validation images saved to: {output_dir}")
    print(f"✅ HTML viewer created: {html_path}")
    print("="*70)


if __name__ == "__main__":
    main()
