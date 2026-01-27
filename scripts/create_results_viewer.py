#!/usr/bin/env python3
"""
Create HTML viewer for video processing results showing R-002 and R-003 outputs.
"""
import json
from pathlib import Path
import argparse


def create_results_viewer(results_path: str, output_html: str, title: str = "Video Processing Results"):
    """Create HTML viewer for results"""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get statistics
    all_players = []
    for r in results:
        if 'players' in r:
            all_players.extend(r['players'])
    
    team_0 = [p for p in all_players if p.get('team_id') == 0]
    team_1 = [p for p in all_players if p.get('team_id') == 1]
    unassigned = [p for p in all_players if p.get('team_id') is None]
    
    # Calculate pitch bounds
    if all_players:
        pitch_x = [p.get('x_pitch', p.get('pitch_position', [0, 0])[0]) for p in all_players]
        pitch_y = [p.get('y_pitch', p.get('pitch_position', [0, 0])[1]) for p in all_players]
        x_min, x_max = min(pitch_x), max(pitch_x)
        y_min, y_max = min(pitch_y), max(pitch_y)
    else:
        x_min = x_max = y_min = y_max = 0
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .pitch-map {{
            width: 100%;
            height: 500px;
            border: 2px solid #333;
            background: #2d5016;
            position: relative;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .player-dot {{
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            border: 2px solid white;
        }}
        .team-0 {{ background: #ff4444; }}
        .team-1 {{ background: #4444ff; }}
        .unassigned {{ background: #ffff44; }}
        .frame-results {{
            margin: 20px 0;
        }}
        .frame-card {{
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .frame-header {{
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }}
        .player-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }}
        .player-item {{
            background: white;
            padding: 8px;
            border-radius: 3px;
            font-size: 12px;
            border-left: 3px solid #ccc;
        }}
        .player-item.team-0 {{ border-left-color: #ff4444; }}
        .player-item.team-1 {{ border-left-color: #4444ff; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Frames Processed</h3>
                <div class="value">{len(results)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Detections</h3>
                <div class="value">{len(all_players)}</div>
            </div>
            <div class="stat-card">
                <h3>Team 0 Players</h3>
                <div class="value">{len(team_0)}</div>
            </div>
            <div class="stat-card">
                <h3>Team 1 Players</h3>
                <div class="value">{len(team_1)}</div>
            </div>
            <div class="stat-card">
                <h3>Unassigned</h3>
                <div class="value">{len(unassigned)}</div>
            </div>
        </div>
        
        <h2>Pitch Position Map</h2>
        <div class="pitch-map" id="pitchMap">
            <!-- Player positions will be plotted here -->
        </div>
        
        <h2>Frame-by-Frame Results</h2>
        <div class="frame-results">
"""
    
    # Add frame results
    for result in results[:20]:  # Show first 20 frames
        if 'players' in result and len(result['players']) > 0:
            frame_id = result.get('frame_id', 0)
            timestamp = result.get('timestamp', 0)
            
            html_content += f"""
            <div class="frame-card">
                <div class="frame-header">Frame {frame_id} (t={timestamp:.2f}s) - {len(result['players'])} players</div>
                <div class="player-list">
"""
            for player in result['players'][:10]:  # Show first 10 players per frame
                team_id = player.get('team_id')
                team_class = f"team-{team_id}" if team_id is not None else "unassigned"
                team_str = f"Team {team_id}" if team_id is not None else "Unassigned"
                # Calculate pixel center from bbox
                bbox = player.get('bbox', [0, 0, 0, 0])
                px = bbox[0] + bbox[2] / 2 if len(bbox) >= 3 else 0
                py = bbox[1] + bbox[3] / 2 if len(bbox) >= 4 else 0
                pitch_x = player.get('x_pitch', 0)
                pitch_y = player.get('y_pitch', 0)
                conf = player.get('confidence', 0)
                
                html_content += f"""
                    <div class="player-item {team_class}">
                        <strong>{team_str}</strong> | 
                        Pixel: ({px:.0f}, {py:.0f}) | 
                        Pitch: ({pitch_x:.2f}, {pitch_y:.2f}) m | 
                        Conf: {conf:.3f}
                    </div>
"""
            html_content += """
                </div>
            </div>
"""
    
    # Add JavaScript for pitch map
    html_content += f"""
        </div>
    </div>
    
    <script>
        // Draw pitch map
        const map = document.getElementById('pitchMap');
        const width = map.offsetWidth;
        const height = map.offsetHeight;
        
        const xRange = {x_max} - {x_min};
        const yRange = {y_max} - {y_min};
        
        const players = {json.dumps(all_players[:100])};  // Limit to 100 for performance
        
        players.forEach(player => {{
            const pitchX = player.x_pitch !== undefined ? player.x_pitch : (player.pitch_position ? player.pitch_position[0] : 0);
            const pitchY = player.y_pitch !== undefined ? player.y_pitch : (player.pitch_position ? player.pitch_position[1] : 0);
            
            // Normalize to 0-1
            const normX = (pitchX - {x_min}) / xRange;
            const normY = (pitchY - {y_min}) / yRange;
            
            // Convert to pixel position (with padding)
            const padding = 20;
            const x = padding + normX * (width - 2 * padding);
            const y = padding + normY * (height - 2 * padding);
            
            const dot = document.createElement('div');
            dot.className = 'player-dot';
            if (player.team_id === 0) {{
                dot.classList.add('team-0');
            }} else if (player.team_id === 1) {{
                dot.classList.add('team-1');
            }} else {{
                dot.classList.add('unassigned');
            }}
            dot.style.left = x + 'px';
            dot.style.top = y + 'px';
            dot.title = `Team ${{player.team_id || '?'}} | (${{pitchX.toFixed(2)}}, ${{pitchY.toFixed(2)}}) m`;
            map.appendChild(dot);
        }});
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML viewer created: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HTML viewer for video processing results")
    parser.add_argument("results", type=str, help="Path to results JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output HTML file path")
    parser.add_argument("--title", "-t", type=str, default="Video Processing Results",
                       help="Title for the viewer")
    
    args = parser.parse_args()
    
    output_path = args.output or args.results.replace('.json', '_viewer.html')
    create_results_viewer(args.results, output_path, args.title)
