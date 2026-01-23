#!/usr/bin/env python3
"""
Create an interactive HTML viewer for model predictions with toggleable boxes
"""
import json
import cv2
import numpy as np
from pathlib import Path
import base64
from PIL import Image
import io


def image_to_base64(image_path):
    """Convert image to base64 data URL"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    ext = image_path.suffix[1:].lower()
    return f"data:image/{ext};base64,{img_base64}"


def create_viewer_html(output_dir: str, summary_path: str = None):
    """Create interactive HTML viewer for predictions"""
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    
    # Load summary if available
    summary = {}
    if summary_path and Path(summary_path).exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    # Get all frame images
    frame_files = sorted(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        print(f"No frame images found in {frames_dir}")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Load predictions from saved frames (we'll need to re-run inference or save predictions)
    # For now, let's create a viewer that shows the frames and allows manual annotation
    # We'll need to re-run inference with threshold=0.0 to get all predictions
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Predictions Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .controls {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #45a049;
        }}
        button.active {{
            background: #2196F3;
        }}
        input[type="range"] {{
            width: 200px;
        }}
        .frame-container {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .frame-info {{
            margin-bottom: 10px;
            font-size: 14px;
            color: #aaa;
        }}
        .frame-image {{
            position: relative;
            display: inline-block;
            max-width: 100%;
        }}
        .frame-image img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        .prediction-box {{
            position: absolute;
            border: 2px solid #00ff00;
            background: rgba(0, 255, 0, 0.1);
            pointer-events: none;
        }}
        .prediction-label {{
            position: absolute;
            top: -20px;
            left: 0;
            background: rgba(0, 255, 0, 0.8);
            color: #000;
            padding: 2px 6px;
            font-size: 12px;
            font-weight: bold;
        }}
        .stats {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .stat-item {{
            background: #333;
            padding: 10px;
            border-radius: 5px;
        }}
        .stat-label {{
            color: #aaa;
            font-size: 12px;
        }}
        .stat-value {{
            color: #4CAF50;
            font-size: 24px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Model Predictions Viewer</h1>
    
    <div class="stats">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Total Frames</div>
                <div class="stat-value" id="total-frames">{len(frame_files)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Current Frame</div>
                <div class="stat-value" id="current-frame">1</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Detections (All)</div>
                <div class="stat-value" id="total-detections">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Detections (Visible)</div>
                <div class="stat-value" id="visible-detections">0</div>
            </div>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <button onclick="previousFrame()">◀ Previous</button>
            <button onclick="nextFrame()">Next ▶</button>
        </div>
        <div class="control-group">
            <button id="toggle-boxes" onclick="toggleBoxes()" class="active">Hide Boxes</button>
        </div>
        <div class="control-group">
            <label>Confidence Threshold:</label>
            <input type="range" id="confidence-slider" min="0" max="1" step="0.001" value="0" 
                   oninput="updateThreshold(this.value)">
            <span id="threshold-value">0.000</span>
        </div>
        <div class="control-group">
            <button onclick="goToFrame(1)">Frame 1</button>
            <input type="number" id="frame-input" min="1" max="{len(frame_files)}" value="1" 
                   onchange="goToFrame(parseInt(this.value))" style="width: 80px;">
            <span>of {len(frame_files)}</span>
        </div>
    </div>
    
    <div id="frames-container"></div>
    
    <script>
        let currentFrameIndex = 0;
        let showBoxes = true;
        let confidenceThreshold = 0.0;
        let predictionsData = [];
        
        // Load predictions data
        async function loadPredictions() {{
            try {{
                const response = await fetch('predictions.json');
                predictionsData = await response.json();
                console.log('Loaded predictions:', predictionsData.length);
                renderFrame(currentFrameIndex);
            }} catch (error) {{
                console.error('Error loading predictions:', error);
                // Create empty predictions array
                predictionsData = Array({len(frame_files)}).fill({{detections: []}});
                renderFrame(currentFrameIndex);
            }}
        }}
        
        function renderFrame(index) {{
            if (index < 0 || index >= {len(frame_files)}) return;
            
            currentFrameIndex = index;
            const frameFile = 'frame_' + String(index).padStart(6, '0') + '.jpg';
            const framePath = `frames/${{frameFile}}`;
            
            const predictions = predictionsData[index] || {{detections: []}};
            const filteredDetections = predictions.detections.filter(d => d.score >= confidenceThreshold);
            
            document.getElementById('current-frame').textContent = index + 1;
            document.getElementById('total-detections').textContent = predictions.detections.length;
            document.getElementById('visible-detections').textContent = filteredDetections.length;
            
            let html = `
                <div class="frame-container">
                    <div class="frame-info">
                        Frame ${{index + 1}} / {len(frame_files)} | 
                        Detections: ${{predictions.detections.length}} (showing: ${{filteredDetections.length}})
                    </div>
                    <div class="frame-image" id="frame-image">
                        <img src="${{framePath}}" alt="Frame ${{index + 1}}" id="frame-img" 
                             onload="drawBoxes()">
                        <div id="boxes-container"></div>
                    </div>
                </div>
            `;
            
            document.getElementById('frames-container').innerHTML = html;
            
            // Store detections for drawing
            window.currentDetections = filteredDetections;
            window.frameImage = document.getElementById('frame-img');
            
            // Draw boxes after image loads
            if (window.frameImage.complete) {{
                drawBoxes();
            }}
        }}
        
        function drawBoxes() {{
            const container = document.getElementById('boxes-container');
            if (!container || !showBoxes) {{
                container.innerHTML = '';
                return;
            }}
            
            const detections = window.currentDetections || [];
            const img = window.frameImage;
            if (!img) return;
            
            // Get actual image dimensions
            const imgRect = img.getBoundingClientRect();
            const scaleX = imgRect.width / img.naturalWidth;
            const scaleY = imgRect.height / img.naturalHeight;
            
            container.innerHTML = '';
            container.style.position = 'absolute';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = imgRect.width + 'px';
            container.style.height = imgRect.height + 'px';
            
            detections.forEach((det, idx) => {{
                const [x1, y1, x2, y2] = det.bbox;
                const box = document.createElement('div');
                box.className = 'prediction-box';
                box.style.left = (x1 * scaleX) + 'px';
                box.style.top = (y1 * scaleY) + 'px';
                box.style.width = ((x2 - x1) * scaleX) + 'px';
                box.style.height = ((y2 - y1) * scaleY) + 'px';
                
                const label = document.createElement('div');
                label.className = 'prediction-label';
                label.textContent = `${{det.class_name}}: ${{(det.score * 100).toFixed(2)}}%`;
                box.appendChild(label);
                
                container.appendChild(box);
            }});
        }}
        
        function nextFrame() {{
            if (currentFrameIndex < {len(frame_files)} - 1) {{
                renderFrame(currentFrameIndex + 1);
            }}
        }}
        
        function previousFrame() {{
            if (currentFrameIndex > 0) {{
                renderFrame(currentFrameIndex - 1);
            }}
        }}
        
        function goToFrame(frameNum) {{
            const index = Math.max(0, Math.min({len(frame_files)} - 1, frameNum - 1));
            renderFrame(index);
            document.getElementById('frame-input').value = index + 1;
        }}
        
        function toggleBoxes() {{
            showBoxes = !showBoxes;
            const btn = document.getElementById('toggle-boxes');
            btn.textContent = showBoxes ? 'Hide Boxes' : 'Show Boxes';
            btn.classList.toggle('active', showBoxes);
            drawBoxes();
        }}
        
        function updateThreshold(value) {{
            confidenceThreshold = parseFloat(value);
            document.getElementById('threshold-value').textContent = value;
            renderFrame(currentFrameIndex);
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === 'ArrowLeft') previousFrame();
            if (e.key === ' ') {{ e.preventDefault(); toggleBoxes(); }}
        }});
        
        // Initialize
        loadPredictions();
    </script>
</body>
</html>
"""
    
    # Save HTML
    html_path = output_dir / "viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created viewer HTML: {html_path}")
    return html_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create interactive prediction viewer")
    parser.add_argument("--output-dir", type=str, default="data/test_output_training",
                        help="Directory containing frames and predictions")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to summary.json file")
    
    args = parser.parse_args()
    
    create_viewer_html(args.output_dir, args.summary)
