#!/usr/bin/env python3
"""
Validation viewer: Shows bounding boxes and 2D pitch maps side-by-side.
Focuses on validating bbox accuracy and 2D mapping correctness.
Extracts frames from video and generates viewer that loads frame_data.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root for cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# cv2 imported only if needed in extract_frames


def extract_frames(video_path: str, output_dir: Path, num_frames: int) -> None:
    """Extract first num_frames from video to output_dir/frames/ as frame_XXX.jpg."""
    try:
        import cv2
    except ImportError:
        print("⚠️  cv2 not available, skipping frame extraction (assuming frames already exist)")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out_path = frames_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
    cap.release()
    print(f"Extracted {min(num_frames, i + 1)} frames to {frames_dir}")


def create_viewer_html(
    frame_data_path: str,
    output_html: str,
    output_dir: Path,
    title: str = "Validation: Bbox + 2D Map",
) -> Path:
    """Generate HTML validation viewer: frame image with bboxes (left) + 2D pitch map (right)."""
    with open(frame_data_path, "r") as f:
        results = json.load(f)

    # Flatten players for pitch map and bounds
    all_players = []
    for r in results:
        for p in r.get("players", []):
            all_players.append(p)
    x_min = min((p.get("x_pitch", 0) for p in all_players), default=0)
    x_max = max((p.get("x_pitch", 0) for p in all_players), default=0)
    y_min = min((p.get("y_pitch", 0) for p in all_players), default=0)
    y_max = max((p.get("y_pitch", 0) for p in all_players), default=0)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    output_dir = Path(output_dir)
    frames_prefix = "frames"

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>""",
        title,
        """</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .stat-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
        .stat-card .value { font-size: 24px; font-weight: bold; color: #333; }
        .frame-pitch-map { width: 100%; height: 100%; min-width: 0; border: 2px solid #333; background: #2d5016; position: relative; border-radius: 5px; }
        .player-dot { position: absolute; width: 10px; height: 10px; border-radius: 50%; border: 2px solid white; cursor: pointer; }
        .player-dot.team-0 { background: #ff4444; }
        .player-dot.team-1 { background: #4444ff; }
        .player-dot.unassigned { background: #ffff44; }
        .player-dot:hover { transform: scale(1.5); z-index: 10; }
        .frame-results { margin: 20px 0; }
        .frame-card { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #2196F3; }
        .frame-header { font-weight: bold; color: #2196F3; margin-bottom: 10px; font-size: 16px; }
        .frame-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; align-items: stretch; }
        .frame-row .img-wrap { width: 100%; min-height: 0; }
        .img-wrap { position: relative; display: block; line-height: 0; }
        .img-wrap img { display: block; width: 100%; height: auto; vertical-align: bottom; }
        .bbox-overlay { position: absolute; border: 2px solid; pointer-events: none; box-sizing: border-box; }
        .bbox-overlay.team-0 { border-color: #ff4444; }
        .bbox-overlay.team-1 { border-color: #4444ff; }
        .bbox-overlay.unassigned { border-color: #ffff44; }
        .validation-info { margin-top: 10px; padding: 10px; background: #e3f2fd; border-radius: 5px; font-size: 12px; color: #1976d2; }
        .validation-info strong { color: #0d47a1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>""",
        title,
        """</h1>
        <div class="stats">
            <div class="stat-card"><h3>Frames</h3><div class="value" id="numFrames">0</div></div>
            <div class="stat-card"><h3>Total detections</h3><div class="value" id="totalDet">0</div></div>
            <div class="stat-card"><h3>Pitch bounds (m)</h3><div class="value" id="pitchBounds">-</div></div>
            <div class="stat-card"><h3>Model</h3><div class="value">person (trained)</div></div>
        </div>
        <h2>Frame-by-frame validation (bbox + 2D map)</h2>
        <div class="frame-results" id="frameList"></div>
    <script>
        const results = """,
        json.dumps(results),
        """;
        const framesPrefix = """,
        json.dumps(frames_prefix),
        """;
        const pitchBounds = { "x_min": """,
        str(x_min),
        """, "x_max": """,
        str(x_max),
        """, "y_min": """,
        str(y_min),
        """, "y_max": """,
        str(y_max),
        """ };
        let totalDet = 0;
        results.forEach((r) => { if (r.players) totalDet += r.players.length; });
        document.getElementById('numFrames').textContent = results.length;
        document.getElementById('totalDet').textContent = totalDet;
        document.getElementById('pitchBounds').textContent = `x:[${pitchBounds.x_min.toFixed(1)}, ${pitchBounds.x_max.toFixed(1)}] y:[${pitchBounds.y_min.toFixed(1)}, ${pitchBounds.y_max.toFixed(1)}]`;

        const pad = 20;
        const xR = pitchBounds.x_max - pitchBounds.x_min;
        const yR = pitchBounds.y_max - pitchBounds.y_min;

        function fillPitchMap(mapEl, players) {
            const mapW = mapEl.offsetWidth;
            const mapH = mapEl.offsetHeight;
            (players || []).forEach(p => {
                const px = p.x_pitch != null ? p.x_pitch : (p.pitch_position ? p.pitch_position[0] : 0);
                const py = p.y_pitch != null ? p.y_pitch : (p.pitch_position ? p.pitch_position[1] : 0);
                const nx = (px - pitchBounds.x_min) / xR;
                const ny = (py - pitchBounds.y_min) / yR;
                const x = pad + nx * (mapW - 2 * pad);
                const y = pad + ny * (mapH - 2 * pad);
                const dot = document.createElement('div');
                dot.className = 'player-dot ' + (p.team_id === 0 ? 'team-0' : p.team_id === 1 ? 'team-1' : 'unassigned');
                dot.style.left = x + 'px';
                dot.style.top = y + 'px';
                dot.title = 'ID: ' + (p.object_id != null ? p.object_id : '?') + ' | Team: ' + (p.team_id != null && p.team_id >= 0 ? p.team_id : 'unassigned') + ' | (' + px.toFixed(2) + ', ' + py.toFixed(2) + ') m';
                mapEl.appendChild(dot);
            });
        }

        // Frame list: each row = image + bboxes (left) | 2D pitch map for that frame (right)
        const list = document.getElementById('frameList');
        results.forEach((frameData) => {
            const sec = document.createElement('div');
            sec.className = 'frame-card';
            const numPlayers = (frameData.players || []).length;
            sec.innerHTML = '<div class="frame-header">Frame ' + frameData.frame_id + ' (t=' + (frameData.timestamp || 0).toFixed(2) + 's) – ' + numPlayers + ' players</div>';
            const row = document.createElement('div');
            row.className = 'frame-row';
            const wrap = document.createElement('div');
            wrap.className = 'img-wrap';
            const img = document.createElement('img');
            img.src = framesPrefix + '/frame_' + String(frameData.frame_id).padStart(3, '0') + '.jpg';
            img.alt = 'Frame ' + frameData.frame_id;
            (frameData.players || []).forEach(p => {
                const b = p.bbox || [0,0,0,0];
                const x = b[0], y = b[1], w = b[2], h = b[3];
                const div = document.createElement('div');
                div.className = 'bbox-overlay ' + (p.team_id === 0 ? 'team-0' : p.team_id === 1 ? 'team-1' : 'unassigned');
                div.style.left = (x / img.naturalWidth * 100) + '%';
                div.style.top = (y / img.naturalHeight * 100) + '%';
                div.style.width = (w / img.naturalWidth * 100) + '%';
                div.style.height = (h / img.naturalHeight * 100) + '%';
                wrap.appendChild(div);
            });
            img.onload = function() {
                const nw = img.naturalWidth;
                const nh = img.naturalHeight;
                wrap.querySelectorAll('.bbox-overlay').forEach((div, i) => {
                    const p = (frameData.players || [])[i];
                    if (!p || !p.bbox) return;
                    const b = p.bbox;
                    const x = b[0], y = b[1], w = b[2], h = b[3];
                    div.style.left = (x / nw * 100) + '%';
                    div.style.top = (y / nh * 100) + '%';
                    div.style.width = (w / nw * 100) + '%';
                    div.style.height = (h / nh * 100) + '%';
                });
            };
            wrap.appendChild(img);
            row.appendChild(wrap);
            const mapEl = document.createElement('div');
            mapEl.className = 'frame-pitch-map';
            fillPitchMap(mapEl, frameData.players || []);
            row.appendChild(mapEl);
            sec.appendChild(row);
            const info = document.createElement('div');
            info.className = 'validation-info';
            const teamCounts = {};
            (frameData.players || []).forEach(p => {
                const tid = p.team_id != null && p.team_id >= 0 ? p.team_id : 'unassigned';
                teamCounts[tid] = (teamCounts[tid] || 0) + 1;
            });
            const teamStr = Object.entries(teamCounts).map(([t, c]) => `Team ${t}: ${c}`).join(', ');
            info.innerHTML = '<strong>Validation:</strong> ' + numPlayers + ' detections | ' + teamStr + ' | Pitch range: x=[' + pitchBounds.x_min.toFixed(1) + ', ' + pitchBounds.x_max.toFixed(1) + ']m, y=[' + pitchBounds.y_min.toFixed(1) + ', ' + pitchBounds.y_max.toFixed(1) + ']m';
            sec.appendChild(info);
            list.appendChild(sec);
        });
    </script>
    </div>
</body>
</html>""",
    ]

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("".join(html_parts))
    print(f"Created: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames and create validation viewer (bbox + 2D map)"
    )
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument("frame_data", type=str, help="Path to frame_data.json")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--num-frames", "-n", type=int, default=20, help="Number of frames")
    parser.add_argument("--title", "-t", type=str, default="Validation: Bbox + 2D Map")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(args.video, output_dir, args.num_frames)
    create_viewer_html(
        args.frame_data,
        str(output_dir / "viewer_with_frames_2d_maps.html"),
        output_dir,
        title=args.title,
    )


if __name__ == "__main__":
    main()
