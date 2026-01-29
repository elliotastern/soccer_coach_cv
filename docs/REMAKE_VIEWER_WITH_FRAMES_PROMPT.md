# How to remake the “frames + bounding boxes” viewer (prompt)

Use the prompt below to have an AI or a dev recreate the same viewer from scratch.

---

## Copy-paste prompt

```
Remake the “frames with bounding boxes” viewer for this project.

**What it does:**
1. Takes (a) a video file, (b) a frame_data.json (array of per-frame results), and (c) an output directory.
2. Extracts the first N frames from the video and saves them as images in output_dir/frames/ (e.g. frame_000.jpg, frame_001.jpg, …).
3. Generates a single HTML file (e.g. viewer_with_frames.html) in the output directory that:
   - Shows the same layout and styling as the existing results viewer in this repo: white container, stat cards (Frames count, Total detections), frame cards with blue left border and frame header (e.g. “Frame 0 (t=0.00s) – 14 players”).
   - For each frame, displays the frame image and overlays bounding boxes from frame_data.json. Each result in frame_data has frame_id, timestamp, and players[]. Each player has bbox: [x, y, width, height] in image pixels (top-left origin), and team_id (0, 1, or -1/unassigned).
   - Uses the same team colors as the rest of the project: red (#ff4444) for team 0, blue (#4444ff) for team 1, yellow (#ffff44) for unassigned. Boxes are drawn as overlays (position: absolute) on top of the image, scaled when the image is displayed with max-width so coordinates stay correct after the image loads (scale bbox by displayed size / natural size).

**Tech:**
- One Python script (e.g. scripts/create_viewer_with_frames.py) that: reads the video with OpenCV, extracts frames to output_dir/frames/, reads frame_data.json, and writes the HTML (inline CSS and JS, no external deps). CLI: video path, frame_data path, --output-dir, --num-frames (default 20), --title.
- HTML is self-contained: it embeds the frame data as a JS variable and loads frame images from the relative path “frames/frame_XXX.jpg”. No server-side rendering; open the HTML from a static server (e.g. same server that serves the rest of the project).

**Reference in this repo:**
- Styling: match scripts/create_results_viewer.py (container, stat-card, frame-card, frame-header).
- Bbox format: same as data/output/37a_20frames/frame_data.json (each player has bbox: [x, y, w, h]).
```

---

## One-line version (shorter prompt)

```
Add a script that: (1) extracts N frames from a video into output_dir/frames/, (2) reads frame_data.json (array of {frame_id, timestamp, players: [{bbox: [x,y,w,h], team_id}]}), (3) writes a single HTML viewer in output_dir that shows each frame image with bboxes overlaid, using the same layout and colors as create_results_viewer.py (stat cards, frame cards, team colors red/blue/yellow). CLI: video, frame_data path, --output-dir, --num-frames, --title. Scale bbox coords when image is displayed (onload: displayed size / natural size).
```

---

## After remaking: how to run

```bash
# From project root
python scripts/create_viewer_with_frames.py \
  data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
  data/output/37a_20frames/frame_data.json \
  --output-dir data/output/37a_20frames \
  --num-frames 20 \
  --title "37a Player Detection (20 frames) with bboxes"
```

Then open: `http://localhost:6851/data/output/37a_20frames/viewer_with_frames.html` (or whatever port the project’s static server uses).
