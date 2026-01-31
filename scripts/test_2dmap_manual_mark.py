#!/usr/bin/env python3
"""
Manual homography: mark up to 4 corners + center on one frame, then build 2D map for first 10 frames.
Often only 2 corners + center are marked; the other two corners are inferred from the center.
Creates test_2dmap_manual_mark.html: picture left (with player bboxes), 2D map right (warped bboxes, center, corners).
Uses 99-epoch weights for player bounding boxes when --model is provided.
Run from project root. View: http://localhost:5005/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html
"""
import argparse
import base64
import json
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import numpy as np
except ImportError:
    print("This script requires OpenCV (cv2) and numpy.")
    sys.exit(1)

# Homography helpers (for center/corners on map); always try to load
_transform_point = None
try:
    from src.analysis.homography import transform_point as _transform_point
except Exception:
    pass

# Optional: player detection (99-epoch weights) and box transform
_detector = None
_transform_boxes = None
_torch = None


def _load_detector_with_weights(model_path: str):
    """Load RF-DETR detector with given checkpoint. Returns (detector, transform_boxes) or (None, None)."""
    global _detector, _transform_boxes, _torch
    try:
        import torch as _torch_mod
        _torch = _torch_mod
        from rfdetr import RFDETRMedium
        from src.analysis.homography import transform_boxes as _tb
        _transform_boxes = _tb
        _detector = RFDETRMedium(pretrain_weights=model_path)
        return _detector, _tb
    except Exception as e:
        print(f"Optional detection unavailable: {e}")
        return None, None


def _load_detector_no_weights():
    """Load RF-DETR detector with built-in COCO weights (no checkpoint). Returns (detector, transform_boxes) or (None, None)."""
    global _detector, _transform_boxes, _torch
    try:
        import torch as _torch_mod
        _torch = _torch_mod
        from rfdetr import RFDETRMedium
        from src.analysis.homography import transform_boxes as _tb
        _transform_boxes = _tb
        _detector = RFDETRMedium()
        return _detector, _tb
    except Exception as e:
        print(f"Optional detection unavailable: {e}")
        return None, None


def _get_player_boxes_xyxy(defished_bgr, detector, threshold=0.3):
    """Run player detection on defished frame; return list of [x_min, y_min, x_max, y_max] in image coords."""
    frame_rgb = cv2.cvtColor(defished_bgr, cv2.COLOR_BGR2RGB)
    raw = detector.predict(frame_rgb, threshold=threshold)
    boxes = []
    if hasattr(raw, "class_id"):
        for i in range(len(raw.class_id)):
            cid = int(raw.class_id[i])
            # 0 = player (trained 1/2-class), 1 = COCO person
            if cid in (0, 1):
                xyxy = raw.xyxy[i]
                boxes.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
    return boxes


def _make_diagram_background(w_map, h_map, center_map_xy=None):
    """Create a green pitch diagram. If center_map_xy given, pitch center line and circle use a clamped center so the circle fits inside."""
    diagram = np.zeros((h_map, w_map, 3), dtype=np.uint8)
    diagram[:, :] = (34, 139, 34)  # pitch green
    cv2.rectangle(diagram, (0, 0), (w_map - 1, h_map - 1), (255, 255, 255), 2)
    r = min(w_map, h_map) // 8
    if center_map_xy is not None:
        cx = int(center_map_xy[0])
        cy = int(center_map_xy[1])
        cx = max(r, min(w_map - 1 - r, cx))
        cy = max(r, min(h_map - 1 - r, cy))
    else:
        cx, cy = w_map // 2, h_map // 2
    cv2.line(diagram, (cx, 0), (cx, h_map - 1), (255, 255, 255), 1)
    cv2.circle(diagram, (cx, cy), r, (255, 255, 255), 1)
    return diagram


def _draw_boxes_and_landmarks_on_map(map_frame, H, w_map, h_map, boxes_xyxy_image, center_image_xy, marked_corner_indices=None):
    """Draw warped player boxes, center point, and only the corners that were actually marked."""
    if marked_corner_indices is None:
        marked_corner_indices = [0, 1, 2, 3]
    # Warp boxes to map; draw only player positions (yellow dots), not green rectangles
    if _torch is not None and _transform_boxes is not None and boxes_xyxy_image:
        boxes_t = _torch.tensor(boxes_xyxy_image, dtype=_torch.float32)
        map_boxes = _transform_boxes(H, boxes_t).numpy()
        for box in map_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            cv2.circle(map_frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)
    # Pitch center (from manual mark) – where the user placed center
    if _transform_point is not None and center_image_xy is not None:
        cx, cy = _transform_point(H, (center_image_xy[0], center_image_xy[1]))
        cv2.circle(map_frame, (int(cx), int(cy)), 10, (0, 0, 255), 2)
        cv2.circle(map_frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    # Only the corners that were actually marked (0=TL, 1=TR, 2=BR, 3=BL)
    map_corner_pts = [(0, 0), (w_map - 1, 0), (w_map - 1, h_map - 1), (0, h_map - 1)]
    for idx in marked_corner_indices:
        if 0 <= idx < 4:
            pt = map_corner_pts[idx]
            cv2.circle(map_frame, pt, 8, (255, 0, 0), 2)
            cv2.circle(map_frame, pt, 2, (255, 0, 0), -1)

MARK_SERVER_PORT = 5006

DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "data/output/homography_calibration.json"
MARKS_PATH = PROJECT_ROOT / "data/output/2dmap_manual_mark/manual_marks.json"
NUM_FRAMES = 10
DEFAULT_MAP_W, DEFAULT_MAP_H = 600, 800

# Click order: 1=TL, 2=TR, 3=BR, 4=BL, 5=Center. 3 and 4 optional (press 's' to skip).
LABELS = ["Corner 1 (Top-Left)", "Corner 2 (Top-Right)", "Corner 3 (Bottom-Right)", "Corner 4 (Bottom-Left)", "Center"]


def crop_black_borders(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame.astype(np.float64)
    is_black = gray < black_thresh
    y1 = 0
    for y in range(h):
        if (1 - np.mean(is_black[y, :])) >= non_black_min:
            y1 = y
            break
    y2 = h
    for y in range(h - 1, -1, -1):
        if (1 - np.mean(is_black[y, :])) >= non_black_min:
            y2 = y + 1
            break
    band = is_black[y1:y2, :]
    x1 = 0
    for x in range(w):
        if (1 - np.mean(band[:, x])) >= non_black_min:
            x1 = x
            break
    x2 = w
    for x in range(w - 1, -1, -1):
        if (1 - np.mean(band[:, x])) >= non_black_min:
            x2 = x + 1
            break
    if x2 <= x1 or y2 <= y1:
        return frame
    if (x2 - x1) * (y2 - y1) < 0.1 * h * w:
        return frame
    inset = max(1, int(min(w, h) * margin_pct))
    x1 = min(x1 + inset, x2 - 1)
    y1 = min(y1 + inset, y2 - 1)
    x2 = max(x2 - inset, x1 + 1)
    y2 = max(y2 - inset, y1 + 1)
    return frame[y1:y2, x1:x2]


def crop_to_square(frame):
    h, w = frame.shape[:2]
    s = min(w, h)
    x = (w - s) // 2
    y = (h - s) // 2
    return frame[y : y + s, x : x + s]


def defish_frame(frame, k, alpha=0.0):
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return crop_black_borders(remapped)


def collect_marks_interactive(frame_display, marks_path):
    """User clicks: C1 (TL), C2 (TR), [C3 (BR) skip with 's'], [C4 (BL) skip with 's'], Center."""
    points = []  # list of {"role": "corner1"|...|"center", "image_xy": [x,y], "inferred": bool}
    step = 0
    center_im = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal step, center_im
        if event != cv2.EVENT_LBUTTONDOWN or step >= 5:
            return
        if step == 0:
            points.append({"role": "corner1", "image_xy": [int(x), int(y)], "inferred": False})
            step = 1
        elif step == 1:
            points.append({"role": "corner2", "image_xy": [int(x), int(y)], "inferred": False})
            step = 2
        elif step == 2:
            points.append({"role": "corner3", "image_xy": [int(x), int(y)], "inferred": False})
            step = 3
        elif step == 3:
            points.append({"role": "corner4", "image_xy": [int(x), int(y)], "inferred": False})
            step = 4
        elif step == 4:
            points.append({"role": "center", "image_xy": [int(x), int(y)], "inferred": False})
            center_im = (x, y)
            step = 5

    win = "Mark corners and center (1=TL, 2=TR, 3=BR, 4=BL, 5=Center; S=skip 3/4)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    print("Click in order: Corner 1 (Top-Left), Corner 2 (Top-Right), then optionally Corner 3 (Bottom-Right), Corner 4 (Bottom-Left), then Center.")
    print("Press S to skip Corner 3 and/or 4 (only 2 corners + center). Press Q to cancel.")

    while step < 5:
        disp = frame_display.copy()
        for i, p in enumerate(points):
            pt = tuple(p["image_xy"])
            if p.get("inferred") and pt == (0, 0):
                continue
            color = (0, 255, 0) if p.get("inferred") else (0, 0, 255)
            cv2.circle(disp, pt, 8, color, -1)
            cv2.putText(disp, p["role"].replace("corner", "C").replace("center", "M"), (pt[0] + 10, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                if "corner" in points[i]["role"] and "corner" in points[i + 1]["role"]:
                    p1, p2 = tuple(points[i]["image_xy"]), tuple(points[i + 1]["image_xy"])
                    if p1 != (0, 0) and p2 != (0, 0):
                        cv2.line(disp, p1, p2, (0, 255, 0), 1)
        msg = LABELS[step] if step < 5 else "Done"
        cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            return None
        if key == ord("s") or key == ord("S"):
            if step == 2:
                points.append({"role": "corner3", "image_xy": [0, 0], "inferred": True})
                step = 3
            elif step == 3:
                points.append({"role": "corner4", "image_xy": [0, 0], "inferred": True})
                step = 4

    cv2.destroyWindow(win)

    # If we have only C1, C2, Center: infer C3 and C4 (or overwrite placeholders)
    corner1 = next((p["image_xy"] for p in points if p["role"] == "corner1"), None)
    corner2 = next((p["image_xy"] for p in points if p["role"] == "corner2"), None)
    corner3_p = next((p for p in points if p["role"] == "corner3"), None)
    corner4_p = next((p for p in points if p["role"] == "corner4"), None)
    center_pt = next((p["image_xy"] for p in points if p["role"] == "center"), None)
    if not center_pt:
        return None
    cx, cy = center_pt

    if corner3_p and corner3_p.get("inferred"):
        # BR = 2*center - TL
        corner3_p["image_xy"] = [2 * cx - corner1[0], 2 * cy - corner1[1]]
    if corner4_p and corner4_p.get("inferred"):
        # BL = 2*center - TR
        corner4_p["image_xy"] = [2 * cx - corner2[0], 2 * cy - corner2[1]]

    # Build 4 corners in order TL, TR, BR, BL for homography
    c1 = corner1
    c2 = corner2
    c3 = next((p["image_xy"] for p in points if p["role"] == "corner3"), None)
    c4 = next((p["image_xy"] for p in points if p["role"] == "corner4"), None)
    if c3 is None:
        c3 = [2 * cx - c1[0], 2 * cy - c1[1]]
    if c4 is None:
        c4 = [2 * cx - c2[0], 2 * cy - c2[1]]

    src_corners = np.float32([c1, c2, c3, c4])
    marks_data = {
        "frame_index": 0,
        "points": points,
        "src_corners_order": "TL, TR, BR, BL",
        "src_corners_xy": [c1, c2, c3, c4],
    }
    marks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marks_path, "w") as f:
        json.dump(marks_data, f, indent=2)
    print(f"Saved marks to {marks_path}")

    return src_corners


def _points_to_src_corners(points):
    """From points list (with roles corner1..4, center), compute [TL, TR, BR, BL] and marks_data."""
    corner1 = next((p["image_xy"] for p in points if p["role"] == "corner1"), None)
    corner2 = next((p["image_xy"] for p in points if p["role"] == "corner2"), None)
    corner3_p = next((p for p in points if p["role"] == "corner3"), None)
    corner4_p = next((p for p in points if p["role"] == "corner4"), None)
    center_pt = next((p["image_xy"] for p in points if p["role"] == "center"), None)
    if not center_pt or not corner1 or not corner2:
        return None, None
    cx, cy = center_pt
    if corner3_p and corner3_p.get("inferred"):
        corner3_p["image_xy"] = [2 * cx - corner1[0], 2 * cy - corner1[1]]
    if corner4_p and corner4_p.get("inferred"):
        corner4_p["image_xy"] = [2 * cx - corner2[0], 2 * cy - corner2[1]]
    c3 = next((p["image_xy"] for p in points if p["role"] == "corner3"), [2 * cx - corner1[0], 2 * cy - corner1[1]])
    c4 = next((p["image_xy"] for p in points if p["role"] == "corner4"), [2 * cx - corner2[0], 2 * cy - corner2[1]])
    c1, c2 = corner1, corner2
    marks_data = {
        "frame_index": 0,
        "points": points,
        "src_corners_order": "TL, TR, BR, BL",
        "src_corners_xy": [c1, c2, c3, c4],
    }
    return np.float32([c1, c2, c3, c4]), marks_data


def collect_marks_web_fallback(frame_bgr, marks_path):
    """No GUI: save frame, serve mark UI in browser, wait for POST to save marks, then return src_corners."""
    marks_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path = marks_path.parent / "mark_frame.jpg"
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(frame_path, "wb") as f:
        f.write(buf.tobytes())
    img_b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")

    html_content = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Mark corners and center</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
    h1 { color: #4CAF50; }
    .top-bar { margin: 10px 0 8px 0; padding: 14px 18px; background: #2d2d2d; border-radius: 8px; }
    .current-mark { font-size: 18px; font-weight: bold; color: #fff; }
    .current-mark span { color: #4CAF50; }
    .skip-row { margin: 0 0 16px 0; text-align: center; }
    .skip-btn { padding: 18px 48px; font-size: 22px; font-weight: bold; cursor: pointer; background: #d32f2f; color: #fff; border: 3px solid #fff; border-radius: 8px; min-width: 220px; }
    .skip-btn:hover:not(:disabled) { background: #f44336; }
    .skip-btn:disabled { background: #8b0000; color: #ccc; border-color: #a00; cursor: not-allowed; opacity: 0.9; }
    #imgWrap { position: relative; display: inline-block; cursor: crosshair; margin: 10px 0; }
    #imgWrap img { max-width: 100%; height: auto; display: block; vertical-align: top; }
    #marksLayer { position: absolute; left: 0; top: 0; pointer-events: none; }
    .mark { position: absolute; transform: translate(-50%, -50%); display: flex; align-items: center; gap: 6px; pointer-events: auto; cursor: move; }
    .mark-dot { width: 16px; height: 16px; border-radius: 50%; background: red; border: 2px solid white; box-shadow: 0 0 4px #000; flex-shrink: 0; }
    .mark.inferred .mark-dot { background: #0f0; }
    .mark-label { font-size: 13px; font-weight: bold; color: white; text-shadow: 0 0 2px black, 0 1px 3px black; white-space: nowrap; }
    .save-section { margin-top: 24px; padding: 20px; text-align: center; background: #2d2d2d; border-radius: 8px; }
    .save-btn { padding: 18px 48px; font-size: 22px; font-weight: bold; cursor: pointer; background: #4CAF50; color: #fff; border: none; border-radius: 8px; }
    .save-btn:hover:not(:disabled) { background: #66BB6A; }
    .save-btn:disabled { background: #555; cursor: not-allowed; }
    .reset-btn { padding: 18px 32px; font-size: 18px; font-weight: bold; cursor: pointer; background: #555; color: #fff; border: 2px solid #888; border-radius: 8px; margin-left: 12px; }
    .reset-btn:hover { background: #666; }
    #status { margin-top: 10px; color: #aaa; }
    .skipped-container { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 14px; background: #2d2d2d; border-radius: 8px; margin: 10px 0; }
    .skipped-container .skipped-label { margin-right: 8px; color: #aaa; font-size: 14px; }
    .skipped-chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: #555; border-radius: 20px; cursor: grab; border: 2px solid #0f0; }
    .skipped-chip:hover { background: #666; }
    .skipped-chip .chip-dot { width: 12px; height: 12px; border-radius: 50%; background: #0f0; flex-shrink: 0; }
    .skipped-chip .chip-label { font-size: 12px; font-weight: bold; color: #fff; }
  </style>
</head>
<body>
  <h1>Mark corners and center</h1>
  <div class="top-bar">
    <div class="current-mark">Current mark: <span id="currentMarkText">Corner 1 (Top-Left)</span></div>
  </div>
  <div class="skip-row">
    <button type="button" class="skip-btn" id="skipBtn">Skip (—)</button>
  </div>
  <p style="color:#aaa;">Click on the image to place each mark. Drag red dots to adjust. Skipped marks appear below; drag them onto the frame to place.</p>
  <div class="skipped-container" id="skippedContainer">
    <span class="skipped-label">Skipped (drag onto frame to place):</span>
    <div id="skippedChips"></div>
  </div>
  <div id="imgWrap">
    <img id="img" src="" alt="Frame" />
    <div id="marksLayer"></div>
  </div>
  <div class="save-section">
    <button type="button" class="save-btn" id="save" disabled>Save positions</button>
    <button type="button" class="reset-btn" id="reset">Reset marks</button>
  </div>
  <p id="status"></p>
  <script>
    const labels = ['Corner 1 (Top-Left)', 'Corner 2 (Top-Right)', 'Corner 3 (Bottom-Right)', 'Corner 4 (Bottom-Left)', 'Center'];
    const shortLabels = ['1: TL', '2: TR', '3: BR', '4: BL', 'Center'];
    let points = [];
    let step = 0;
    let dragIdx = null;
    let dragStart = null;
    let chipDragIdx = null;
    let justDroppedChip = false;
    const img = document.getElementById('img');
    const layer = document.getElementById('marksLayer');
    const skippedChipsEl = document.getElementById('skippedChips');
    const statusEl = document.getElementById('status');
    const saveBtn = document.getElementById('save');
    const resetBtn = document.getElementById('reset');
    const skipBtn = document.getElementById('skipBtn');
    const currentMarkEl = document.getElementById('currentMarkText');

    function clientToImage(x, y) {
      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      return [
        Math.round((x - rect.left) * scaleX),
        Math.round((y - rect.top) * scaleY)
      ];
    }

    function updateCurrentMark() {
      currentMarkEl.textContent = step < 5 ? labels[step] : 'All done';
      if (step < 5) skipBtn.textContent = 'Skip ' + labels[step];
      else skipBtn.textContent = 'Skip (—)';
      if (step >= 0 && step <= 3) {
        skipBtn.disabled = false;
        skipBtn.removeAttribute('disabled');
      } else {
        skipBtn.disabled = true;
        skipBtn.setAttribute('disabled', 'disabled');
      }
    }

    img.src = 'data:image/jpeg;base64,""" + img_b64 + """';
    img.onload = function() {
      document.getElementById('imgWrap').addEventListener('click', function(e) {
        if (e.target !== img) return;
        if (justDroppedChip) return;
        if (step >= 5) return;
        const [x, y] = clientToImage(e.clientX, e.clientY);
        const role = step === 0 ? 'corner1' : step === 1 ? 'corner2' : step === 2 ? 'corner3' : step === 3 ? 'corner4' : 'center';
        points.push({ role, image_xy: [x, y], inferred: false });
        step++;
        redraw();
        updateCurrentMark();
        if (step === 5) saveBtn.disabled = false;
      });
    };

    function isInSkippedContainer(p) {
      return p.inferred && p.image_xy[0] === 0 && p.image_xy[1] === 0;
    }

    function redraw() {
      layer.innerHTML = '';
      skippedChipsEl.innerHTML = '';
      const rect = img.getBoundingClientRect();
      layer.style.width = rect.width + 'px';
      layer.style.height = rect.height + 'px';
      points.forEach((p, i) => {
        if (isInSkippedContainer(p)) {
          const chip = document.createElement('div');
          chip.className = 'skipped-chip';
          chip.dataset.idx = String(i);
          const dot = document.createElement('div');
          dot.className = 'chip-dot';
          const lbl = document.createElement('span');
          lbl.className = 'chip-label';
          lbl.textContent = shortLabels[i];
          chip.appendChild(dot);
          chip.appendChild(lbl);
          chip.addEventListener('mousedown', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            chipDragIdx = i;
          });
          skippedChipsEl.appendChild(chip);
        } else {
          const xPx = (p.image_xy[0] / img.naturalWidth) * rect.width;
          const yPx = (p.image_xy[1] / img.naturalHeight) * rect.height;
          const wrap = document.createElement('div');
          wrap.className = 'mark' + (p.inferred ? ' inferred' : '');
          wrap.style.left = xPx + 'px';
          wrap.style.top = yPx + 'px';
          wrap.dataset.idx = i;
          const dot = document.createElement('div');
          dot.className = 'mark-dot';
          const lbl = document.createElement('span');
          lbl.className = 'mark-label';
          lbl.textContent = shortLabels[i];
          wrap.appendChild(dot);
          wrap.appendChild(lbl);
          wrap.addEventListener('mousedown', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            dragIdx = parseInt(this.dataset.idx, 10);
            dragStart = { x: ev.clientX, y: ev.clientY, xy: points[dragIdx].image_xy.slice() };
          });
          layer.appendChild(wrap);
        }
      });
      statusEl.textContent = step < 5 ? 'Click on the image for: ' + labels[step] : 'All marks placed. Drag dots to adjust, then click Save positions.';
    }

    document.addEventListener('mousemove', function(e) {
      if (dragIdx === null) return;
      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const dx = (e.clientX - dragStart.x) * scaleX;
      const dy = (e.clientY - dragStart.y) * scaleY;
      points[dragIdx].image_xy[0] = Math.round(dragStart.xy[0] + dx);
      points[dragIdx].image_xy[1] = Math.round(dragStart.xy[1] + dy);
      dragStart = { x: e.clientX, y: e.clientY, xy: points[dragIdx].image_xy.slice() };
      redraw();
    });
    document.addEventListener('mouseup', function(e) {
      if (chipDragIdx !== null) {
        const r = img.getBoundingClientRect();
        if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
          const [x, y] = clientToImage(e.clientX, e.clientY);
          points[chipDragIdx].image_xy[0] = x;
          points[chipDragIdx].image_xy[1] = y;
          justDroppedChip = true;
          setTimeout(function() { justDroppedChip = false; }, 100);
          redraw();
        }
        chipDragIdx = null;
      }
      dragIdx = null;
    });
    document.addEventListener('mouseleave', function() { dragIdx = null; chipDragIdx = null; });

    skipBtn.addEventListener('click', function(ev) {
      if (ev) { ev.preventDefault(); ev.stopPropagation(); }
      if (step === 0) {
        points.push({ role: 'corner1', image_xy: [0, 0], inferred: true });
        step = 1;
        updateCurrentMark();
        redraw();
      } else if (step === 1) {
        points.push({ role: 'corner2', image_xy: [0, 0], inferred: true });
        step = 2;
        updateCurrentMark();
        redraw();
      } else if (step === 2) {
        points.push({ role: 'corner3', image_xy: [0, 0], inferred: true });
        step = 3;
        updateCurrentMark();
        redraw();
      } else if (step === 3) {
        points.push({ role: 'corner4', image_xy: [0, 0], inferred: true });
        step = 4;
        updateCurrentMark();
        redraw();
      }
      if (step === 5) saveBtn.disabled = false;
    });

    saveBtn.onclick = function() {
      const cx = points.find(p => p.role === 'center').image_xy[0];
      const cy = points.find(p => p.role === 'center').image_xy[1];
      let c1 = points.find(p => p.role === 'corner1').image_xy;
      let c2 = points.find(p => p.role === 'corner2').image_xy;
      let c3 = points.find(p => p.role === 'corner3').image_xy;
      let c4 = points.find(p => p.role === 'corner4').image_xy;
      if (points.find(p => p.role === 'corner1').inferred) c1 = [2*cx - c2[0], 2*cy - c2[1]];
      if (points.find(p => p.role === 'corner2').inferred) c2 = [2*cx - c1[0], 2*cy - c1[1]];
      if (points.find(p => p.role === 'corner3').inferred) c3 = [2*cx - c1[0], 2*cy - c1[1]];
      if (points.find(p => p.role === 'corner4').inferred) c4 = [2*cx - c2[0], 2*cy - c2[1]];
      const marks = { frame_index: 0, points, src_corners_order: 'TL, TR, BR, BL', src_corners_xy: [c1, c2, c3, c4] };
      fetch('/save_marks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(marks) })
        .then(r => r.json()).then(d => { statusEl.textContent = d.ok ? 'Saved. You can close this tab.' : (d.error || 'Save failed'); })
        .catch(e => { statusEl.textContent = 'Error: ' + e.message; });
    };

    resetBtn.addEventListener('click', function() {
      points = [];
      step = 0;
      updateCurrentMark();
      redraw();
      saveBtn.disabled = true;
      statusEl.textContent = 'Marks reset. Click on the image for: ' + labels[step];
    });

    updateCurrentMark();
  </script>
</body>
</html>"""

    mark_ui_path = marks_path.parent / "mark_ui.html"
    mark_ui_path.write_text(html_content, encoding="utf-8")

    marks_saved = threading.Event()
    marks_file_path = str(marks_path)

    class MarkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/mark_ui.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html_content.encode("utf-8"))
            elif self.path == "/mark_frame.jpg":
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/save_marks":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    with open(marks_file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}')
                    marks_saved.set()
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", MARK_SERVER_PORT), MarkHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("No display available. Open in your browser to mark:")
    print(f"  http://localhost:{MARK_SERVER_PORT}/mark_ui.html")
    print("Mark: 1=TL, 2=TR, (3=BR or Skip), (4=BL or Skip), 5=Center. Then click Save marks.")
    print("Waiting for you to save marks...")
    marks_saved.wait(timeout=600)
    server.shutdown()
    time.sleep(0.3)

    if not marks_path.exists():
        print("Marks were not saved.")
        return None
    with open(marks_path, "r") as f:
        data = json.load(f)
    src_corners = np.float32(data.get("src_corners_xy", []))
    if len(src_corners) != 4:
        return None
    print(f"Saved marks to {marks_path}")
    return src_corners


def _marked_corner_indices_from_points(points):
    """From points list (roles corner1..4), return indices [0..3] for corners that were actually placed (not skipped)."""
    if not points:
        return []
    role_to_idx = {"corner1": 0, "corner2": 1, "corner3": 2, "corner4": 3}
    indices = []
    for p in points:
        role = p.get("role")
        if role not in role_to_idx:
            continue
        xy = p.get("image_xy", [0, 0])
        if p.get("inferred") and xy == [0, 0]:
            continue
        if xy == [0, 0]:
            continue
        indices.append(role_to_idx[role])
    return sorted(indices)


def _count_unique_corners(src_corners, dist_thresh=5.0):
    """Return number of unique points in src_corners (points within dist_thresh are same)."""
    if src_corners is None or len(src_corners) < 2:
        return 0
    pts = np.asarray(src_corners, dtype=np.float64)
    n = len(pts)
    unique = 1
    for i in range(1, n):
        same = False
        for j in range(i):
            if np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]) < dist_thresh:
                same = True
                break
        if not same:
            unique += 1
    return unique


def _quad_area(src_corners):
    """Area of quadrilateral (TL, TR, BR, BL) via cross product. Non-negative if ordered CCW."""
    if src_corners is None or len(src_corners) != 4:
        return 0.0
    p = np.asarray(src_corners, dtype=np.float64)
    return 0.5 * abs(
        (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
        - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        + (p[2, 0] - p[0, 0]) * (p[3, 1] - p[0, 1])
        - (p[3, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
    )


def _get_two_distinct_corners(src_corners, dist_thresh=5.0):
    """From 4 corners (possibly degenerate), return (pt_a, pt_b) or (None, None) if not exactly 2 distinct."""
    if src_corners is None or len(src_corners) < 2:
        return None, None
    pts = np.asarray(src_corners, dtype=np.float64)
    distinct = [pts[0]]
    for i in range(1, len(pts)):
        if all(np.hypot(pts[i, 0] - d[0], pts[i, 1] - d[1]) >= dist_thresh for d in distinct):
            distinct.append(pts[i])
    if len(distinct) != 2:
        return None, None
    return tuple(distinct[0]), tuple(distinct[1])


def _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, image_shape):
    """
    Build 4-point quad [TL, TR, BR, BL] from two distinct corners using the image boundary.
    Does not infer off-frame corners: the bottom edge is the bottom of the frame.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    ax, ay = float(pt_a[0]), float(pt_a[1])
    bx, by = float(pt_b[0]), float(pt_b[1])
    y_bottom = h - 1
    tl = np.array([ax, ay], dtype=np.float32)
    tr = np.array([bx, by], dtype=np.float32)
    br = np.array([bx, y_bottom], dtype=np.float32)
    bl = np.array([ax, y_bottom], dtype=np.float32)
    return np.float32([tl, tr, br, bl])


def homography_from_marks(src_corners, w_map, h_map):
    """dst: TL, TR, BR, BL in map pixels."""
    dst = np.float32([[0, 0], [w_map, 0], [w_map, h_map], [0, h_map]])
    H = cv2.getPerspectiveTransform(src_corners, dst)
    return H


def save_frame_image(frame_jpg_bytes, output_dir, frame_id, suffix):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    filename = f"frame_{frame_id}_{suffix}.jpg"
    filepath = frames_dir / filename
    with open(filepath, "wb") as f:
        f.write(frame_jpg_bytes)
    return f"frames/{filename}"


def main():
    parser = argparse.ArgumentParser(
        description="Mark up to 4 corners + center on one frame; build 2D map for first 10 frames and HTML."
    )
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--calibration", "-c", type=str, default=str(DEFAULT_CALIB), help="Optional: homography_calibration.json for k, alpha, map_size")
    parser.add_argument("-n", "--num-frames", type=int, default=NUM_FRAMES, help=f"Number of frames (default {NUM_FRAMES})")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--mark", action="store_true", help="Force re-mark even if manual_marks.json exists")
    parser.add_argument("--web", action="store_true", help="Use web marking UI only (skip OpenCV window)")
    parser.add_argument("--use-saved", action="store_true", help="Use saved manual_marks.json only (no GUI)")
    default_model = str(PROJECT_ROOT / "models" / "checkpoints" / "checkpoint_epoch_99_lightweight.pth")
    parser.add_argument("--model", "-m", type=str, default=default_model, help=f"Player detection weights (99-epoch); default: {default_model}")
    args = parser.parse_args()

    k_value = -0.32
    alpha = 0.0
    w_map, h_map = DEFAULT_MAP_W, DEFAULT_MAP_H
    calib_path = Path(args.calibration)
    if calib_path.exists():
        with open(calib_path, "r") as f:
            calib = json.load(f)
        k_value = calib.get("k_value", k_value)
        alpha = calib.get("alpha", alpha)
        if "map_size" in calib:
            w_map, h_map = calib["map_size"][0], calib["map_size"][1]

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else PROJECT_ROOT / "data/output/2dmap_manual_mark/test_2dmap_manual_mark.html"
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame for marking
    ret, frame0 = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)
    frame0 = cv2.resize(frame0, (0, 0), fx=0.5, fy=0.5)
    defished0 = defish_frame(frame0, k_value, alpha=alpha)
    defished0 = crop_to_square(defished0)

    src_corners = None
    if args.use_saved and MARKS_PATH.exists():
        with open(MARKS_PATH, "r") as f:
            marks_data = json.load(f)
        src_corners = np.float32(marks_data.get("src_corners_xy", []))
        if len(src_corners) != 4:
            src_corners = None
        else:
            print(f"Using saved marks from {MARKS_PATH}")
    if src_corners is None and (args.mark or not MARKS_PATH.exists()):
        if args.web:
            src_corners = collect_marks_web_fallback(defished0, MARKS_PATH)
        else:
            try:
                src_corners = collect_marks_interactive(defished0, MARKS_PATH)
            except cv2.error:
                src_corners = collect_marks_web_fallback(defished0, MARKS_PATH)
        if src_corners is None:
            print("Marking cancelled or failed.")
            sys.exit(1)
    elif src_corners is None:
        with open(MARKS_PATH, "r") as f:
            marks_data = json.load(f)
        src_corners = np.float32(marks_data.get("src_corners_xy", []))
        if len(src_corners) != 4:
            print("Run with --mark to mark corners and center on the first frame.")
            sys.exit(1)

    # Center point and which corners were actually placed (from saved points)
    marks_data_path = MARKS_PATH if MARKS_PATH.exists() else None
    center_image_xy = None
    points_from_file = []
    if marks_data_path:
        with open(marks_data_path, "r") as f:
            _md = json.load(f)
        points_from_file = _md.get("points", [])
        for p in points_from_file:
            if p.get("role") == "center":
                center_image_xy = p.get("image_xy")
                break
    marked_corner_indices = _marked_corner_indices_from_points(points_from_file) if points_from_file else [0, 1, 2, 3]

    # When only 1–2 distinct corners (others skipped), use frame boundary; do not infer off-frame corners
    if _count_unique_corners(src_corners) < 3 or _quad_area(src_corners) < 100.0:
        pt_a, pt_b = _get_two_distinct_corners(src_corners)
        if pt_a is not None and pt_b is not None:
            built = _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, defished0.shape)
            if built is not None:
                src_corners = built
                if not points_from_file:
                    marked_corner_indices = [0, 1]
                print("Using frame boundary for quad (corners not in frame were not inferred).")

    H = homography_from_marks(src_corners, w_map, h_map)
    # Center position on diagram (from user's mark) for aligning pitch center line/circle
    center_map_xy = None
    if _transform_point is not None and center_image_xy is not None:
        center_map_xy = _transform_point(H, (center_image_xy[0], center_image_xy[1]))

    # Optional: load player detector (99-epoch, then alternate, then COCO fallback)
    detector = None
    model_path = Path(args.model)
    alternate_model = PROJECT_ROOT / "models" / "checkpoint_best_total_after_100_epochs.pth"
    if model_path.exists():
        detector, _ = _load_detector_with_weights(str(model_path))
        if detector:
            print(f"Using player detection: {model_path}")
    if detector is None and alternate_model.exists():
        detector, _ = _load_detector_with_weights(str(alternate_model))
        if detector:
            print(f"Using player detection: {alternate_model}")
    if detector is None:
        detector, _ = _load_detector_no_weights()
        if detector:
            print("Using player detection: RF-DETR COCO weights (no checkpoint).")
        else:
            print("Skipping player bboxes (no model or detection unavailable).")
            print("  Install rfdetr to enable bounding boxes: pip install rfdetr")

    n_frames = min(args.num_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or args.num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector_note = "Player bounding boxes: shown." if detector else "Player bounding boxes: not available (install rfdetr: pip install rfdetr)."

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>2D map check – manual marks</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #4CAF50; }
        .info { background: #2d2d2d; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #2d2d2d; }
        th { background: #1a1a1a; padding: 10px; text-align: left; color: #4CAF50; }
        td { padding: 10px; border-top: 1px solid #444; }
        img { max-width: 100%; height: auto; border: 2px solid #444; border-radius: 5px; }
        .col-picture { width: 50%; }
        .col-map { width: 50%; }
    </style>
</head>
<body>
    <h1>2D map check – manual marks</h1>
    <div class="info">
        <strong>First """,
        str(n_frames),
        """ frames.</strong> Left: frame with player bounding boxes only. Right: 2D diagram (pitch outline + player positions on the plane).<br>
        <strong>Video:</strong> """,
        video_path.name,
        """<br>
        <strong>""",
        detector_note,
        """</strong>
    </div>
    <table>
        <tr>
            <th>Frame #</th>
            <th class="col-picture">Frame (bboxes + marks)</th>
            <th class="col-map">2D diagram (players on pitch)</th>
        </tr>""",
    ]

    for frame_num in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        defished = defish_frame(frame, k_value, alpha=alpha)
        defished = crop_to_square(defished)

        boxes_xyxy = []
        if detector:
            boxes_xyxy = _get_player_boxes_xyxy(defished, detector, threshold=0.3)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(defished, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                for pt in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                    cv2.circle(defished, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cv2.circle(defished, (int(cx), int(cy)), 4, (255, 255, 0), -1)

        # Landmarks on frame: pitch center (red) and marked corners (blue)
        if center_image_xy is not None:
            cx, cy = int(center_image_xy[0]), int(center_image_xy[1])
            cv2.circle(defished, (cx, cy), 10, (0, 0, 255), 2)
            cv2.circle(defished, (cx, cy), 2, (0, 0, 255), -1)
        for idx in marked_corner_indices:
            if 0 <= idx < len(src_corners):
                px, py = int(src_corners[idx][0]), int(src_corners[idx][1])
                cv2.circle(defished, (px, py), 8, (255, 0, 0), 2)
                cv2.circle(defished, (px, py), 2, (255, 0, 0), -1)

        # Right: 2D diagram (pitch aligned to user's center + corners; player positions)
        map_frame = _make_diagram_background(w_map, h_map, center_map_xy)
        _draw_boxes_and_landmarks_on_map(map_frame, H, w_map, h_map, boxes_xyxy, center_image_xy, marked_corner_indices)

        _, buf_pic = cv2.imencode(".jpg", defished, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_map = cv2.imencode(".jpg", map_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        pic_path = save_frame_image(buf_pic.tobytes(), out_dir, frame_num, "picture")
        map_path = save_frame_image(buf_map.tobytes(), out_dir, frame_num, "map")

        html_parts.append(
            f"""
        <tr>
            <td><strong>Frame {frame_num}</strong></td>
            <td class="col-picture"><img src="{pic_path}" alt="Picture" loading="lazy" /></td>
            <td class="col-map"><img src="{map_path}" alt="2D map" loading="lazy" /></td>
        </tr>"""
        )

    html_parts.append(
        """
    </table>
</body>
</html>"""
    )

    out_path.write_text("".join(html_parts), encoding="utf-8")
    cap.release()

    print(f"Created: {out_path}")
    print(f"  Open: http://localhost:5005/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html")


if __name__ == "__main__":
    main()
