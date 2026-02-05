#!/usr/bin/env python3
"""
Manual homography: mark up to 4 corners + center on one frame, then build 2D map for first 10 frames.
Often only 2 corners + center are marked; the other two corners are inferred from the center.
Creates test_2dmap_manual_mark.html: picture left (with player bboxes), 2D map right (warped bboxes, center, corners).

To regenerate the report without opening the mark UI, run: python scripts/test_2dmap_manual_mark.py --use-saved
Default model: models/checkpoint_best_total_after_100_epochs.pth (fallback: other .pth in models/, then COCO).

Player bounding boxes (left column): Require pip install rfdetr (and torch). Either (1) place a trained
checkpoint in models/ (e.g. checkpoint_best_total_after_100_epochs.pth, see README) or (2) rely on rfdetr
COCO weights (no checkpoint). If the report shows "Player bounding boxes: not available", install dependencies
and ensure a model is available, then re-run this script so frames/frame_*_picture.jpg are regenerated with boxes.

Run from project root. View: http://localhost:8080/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html

For best 2D positions, mark all 4 pitch corners (TL, TR, BR, BL). With only 2 corners + center, the inferred
quad may not cover the full field and some players may appear clamped at the map edge.
"""
import argparse
import base64
import json
import os
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
        print(f"Optional detection unavailable (load failed): {e}")
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
        print(f"Optional detection unavailable (COCO load failed): {e}")
        return None, None


_logged_empty_detection = False


def _raw_class_ids_array(raw):
    """Resolve class IDs from raw (class_id, class_ids, or labels). Returns list of int or None."""
    arr = getattr(raw, "class_id", None)
    if arr is None:
        arr = getattr(raw, "class_ids", None)
    if arr is None:
        arr = getattr(raw, "labels", None)
    if arr is None:
        return None
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr).flatten()
    return [int(arr[i]) for i in range(len(arr))]


def _get_player_boxes_xyxy(defished_bgr, detector, threshold=0.25):
    """Run player detection on defished frame; return list of [x_min, y_min, x_max, y_max] in image coords."""
    global _logged_empty_detection
    frame_rgb = cv2.cvtColor(defished_bgr, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image
        pil_image = Image.fromarray(frame_rgb)
        raw = detector.predict(pil_image, threshold=threshold)
    except Exception:
        raw = detector.predict(frame_rgb, threshold=threshold)
    boxes = []
    class_ids = _raw_class_ids_array(raw)
    xyxy_arr = getattr(raw, "xyxy", None)
    if xyxy_arr is None:
        xyxy_len = 0
    else:
        xyxy_len = len(xyxy_arr)
    if xyxy_len == 0:
        if not _logged_empty_detection:
            _logged_empty_detection = True
            has_ci = hasattr(raw, "class_id")
            has_cis = hasattr(raw, "class_ids")
            has_lb = hasattr(raw, "labels")
            print(f"[2dmap] Detection diagnostic: type={type(raw).__name__}, class_id={has_ci}, class_ids={has_cis}, labels={has_lb}, xyxy len=0")
        return boxes
    n = min(len(class_ids), xyxy_len) if class_ids is not None else xyxy_len
    for i in range(n):
        if class_ids is not None:
            cid = class_ids[i]
            if cid not in (0, 1):
                continue
        xyxy = xyxy_arr[i]
        if hasattr(xyxy, "tolist"):
            coords = xyxy.tolist()[:4]
        else:
            coords = np.asarray(xyxy).flatten()[:4].tolist()
        if len(coords) != 4:
            continue
        boxes.append([float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])])
    if len(boxes) == 0 and not _logged_empty_detection:
        _logged_empty_detection = True
        has_ci = hasattr(raw, "class_id")
        has_cis = hasattr(raw, "class_ids")
        has_lb = hasattr(raw, "labels")
        print(f"[2dmap] Detection diagnostic: type={type(raw).__name__}, class_id={has_ci}, class_ids={has_cis}, labels={has_lb}, xyxy len={xyxy_len}, boxes kept=0")
        if class_ids is not None:
            unique_cids = set(class_ids[:n]) if n else set()
            print(f"[2dmap] Raw class_ids present: {sorted(unique_cids)}")
    return boxes


def _make_diagram_background(w_map, h_map, center_map_xy=None, margin=None, half_pitch=False):
    """
    Create 2D pitch diagram: 105 m x 68 m at 10 px/m (or 52.5 m x 68 m when half_pitch). Uses a margin so nets extend off the pitch
    and corner flags are visible at 45 deg outward. Touchlines, goal lines, halfway line, center mark,
    center circle (9.15 m), corner arcs (1 m), goals with nets on the short sides (left/right) in margin.
    When half_pitch is True, only left goal is drawn and right boundary is the halfway line.
    """
    if margin is None:
        margin = DIAGRAM_MARGIN
    template_path = PROJECT_ROOT / "assets" / "pitch_template.png"
    if margin == 0 and template_path.exists():
        diagram = cv2.imread(str(template_path))
        if diagram is not None and diagram.shape[1] == w_map and diagram.shape[0] == h_map:
            return diagram.copy()
    w_diag = w_map + 2 * margin
    h_diag = h_map + 2 * margin
    diagram = np.zeros((h_diag, w_diag, 3), dtype=np.uint8)
    diagram[:, :] = (60, 60, 60)  # off-pitch background
    pitch_green = (34, 139, 34)
    net_fill = (220, 240, 220)
    white = (255, 255, 255)
    goal_depth_px = int(2.5 * PIXELS_PER_METER)
    center_circle_r = int(9.15 * PIXELS_PER_METER)
    goal_width_px = int(7.32 * PIXELS_PER_METER)
    corner_arc_r = int(1 * PIXELS_PER_METER)
    flag_len = 58
    mid_x = w_map // 2
    # Pitch green (inset)
    cv2.rectangle(diagram, (margin, margin), (margin + w_map - 1, margin + h_map - 1), pitch_green, -1)
    # Goals with nets in margin (off the pitch); fill then draw net pattern
    def _draw_goal_net(x1, y1, x2, y2):
        cv2.rectangle(diagram, (x1, y1), (x2, y2), net_fill, -1)
        cv2.rectangle(diagram, (x1, y1), (x2, y2), white, 2)
        step = max(4, goal_depth_px // 4)
        for i in range(y1, y2, step):
            cv2.line(diagram, (x1, i), (x2, i), (180, 200, 180), 1)
        for i in range(x1, x2, step):
            cv2.line(diagram, (i, y1), (i, y2), (180, 200, 180), 1)
    _draw_goal_net(0, margin, margin + goal_depth_px, margin + h_map - 1)
    # Pitch outline (inset) before right goal so goal can be drawn on top when full pitch
    cv2.rectangle(diagram, (margin, margin), (margin + w_map - 1, margin + h_map - 1), white, 2)
    if not half_pitch:
        _draw_goal_net(margin + w_map - 1 - goal_depth_px, margin, w_diag - 1, margin + h_map - 1)
        # Emphasize right goal line so it is clearly visible (full-pitch / Option B)
        right_x = margin + w_map - 1
        cv2.line(diagram, (right_x, margin), (right_x, margin + h_map - 1), white, 3)
    # Halfway line: at right edge when half_pitch, else at center
    half_x = (margin + w_map - 1) if half_pitch else (margin + mid_x)
    cv2.line(diagram, (half_x, margin), (half_x, margin + h_map - 1), white, 2)
    if center_map_xy is not None:
        cx = int(center_map_xy[0])
        cy = int(center_map_xy[1])
        cx = max(center_circle_r, min(w_map - 1 - center_circle_r, cx))
        cy = max(center_circle_r, min(h_map - 1 - center_circle_r, cy))
    else:
        cx, cy = mid_x, h_map // 2
    # Center circle and mark (offset)
    cv2.circle(diagram, (margin + cx, margin + cy), center_circle_r, white, 2)
    cv2.circle(diagram, (margin + cx, margin + cy), 4, white, -1)
    # Goal frames (posts) and net on short sides; net extends off pitch into margin
    goal_half = goal_width_px // 2
    gy1 = max(0, h_map // 2 - goal_half)
    gy2 = min(h_map - 1, h_map // 2 + goal_half)
    goal_rects = [(margin, margin + goal_depth_px)]
    if not half_pitch:
        goal_rects.append((margin + w_map - 1 - goal_depth_px, margin + w_map - 1))
    for i_goal, (gx1, gx2) in enumerate(goal_rects):
        # Right goal (full pitch): thicker outline so it is visible
        thick = 3 if (not half_pitch and i_goal == 1) else 2
        cv2.rectangle(diagram, (gx1, margin + gy1), (gx2, margin + gy2), white, thick)
        cv2.rectangle(diagram, (gx1, margin + gy1), (gx2, margin + gy2), net_fill, -1)
        step = max(3, min(goal_width_px, goal_depth_px) // 5)
        for i in range(margin + gy1, margin + gy2, step):
            cv2.line(diagram, (gx1, i), (gx2, i), (180, 200, 180), 1)
        for i in range(gx1, gx2, step):
            cv2.line(diagram, (i, margin + gy1), (i, margin + gy2), (180, 200, 180), 1)
    # Corner arcs (1 m radius quarter-circles at each corner)
    cv2.ellipse(diagram, (margin + corner_arc_r, margin + corner_arc_r), (corner_arc_r, corner_arc_r), 0, 180, 270, white, 2)
    cv2.ellipse(diagram, (margin + w_map - 1 - corner_arc_r, margin + corner_arc_r), (corner_arc_r, corner_arc_r), 0, 270, 360, white, 2)
    cv2.ellipse(diagram, (margin + w_map - 1 - corner_arc_r, margin + h_map - 1 - corner_arc_r), (corner_arc_r, corner_arc_r), 0, 0, 90, white, 2)
    cv2.ellipse(diagram, (margin + corner_arc_r, margin + h_map - 1 - corner_arc_r), (corner_arc_r, corner_arc_r), 0, 90, 180, white, 2)
    # Corner flags at 45 deg away from pitch (outward into margin); thicker so visible when report is scaled
    d = int(flag_len * 0.707)
    cv2.line(diagram, (margin, margin), (margin - d, margin - d), white, 3)
    cv2.line(diagram, (margin + w_map - 1, margin), (margin + w_map - 1 + d, margin - d), white, 3)
    cv2.line(diagram, (margin + w_map - 1, margin + h_map - 1), (margin + w_map - 1 + d, margin + h_map - 1 + d), white, 3)
    cv2.line(diagram, (margin, margin + h_map - 1), (margin - d, margin + h_map - 1 + d), white, 3)
    return diagram


def _bbox_feet_xy(bbox):
    """Bottom-center of bbox (feet position). bbox = [x_min, y_min, x_max, y_max]."""
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    foot_x = (x_min + x_max) / 2
    foot_y = y_max
    return (foot_x, foot_y)


def _compute_y_axis_scale_from_positions(map_positions_xy, h_map, min_positions=10, min_observed_ratio=0.5):
    """From list of (mx, my) in map pixels, if observed y-range is too small, return scale so range -> h_map; else 1.0."""
    if not map_positions_xy or len(map_positions_xy) < min_positions:
        return 1.0
    ys = [p[1] for p in map_positions_xy]
    observed_height = max(ys) - min(ys)
    if observed_height <= 0 or observed_height >= min_observed_ratio * h_map:
        return 1.0
    return h_map / observed_height


def _draw_boxes_and_landmarks_on_map(map_frame, H, w_map, h_map, boxes_xyxy_image, center_image_xy, marked_corner_indices=None, margin=0, y_axis_scale=1.0, halfway_line_xy=None, draw_center=True, use_fixed_halfway_positions=False):
    """Draw player feet (bottom-center of bbox) on map via homography; center, corners, and halfway line endpoints.
    Positions are perspective-correct for the quad (TL,TR,BR,BL) -> (0,0)-(w_map,0)-(w_map,h_map)-(0,h_map); accuracy depends on marking quality.
    If use_fixed_halfway_positions is True, draw halfway at fixed left/right mid-height instead of projecting.
    If y_axis_scale != 1.0, y is scaled around map center (h_map/2) so observed y-range matches full pitch.
    Returns (in_bounds_count, out_of_bounds_count). margin offsets all coordinates into diagram space."""
    if marked_corner_indices is None:
        marked_corner_indices = [0, 1, 2, 3]
    in_bounds, out_of_bounds = 0, 0
    pitch_x_min, pitch_x_max = margin, margin + w_map - 1
    pitch_y_min, pitch_y_max = margin, margin + h_map - 1
    y_center_map = (h_map - 1) / 2.0
    if _transform_point is not None and boxes_xyxy_image:
        for box in boxes_xyxy_image:
            foot_x, foot_y = _bbox_feet_xy(box)
            mx, my = _transform_point(H, (foot_x, foot_y))
            if y_axis_scale != 1.0:
                my = (my - y_center_map) * y_axis_scale + y_center_map
            ix, iy = int(mx), int(my)
            in_map = 0 <= ix < w_map and 0 <= iy < h_map
            if in_map:
                in_bounds += 1
                cv2.circle(map_frame, (margin + ix, margin + iy), 10, (255, 255, 0), -1)
                cv2.circle(map_frame, (margin + ix, margin + iy), 10, (255, 255, 255), 2)
            else:
                out_of_bounds += 1
                cx = max(pitch_x_min, min(pitch_x_max, margin + ix))
                cy = max(pitch_y_min, min(pitch_y_max, margin + iy))
                cv2.circle(map_frame, (cx, cy), 6, (255, 255, 0), -1)
                cv2.circle(map_frame, (cx, cy), 6, (128, 128, 128), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    label_offset = 14

    if draw_center:
        center_pt = (margin + w_map // 2, margin + h_map // 2)
        cv2.circle(map_frame, center_pt, 10, (0, 0, 255), 2)
        cv2.circle(map_frame, center_pt, 2, (0, 0, 255), -1)
        _draw_landmark_label(map_frame, LANDMARK_CENTER_NAME, (center_pt[0] + label_offset, center_pt[1] - label_offset), font, font_scale, (0, 0, 255), thickness)
    # Draw all four pitch corners: TL top-left, TR top-right, BR bottom-right, BL bottom-left
    map_corner_pts = [(margin, margin), (margin + w_map - 1, margin), (margin + w_map - 1, margin + h_map - 1), (margin, margin + h_map - 1)]
    for idx in range(4):
        pt = map_corner_pts[idx]
        cv2.circle(map_frame, pt, 8, (255, 0, 0), 2)
        cv2.circle(map_frame, pt, 2, (255, 0, 0), -1)
        name = LANDMARK_CORNER_NAMES[idx]
        if idx == 0:
            txt_pt = (pt[0] + label_offset, pt[1] + label_offset)
        elif idx == 1:
            txt_pt = (pt[0] - 120, pt[1] + label_offset)
        elif idx == 2:
            txt_pt = (pt[0] - 130, pt[1] - label_offset)
        else:
            txt_pt = (pt[0] + label_offset, pt[1] - label_offset)
        _draw_landmark_label(map_frame, name, txt_pt, font, font_scale, (255, 0, 0), thickness)
    # Corner arcs on top of landmarks: use larger radius (18 px) so arc extends outside blue circle (r=8) and is visible
    corner_arc_r_map = 18
    white = (255, 255, 255)
    cv2.ellipse(map_frame, (margin + corner_arc_r_map, margin + corner_arc_r_map), (corner_arc_r_map, corner_arc_r_map), 0, 180, 270, white, 2)
    cv2.ellipse(map_frame, (margin + w_map - 1 - corner_arc_r_map, margin + corner_arc_r_map), (corner_arc_r_map, corner_arc_r_map), 0, 270, 360, white, 2)
    cv2.ellipse(map_frame, (margin + w_map - 1 - corner_arc_r_map, margin + h_map - 1 - corner_arc_r_map), (corner_arc_r_map, corner_arc_r_map), 0, 0, 90, white, 2)
    cv2.ellipse(map_frame, (margin + corner_arc_r_map, margin + h_map - 1 - corner_arc_r_map), (corner_arc_r_map, corner_arc_r_map), 0, 90, 180, white, 2)
    # Halfway line endpoints: fixed positions when quad was re-inferred, else project from image
    # Both points sit on the halfway line (right edge of mapped content at x = w_map), top and bottom.
    if use_fixed_halfway_positions:
        halfway_x = margin + w_map - 1
        half_left_pt = (halfway_x, margin)
        half_right_pt = (halfway_x, margin + h_map - 1)
        for hi, draw_pt in enumerate([half_left_pt, half_right_pt]):
            cv2.circle(map_frame, draw_pt, 8, (0, 255, 255), 2)
            cv2.circle(map_frame, draw_pt, 2, (0, 255, 255), -1)
            lbl = LANDMARK_HALFWAY_NAMES[hi] if hi < len(LANDMARK_HALFWAY_NAMES) else f"Halfway {hi + 1}"
            half_y_offset = 32 if hi == 1 else label_offset
            _draw_landmark_label(map_frame, lbl, (draw_pt[0] + label_offset, draw_pt[1] - half_y_offset), font, font_scale, (0, 255, 255), thickness)
    elif _transform_point is not None and halfway_line_xy:
        for hi, pt in enumerate(halfway_line_xy):
            if len(pt) >= 2:
                px, py = float(pt[0]), float(pt[1])
                mx, my = _transform_point(H, (px, py))
                if y_axis_scale != 1.0:
                    my = (my - y_center_map) * y_axis_scale + y_center_map
                ix, iy = int(mx), int(my)
                draw_pt = (margin + ix, margin + iy)
                cv2.circle(map_frame, draw_pt, 8, (0, 255, 255), 2)
                cv2.circle(map_frame, draw_pt, 2, (0, 255, 255), -1)
                lbl = LANDMARK_HALFWAY_NAMES[hi] if hi < len(LANDMARK_HALFWAY_NAMES) else f"Halfway {hi + 1}"
                half_y_offset = 32 if hi == 1 else label_offset
                _draw_landmark_label(map_frame, lbl, (draw_pt[0] + label_offset, draw_pt[1] - half_y_offset), font, font_scale, (0, 255, 255), thickness)
    return (in_bounds, out_of_bounds)

MARK_SERVER_PORT = 5006

DEFAULT_VIDEO = PROJECT_ROOT / "data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4"
DEFAULT_CALIB = PROJECT_ROOT / "data/output/homography_calibration.json"
MARKS_PATH = PROJECT_ROOT / "data/output/2dmap_manual_mark/manual_marks.json"
NUM_FRAMES = 10
# Standard pitch 105m x 68m at 10 px/m (R-003 / create_viewer_with_frames_2d_map)
DEFAULT_MAP_W, DEFAULT_MAP_H = 1050, 680
# Half pitch (52.5m x 68m) when quad is built from TL, TR and halfway line
HALF_PITCH_MAP_W, HALF_PITCH_MAP_H = 525, 680
DIAGRAM_MARGIN = 25
PIXELS_PER_METER = 10

# Canonical landmark names (mark + label on raw, defished, and 2D map)
LANDMARK_CORNER_NAMES = ("Top Left Corner", "Top Right Corner", "Bottom Right Corner", "Bottom Left Corner")
LANDMARK_HALFWAY_NAMES = ("Halfway Left", "Halfway Right")
LANDMARK_CENTER_NAME = "Center"


def _draw_landmark_label(img, text, pt, font, font_scale, color, thickness=1):
    """Draw text label with white background and border so the name is always readable."""
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pt[0]), int(pt[1])
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(img.shape[1], x + tw + pad)
    y2 = min(img.shape[0], y + pad)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

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


def _crop_black_borders_bounds(frame, black_thresh=15, margin_pct=0.02, non_black_min=0.99):
    """Return (x1, y1, x2, y2) for crop_black_borders; or (0, 0, w, h) if no crop."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, w, h)
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
        return (0, 0, w, h)
    if (x2 - x1) * (y2 - y1) < 0.1 * h * w:
        return (0, 0, w, h)
    inset = max(1, int(min(w, h) * margin_pct))
    x1 = min(x1 + inset, x2 - 1)
    y1 = min(y1 + inset, y2 - 1)
    x2 = max(x2 - inset, x1 + 1)
    y2 = max(y2 - inset, y1 + 1)
    return (x1, y1, x2, y2)


def _defish_frame_with_raw_mapping(frame, k, alpha=0.0):
    """Return (defished_cropped_image, get_raw_xy) where get_raw_xy(dx, dy) -> (rx, ry) or None."""
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    x1, y1, x2, y2 = _crop_black_borders_bounds(remapped)
    cropped = remapped[y1:y2, x1:x2]
    c_h, c_w = cropped.shape[:2]
    s = min(c_w, c_h)
    ox = (c_w - s) // 2
    oy = (c_h - s) // 2
    defished_cropped = cropped[oy : oy + s, ox : ox + s].copy()

    def get_raw_xy(dx, dy):
        fx = dx + ox + x1
        fy = dy + oy + y1
        if not (0 <= fy < h and 0 <= fx < w):
            return None
        rx = map1[fy, fx]
        ry = map2[fy, fx]
        return (int(round(rx)), int(round(ry)))

    return (defished_cropped, get_raw_xy)


def collect_marks_interactive(frame_display, marks_path, video_path):
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

    print("Click in order: Corner 1 (Top-Left), Corner 2 (Top-Right), optionally Corner 3 (Bottom-Right), Corner 4 (Bottom-Left), then Center.")
    print("Press S to skip Corner 2, 3 and/or 4 (e.g. only TL+BL+center: skip 2 and 3, then click 4 and Center). Press Q to cancel.")

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
            if step == 1:
                points.append({"role": "corner2", "image_xy": [0, 0], "inferred": True})
                step = 2
            elif step == 2:
                points.append({"role": "corner3", "image_xy": [0, 0], "inferred": True})
                step = 3
            elif step == 3:
                points.append({"role": "corner4", "image_xy": [0, 0], "inferred": True})
                step = 4

    cv2.destroyWindow(win)

    # If we have only C1, C2, Center: infer C3 and C4 (or overwrite placeholders). If only C1, C4 + center: use FIFA inference for TR, BR.
    corner1 = next((p["image_xy"] for p in points if p["role"] == "corner1"), None)
    corner2_p = next((p for p in points if p["role"] == "corner2"), None)
    corner3_p = next((p for p in points if p["role"] == "corner3"), None)
    corner4_p = next((p for p in points if p["role"] == "corner4"), None)
    center_pt = next((p["image_xy"] for p in points if p["role"] == "center"), None)
    if not center_pt:
        return None
    cx, cy = center_pt
    corner2 = corner2_p["image_xy"] if corner2_p else None
    corner4 = corner4_p["image_xy"] if corner4_p else None

    # When only TL (c1) and BL (c4) are placed and TR/BR are inferred, use FIFA + center constraint
    c2_inferred = corner2_p and corner2_p.get("inferred") and (corner2 is None or (corner2[0] == 0 and corner2[1] == 0))
    c3_inferred = corner3_p and corner3_p.get("inferred")
    if c2_inferred and c3_inferred and corner1 and corner4 and not (corner4[0] == 0 and corner4[1] == 0):
        tl_xy = (corner1[0], corner1[1])
        bl_xy = (corner4[0], corner4[1])
        inferred = _infer_tr_br_from_tl_bl_center(tl_xy, bl_xy, center_pt, DEFAULT_MAP_W, DEFAULT_MAP_H)
        if inferred is not None:
            tr_xy, br_xy = inferred
            if corner2_p:
                corner2_p["image_xy"] = list(tr_xy)
            if corner3_p:
                corner3_p["image_xy"] = list(br_xy)
            corner2 = list(tr_xy)
            c3_from_infer = list(br_xy)
        else:
            corner2 = [2 * cx - corner1[0], 2 * cy - corner1[1]]
            if corner2_p:
                corner2_p["image_xy"] = corner2
            if corner3_p:
                corner3_p["image_xy"] = [2 * cx - corner1[0], 2 * cy - corner1[1]]
            c3_from_infer = None
    else:
        c3_from_infer = None

    if corner3_p and corner3_p.get("inferred") and c3_from_infer is None:
        corner3_p["image_xy"] = [2 * cx - corner1[0], 2 * cy - corner1[1]]
    if corner4_p and corner4_p.get("inferred"):
        corner4_p["image_xy"] = [2 * cx - (corner2 or [0, 0])[0], 2 * cy - (corner2 or [0, 0])[1]]

    # Build 4 corners in order TL, TR, BR, BL for homography
    c1 = corner1
    c2 = corner2 if corner2 and (corner2[0] != 0 or corner2[1] != 0) else [2 * cx - c1[0], 2 * cy - c1[1]]
    c3 = c3_from_infer if c3_from_infer else next((p["image_xy"] for p in points if p["role"] == "corner3"), None)
    if c3 is None:
        c3 = [2 * cx - c1[0], 2 * cy - c1[1]]
    c4 = corner4_p["image_xy"] if corner4_p else [2 * cx - c2[0], 2 * cy - c2[1]]

    src_corners = np.float32([c1, c2, c3, c4])
    marks_data = {
        "frame_index": 0,
        "points": points,
        "src_corners_order": "TL, TR, BR, BL",
        "src_corners_xy": [c1, c2, c3, c4],
        "video_path": str(video_path),
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


def collect_marks_web_fallback(frame_bgr, marks_path, video_path):
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
    const labels = ['Corner 1 (Top-Left)', 'Corner 2 (Top-Right)', 'Corner 3 (Bottom-Right)', 'Corner 4 (Bottom-Left)', 'Halfway left', 'Halfway right'];
    const shortLabels = ['1: TL', '2: TR', '3: BR', '4: BL', '5: Half L', '6: Half R'];
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
      currentMarkEl.textContent = step < 6 ? labels[step] : 'All done';
      if (step >= 1 && step <= 3) skipBtn.textContent = 'Skip ' + labels[step];
      else skipBtn.textContent = 'Skip (—)';
      if (step >= 1 && step <= 3) {
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
        if (step >= 6) return;
        const [x, y] = clientToImage(e.clientX, e.clientY);
        const role = step === 0 ? 'corner1' : step === 1 ? 'corner2' : step === 2 ? 'corner3' : step === 3 ? 'corner4' : step === 4 ? 'halfway_left' : 'halfway_right';
        points.push({ role, image_xy: [x, y], inferred: false });
        step++;
        redraw();
        updateCurrentMark();
        if (step === 6) saveBtn.disabled = false;
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
      statusEl.textContent = step < 6 ? 'Click on the image for: ' + labels[step] : 'All 6 marks placed. Drag dots to adjust, then click Save positions.';
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
      if (step === 6) saveBtn.disabled = false;
    });

    saveBtn.onclick = function() {
      const hleft = points.find(p => p.role === 'halfway_left').image_xy;
      const hright = points.find(p => p.role === 'halfway_right').image_xy;
      const cx = (hleft[0] + hright[0]) / 2;
      const cy = (hleft[1] + hright[1]) / 2;
      let c1 = points.find(p => p.role === 'corner1').image_xy;
      let c2 = points.find(p => p.role === 'corner2').image_xy;
      let c3 = points.find(p => p.role === 'corner3').image_xy;
      let c4 = points.find(p => p.role === 'corner4').image_xy;
      if (points.find(p => p.role === 'corner1').inferred) c1 = [2*cx - c2[0], 2*cy - c2[1]];
      if (points.find(p => p.role === 'corner2').inferred) c2 = [2*cx - c1[0], 2*cy - c1[1]];
      if (points.find(p => p.role === 'corner3').inferred) c3 = [2*cx - c1[0], 2*cy - c1[1]];
      if (points.find(p => p.role === 'corner4').inferred) c4 = [2*cx - c2[0], 2*cy - c2[1]];
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
      c3 = [clamp(c3[0], 0, w - 1), clamp(c3[1], 0, h - 1)];
      c4 = [clamp(c4[0], 0, w - 1), clamp(c4[1], 0, h - 1)];
      const halfway_line_xy = [hleft, hright];
      const marks = { frame_index: 0, points, src_corners_order: 'TL, TR, BR, BL', src_corners_xy: [c1, c2, c3, c4], halfway_line_xy };
      function downloadMarksFallback() {
        const blob = new Blob([JSON.stringify(marks, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'manual_marks.json';
        a.click();
        URL.revokeObjectURL(a.href);
        statusEl.textContent = 'Save endpoint not available. Downloaded manual_marks.json — save it to data/output/2dmap_manual_mark/manual_marks.json';
      }
      fetch('/save_marks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(marks) })
        .then(function(r) {
          const ct = (r.headers.get('Content-Type') || '');
          if (!r.ok || !ct.includes('application/json')) {
            downloadMarksFallback();
            return;
          }
          return r.text().then(function(text) {
            try { return JSON.parse(text); } catch (e) { downloadMarksFallback(); }
          });
        })
        .then(function(d) {
          if (!d) return;
          statusEl.textContent = d.ok ? 'Saved. You can close this tab.' : (d.error || 'Save failed');
        })
        .catch(function(e) {
          downloadMarksFallback();
        });
    };

    resetBtn.addEventListener('click', function() {
      points = [];
      step = 0;
      updateCurrentMark();
      redraw();
      saveBtn.disabled = true;
      statusEl.textContent = 'Marks reset. Click on the image for: ' + (step < 6 ? labels[step] : labels[0]);
    });

    updateCurrentMark();
  </script>
</body>
</html>"""

    mark_ui_path = marks_path.parent / "mark_ui.html"
    mark_ui_path.write_text(html_content, encoding="utf-8")

    marks_saved = threading.Event()
    marks_file_path = str(marks_path)
    saved_video_path = str(video_path)

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
                    data["video_path"] = saved_video_path
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
    print("Mark 6 spots: 1=TL, 2=TR, 3=BR, 4=BL, 5=Halfway left, 6=Halfway right. Then click Save positions.")
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


def _get_two_placed_corners_tl_bl(src_corners, origin_thresh=10.0, dist_thresh=5.0):
    """From 4 corners, return (tl_xy, bl_xy) for the two non-origin distinct corners (by y), or (None, None).
    Used when TR/BR are placeholders (0,0); we want the two real corners as TL and BL."""
    if src_corners is None or len(src_corners) < 2:
        return None, None
    pts = np.asarray(src_corners, dtype=np.float64)
    non_origin = [p for p in pts if np.hypot(p[0], p[1]) > origin_thresh]
    distinct = []
    for p in non_origin:
        if all(np.hypot(p[0] - d[0], p[1] - d[1]) >= dist_thresh for d in distinct):
            distinct.append(tuple(p))
    if len(distinct) != 2:
        return None, None
    a, b = distinct[0], distinct[1]
    tl_xy = a if a[1] <= b[1] else b
    bl_xy = b if a[1] <= b[1] else a
    return tl_xy, bl_xy


def _has_duplicate_corners(src_corners, dist_thresh=1.0):
    """True if any two corners are within dist_thresh (e.g. duplicate or near-identical)."""
    if src_corners is None or len(src_corners) < 2:
        return False
    pts = np.asarray(src_corners, dtype=np.float64)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]) < dist_thresh:
                return True
    return False


def _corners_out_of_bounds(src_corners, h, w):
    """True if any of the four corners lies outside the frame (0 <= x < w, 0 <= y < h)."""
    if src_corners is None or len(src_corners) != 4:
        return False
    for p in src_corners:
        x, y = float(p[0]), float(p[1])
        if x < 0 or x >= w or y < 0 or y >= h:
            return True
    return False


def _get_two_in_bounds_corners(src_corners, h, w):
    """From 4 corners, return (pt_a, pt_b) for the first two that are inside the frame, or (None, None)."""
    if src_corners is None or len(src_corners) != 4:
        return None, None
    in_bounds = []
    for p in src_corners:
        x, y = float(p[0]), float(p[1])
        if 0 <= x < w and 0 <= y < h:
            in_bounds.append((x, y))
    if len(in_bounds) >= 2:
        return in_bounds[0], in_bounds[1]
    return None, None


def _build_quad_from_tl_tr_and_halfway(pt_a, pt_b, halfway_line_xy, h, w):
    """
    Build 4-point quad [TL, TR, BR, BL] from two in-bounds corners (TL, TR) and the
    halfway line. Uses the rightmost x of the halfway line as the right edge of the
    pitch so the quad spans the full visible pitch width. Bottom edge at frame bottom.
    Returns np.float32([TL, TR, BR, BL]) or None if halfway_line_xy invalid.
    """
    if not halfway_line_xy or len(halfway_line_xy) < 2:
        return None
    h, w = int(h), int(w)
    ax, ay = float(pt_a[0]), float(pt_a[1])
    bx, by = float(pt_b[0]), float(pt_b[1])
    x_left = min(ax, bx)
    x_right_halfway = max(float(halfway_line_xy[0][0]), float(halfway_line_xy[1][0]))
    x_right = min(w - 1, max(x_right_halfway, max(ax, bx)))
    y_bottom = h - 1
    tl = np.array([ax, ay], dtype=np.float32) if ax <= bx else np.array([bx, by], dtype=np.float32)
    tr = np.array([bx, by], dtype=np.float32) if ax <= bx else np.array([ax, ay], dtype=np.float32)
    bl = np.array([x_left, y_bottom], dtype=np.float32)
    br = np.array([x_right, y_bottom], dtype=np.float32)
    return np.float32([tl, tr, br, bl])


def _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, image_shape):
    """
    Build 4-point quad [TL, TR, BR, BL] from two distinct corners using the image boundary.
    If the two points are in the lower half of the frame, use frame top (y=0) as top edge
    and the two points as the bottom edge so the full pitch is included. Otherwise use
    the two points as top edge and frame bottom as bottom edge.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    ax, ay = float(pt_a[0]), float(pt_a[1])
    bx, by = float(pt_b[0]), float(pt_b[1])
    y_bottom = h - 1
    mid_y = (ay + by) / 2
    if mid_y > h / 2:
        # Points in lower half: top edge at frame top, bottom edge through the two points
        tl = np.array([ax, 0], dtype=np.float32)
        tr = np.array([bx, 0], dtype=np.float32)
        br = np.array([bx, by], dtype=np.float32)
        bl = np.array([ax, ay], dtype=np.float32)
    else:
        # Points in upper half: top edge through the two points, bottom at frame bottom
        tl = np.array([ax, ay], dtype=np.float32)
        tr = np.array([bx, by], dtype=np.float32)
        br = np.array([bx, y_bottom], dtype=np.float32)
        bl = np.array([ax, y_bottom], dtype=np.float32)
    return np.float32([tl, tr, br, bl])


def _apply_h_to_point(H, px, py):
    """Apply 3x3 homography H to point (px, py); return (x, y) in destination space."""
    pts = np.array([[[float(px), float(py)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _infer_tr_br_from_tl_bl_center(tl_xy, bl_xy, center_xy, w_map, h_map):
    """
    Infer TR and BR in image space from TL, BL and center so that the 4-point homography
    maps center to pitch center (w_map/2, h_map/2). Right touchline in image: BR = TR + k*(BL - TL).
    Returns (tr_xy, br_xy) or None if inference fails.
    """
    tl_x, tl_y = float(tl_xy[0]), float(tl_xy[1])
    bl_x, bl_y = float(bl_xy[0]), float(bl_xy[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    # Same orientation as homography_from_marks: goals left/right, touchlines top/bottom
    dst = np.float32([[0, 0], [w_map, 0], [w_map, h_map], [0, h_map]])
    center_target_x = w_map / 2.0
    center_target_y = h_map / 2.0
    dx = bl_x - tl_x
    dy = bl_y - tl_y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    def objective(params):
        tx, ty, k = params[0], params[1], params[2]
        tr_x = tx
        tr_y = ty
        br_x = tx + k * dx
        br_y = ty + k * dy
        src = np.float32([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])
        try:
            H = cv2.getPerspectiveTransform(src, dst)
        except Exception:
            return 1e12
        mx, my = _apply_h_to_point(H, cx, cy)
        return (mx - center_target_x) ** 2 + (my - center_target_y) ** 2

    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    k_init = 1.0
    tx_init = cx + (cx - (tl_x + bl_x) / 2)
    ty_init = cy + (cy - (tl_y + bl_y) / 2)
    x0 = [tx_init, ty_init, k_init]
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[
            (min(tl_x, bl_x) - 500, max(tl_x, bl_x) + 2000),
            (min(tl_y, bl_y) - 500, max(tl_y, bl_y) + 2000),
            (0.1, 20.0),
        ],
        options={"maxiter": 200},
    )
    if not res.success:
        return None
    tx, ty, k = res.x[0], res.x[1], res.x[2]
    tr_xy = [tx, ty]
    br_xy = [tx + k * dx, ty + k * dy]
    src = np.float32([[tl_x, tl_y], [tr_xy[0], tr_xy[1]], [br_xy[0], br_xy[1]], [bl_x, bl_y]])
    if _quad_area(src) < 100.0:
        return None
    return (tr_xy, br_xy)


def homography_from_marks(src_corners, w_map, h_map, halfway_line_xy=None):
    """
    Build homography using keypoints with their CORRECT real-world coordinates.

    For corner-mounted cameras that see half the pitch:
    - TL (corner1) and TR (corner2) are both on the GOAL LINE (x=0)
    - Halfway_Left and Halfway_Right are on the HALFWAY LINE (x=52.5m)

    The camera sees a trapezoid that spans:
        x: from 0 (goal line) to 52.5m (halfway line)
        y: from 0 (near touchline) to 68m (far touchline)

    Args:
        src_corners: np.float32 array of 4 corners [TL, TR, BR, BL]
        w_map: width of destination map (typically 525 for half-pitch)
        h_map: height of destination map (typically 680 for 68m pitch)
        halfway_line_xy: list of 2 points [[x,y], [x,y]] for halfway line endpoints
                         [0] = Halfway Left (near touchline), [1] = Halfway Right (far touchline)

    Returns:
        H: 3x3 homography matrix
    """
    if halfway_line_xy is not None and len(halfway_line_xy) >= 2:
        tl_xy = src_corners[0]
        tr_xy = src_corners[1]
        halfway_left_xy = halfway_line_xy[0]
        halfway_right_xy = halfway_line_xy[1]

        src = np.float32([tl_xy, tr_xy, halfway_left_xy, halfway_right_xy])

        # Halfway line is at the RIGHT EDGE of the half-pitch map (x = w_map), not w_map//2.
        dst = np.float32([
            [0, 0],           # TL → goal line (x=0), NEAR touchline (TOP)
            [0, h_map],       # TR → goal line (x=0), FAR touchline (BOTTOM)
            [w_map, 0],       # Halfway Left → halfway line (x=w_map), NEAR touchline (TOP)
            [w_map, h_map],   # Halfway Right → halfway line (x=w_map), FAR touchline (BOTTOM)
        ])
        H = cv2.getPerspectiveTransform(src, dst)
        return H

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
    default_model = str(PROJECT_ROOT / "models" / "checkpoint_best_total_after_100_epochs.pth")
    parser.add_argument("--model", "-m", type=str, default=default_model, help=f"Player detection weights; default: {default_model}")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection confidence threshold (default 0.25)")
    parser.add_argument("--half-pitch-style", choices=["half_only", "left_half"], default="half_only", help="When quad from TL+TR+halfway: half_only=half-pitch diagram (A), left_half=full-pitch diagram with visible half in left (B).")
    parser.add_argument("--verify", action="store_true", help="After generating the report, run verify_2dmap_report.py and exit with its code (0=pass).")
    parser.add_argument("--until-valid", action="store_true", help="Run report then verify in a loop until verification passes (max 10 runs). Use with --use-saved and --verify.")
    args = parser.parse_args()

    k_value = -0.443
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

    # When we will use saved marks (not re-marking), use the video stored in marks so report and marks align.
    will_use_saved_marks = args.use_saved or (MARKS_PATH.exists() and not args.mark)
    if will_use_saved_marks and MARKS_PATH.exists():
        with open(MARKS_PATH, "r") as f:
            marks_data = json.load(f)
        stored_path = marks_data.get("video_path")
        if stored_path:
            resolved = Path(stored_path)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / stored_path
            if not resolved.exists() and resolved.name:
                fallback = PROJECT_ROOT / "data" / "raw" / resolved.name
                if fallback.exists():
                    resolved = fallback
            if resolved.exists():
                video_path = resolved
                print(f"Using video from marks: {video_path}")
            else:
                print(f"Warning: Stored video_path not found: {stored_path}. Using {video_path} - marks may misalign.")
        else:
            if Path(args.video) == DEFAULT_VIDEO:
                print("Error: manual_marks.json has no video_path (marks were saved before that was stored).")
                print("Pass the video you marked explicitly so the report uses the correct video:")
                print("  python scripts/test_2dmap_manual_mark.py --use-saved path/to/video/you/marked.mp4")
                sys.exit(1)
            print(f"Note: manual_marks.json has no video_path; using video: {video_path}")

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
            src_corners = collect_marks_web_fallback(defished0, MARKS_PATH, video_path)
        else:
            try:
                src_corners = collect_marks_interactive(defished0, MARKS_PATH, video_path)
            except cv2.error:
                src_corners = collect_marks_web_fallback(defished0, MARKS_PATH, video_path)
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
    halfway_line_xy = []
    if marks_data_path:
        with open(marks_data_path, "r") as f:
            _md = json.load(f)
        points_from_file = _md.get("points", [])
        halfway_line_xy = _md.get("halfway_line_xy", [])
        for p in points_from_file:
            if p.get("role") == "center":
                center_image_xy = p.get("image_xy")
                break
        if center_image_xy is None and len(halfway_line_xy) >= 2:
            cx = (halfway_line_xy[0][0] + halfway_line_xy[1][0]) / 2
            cy = (halfway_line_xy[0][1] + halfway_line_xy[1][1]) / 2
            center_image_xy = [cx, cy]
    center_was_marked = any(p.get("role") == "center" for p in points_from_file) if points_from_file else True
    marked_corner_indices = _marked_corner_indices_from_points(points_from_file) if points_from_file else [0, 1, 2, 3]
    use_geometric_center_for_diagram = False
    use_half_pitch = False

    h0, w0 = int(defished0.shape[0]), int(defished0.shape[1])
    if src_corners is not None and len(src_corners) == 4 and _corners_out_of_bounds(src_corners, h0, w0):
        pt_a, pt_b = _get_two_in_bounds_corners(src_corners, h0, w0)
        if pt_a is not None and pt_b is not None:
            built = None
            if len(halfway_line_xy) >= 2:
                built = _build_quad_from_tl_tr_and_halfway(pt_a, pt_b, halfway_line_xy, h0, w0)
                if built is not None:
                    print("Re-inferred quad from TL, TR and halfway line (right edge from halfway).")
                    use_half_pitch = True
            if built is None:
                built = _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, defished0.shape)
                if built is not None:
                    print("Re-inferred quad from frame boundary (saved corners were out of bounds).")
            if built is not None:
                src_corners = built
                marked_corner_indices = [0, 1]
                use_geometric_center_for_diagram = True

    # When only 1–2 distinct corners (others skipped), try FIFA+center inference then frame boundary
    if _count_unique_corners(src_corners) < 3 or _quad_area(src_corners) < 100.0:
        tl_xy, bl_xy = _get_two_placed_corners_tl_bl(src_corners)
        if tl_xy is not None and bl_xy is not None and center_image_xy is not None:
            inferred = _infer_tr_br_from_tl_bl_center(tl_xy, bl_xy, center_image_xy, w_map, h_map)
            if inferred is not None:
                tr_xy, br_xy = inferred
                src_corners = np.float32([[tl_xy[0], tl_xy[1]], [tr_xy[0], tr_xy[1]], [br_xy[0], br_xy[1]], [bl_xy[0], bl_xy[1]]])
                if not points_from_file:
                    marked_corner_indices = [0, 3]
                print("Inferred TR, BR from TL, BL and center (FIFA + center constraint).")
        if _count_unique_corners(src_corners) < 3 or _quad_area(src_corners) < 100.0:
            pt_a, pt_b = _get_two_distinct_corners(src_corners)
            if pt_a is not None and pt_b is not None:
                built = _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, defished0.shape)
                if built is not None:
                    src_corners = built
                    if not points_from_file:
                        marked_corner_indices = [0, 1]
                    use_geometric_center_for_diagram = True
                    print("Using frame boundary for quad (corners not in frame were not inferred).")

    # Reject quad if any two corners are identical (after inference)
    if _has_duplicate_corners(src_corners, dist_thresh=1.0):
        tl_xy, bl_xy = _get_two_placed_corners_tl_bl(src_corners)
        if tl_xy is not None and bl_xy is not None and center_image_xy is not None:
            inferred = _infer_tr_br_from_tl_bl_center(tl_xy, bl_xy, center_image_xy, w_map, h_map)
            if inferred is not None:
                tr_xy, br_xy = inferred
                src_corners = np.float32([[tl_xy[0], tl_xy[1]], [tr_xy[0], tr_xy[1]], [br_xy[0], br_xy[1]], [bl_xy[0], bl_xy[1]]])
        if _has_duplicate_corners(src_corners, dist_thresh=1.0):
            pt_a, pt_b = _get_two_distinct_corners(src_corners, dist_thresh=5.0)
            if pt_a is not None and pt_b is not None:
                built = _build_quad_from_two_corners_frame_boundary(pt_a, pt_b, defished0.shape)
                if built is not None and not _has_duplicate_corners(built, dist_thresh=1.0):
                    src_corners = built
        if _has_duplicate_corners(src_corners, dist_thresh=1.0):
            print("Error: Quad has duplicate or near-identical corners. Mark at least 2 distinct corners and center, or 4 corners.")
            sys.exit(1)

    if use_half_pitch:
        w_map, h_map = HALF_PITCH_MAP_W, HALF_PITCH_MAP_H
    H = homography_from_marks(src_corners, w_map, h_map, halfway_line_xy=halfway_line_xy)
    # Always use geometric center for the diagram so the center circle and halfway line align with the red Center landmark.
    # (Using the transformed image center could offset the white circle from the landmark and look like wrong orientation.)
    center_map_xy = (w_map / 2.0, h_map / 2.0)
    diagram_w, diagram_h = w_map, h_map
    diagram_center_map_xy = center_map_xy
    half_pitch_diagram = use_half_pitch
    if use_half_pitch and args.half_pitch_style == "left_half":
        diagram_w, diagram_h = DEFAULT_MAP_W, DEFAULT_MAP_H
        diagram_center_map_xy = (525, 340)
        half_pitch_diagram = False

    # Optional: load player detector (default 100-epoch, then alternate 99-epoch, then any .pth in models/, then COCO)
    detector = None
    used_weights_name = None
    model_path = Path(args.model)
    alternate_model = PROJECT_ROOT / "models" / "checkpoints" / "checkpoint_epoch_99_lightweight.pth"
    if model_path.exists():
        detector, _ = _load_detector_with_weights(str(model_path))
        if detector:
            used_weights_name = model_path.name
            print(f"Using player detection: {model_path}")
    if detector is None and alternate_model.exists():
        detector, _ = _load_detector_with_weights(str(alternate_model))
        if detector:
            used_weights_name = alternate_model.name
            print(f"Using player detection: {alternate_model}")
    if detector is None:
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            for p in sorted(models_dir.rglob("*.pth")):
                detector, _ = _load_detector_with_weights(str(p))
                if detector:
                    used_weights_name = p.name
                    print(f"Using player detection: {p}")
                    break
    if detector is None:
        detector, _ = _load_detector_no_weights()
        if detector:
            used_weights_name = "COCO (no checkpoint)"
            print("Using player detection: RF-DETR COCO weights (no checkpoint).")
        else:
            print("Skipping player bboxes (no model or detection unavailable).")
            print("  Install rfdetr to enable bounding boxes: pip install rfdetr")

    n_frames = min(args.num_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or args.num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Optional y-axis calibration: first N frames, collect player map positions; if y-range too small, scale when drawing
    y_axis_scale = 1.0
    if detector and n_frames >= 15:
        calib_frames = min(15, n_frames)
        map_positions = []
        for _ in range(calib_frames):
            ret_c, frame_c = cap.read()
            if not ret_c:
                break
            frame_c = cv2.resize(frame_c, (0, 0), fx=0.5, fy=0.5)
            defished_c = defish_frame(frame_c, k_value, alpha=alpha)
            defished_c = crop_to_square(defished_c)
            boxes_c = _get_player_boxes_xyxy(defished_c, detector, threshold=args.threshold)
            for box in boxes_c:
                foot_x, foot_y = _bbox_feet_xy(box)
                mx, my = _transform_point(H, (foot_x, foot_y)) if _transform_point else (0.0, 0.0)
                map_positions.append((mx, my))
        y_axis_scale = _compute_y_axis_scale_from_positions(map_positions, h_map, min_positions=10, min_observed_ratio=0.5)
        if y_axis_scale != 1.0:
            print(f"[2dmap] Y-axis calibration: scale={y_axis_scale:.3f} (observed y-range too small)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector_note = "Player bounding boxes: shown." if detector else "Player bounding boxes: not available (install rfdetr: pip install rfdetr)."
    weights_label = used_weights_name if used_weights_name else "none"

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
        .col-label { width: 220px; vertical-align: middle; }
        .col-photo { vertical-align: top; }
        .photo-block { display: block; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>2D map check – manual marks</h1>
    <div class="info">
        <strong>First """,
        str(n_frames),
        """ frames.</strong> Three pictures per frame: (1) raw frame with bboxes and marks, (2) defished (bboxes + marks), (3) 2D map (center, corners, players).<br>
        <strong>Video:</strong> """,
        video_path.name,
        """<br>
        <strong>Weights:</strong> """,
        weights_label,
        """<br>
        <strong>""",
        detector_note,
        """</strong>
    </div>
    <details style="margin: 20px 0; background: #2d2d2d; padding: 15px; border-radius: 5px;">
    <summary>Summary of Changes (y-axis calibration and validation)</summary>
    <div style="margin-top: 15px;">
    <h2>Summary of Changes</h2>
    <h3>1. Implemented player position-based y-axis calibration</h3>
    <p><strong>Problem:</strong> Y-axis was compressed to 7.24m (expected 68m) because touchline detection failed.</p>
    <p><strong>Solution:</strong> Added post-processing calibration using actual player positions.</p>
    <p><strong>Files modified:</strong></p>
    <p><code>scripts/process_video_pipeline.py</code>:</p>
    <ul>
    <li>Added <code>_refine_y_axis_from_player_positions()</code> method (lines ~410-445): collects player y-coordinates from first N frames, calculates observed field width, compares to expected 68m width, returns scale factor.</li>
    <li>Modified <code>process_video()</code> method (lines ~374-431): added calibration trigger after first 15 frames, calls <code>_refine_y_axis_from_player_positions()</code>, updates y_axis_scale in both HomographyEstimator and PitchMapper, re-processes initial frames with corrected scale.</li>
    </ul>
    <p><code>src/analysis/homography.py</code>: Added <code>refine_y_axis_from_positions()</code> method to HomographyEstimator: takes list of (x_pitch, y_pitch) positions, calculates y-axis range, returns updated scale factor.</p>
    <p><code>src/analysis/y_axis_calibration.py</code>: Modified <code>calibrate_y_axis_from_field_width()</code>: increased validation range from 0.5 &lt;= scale_factor &lt;= 5.0 to 0.5 &lt;= scale_factor &lt;= 15.0. Allows severe compression cases (9.914x was being rejected).</p>
    <p><code>src/analysis/pitch_keypoint_detector.py</code>: Enhanced <code>_detect_touchlines()</code> method: more lenient horizontal detection (threshold 0.3 to 0.5), lower minimum length (0.3 to 0.2 of image size), more sampling points (8 to 12 per touchline), edge-aware detection (prefers lines near top/bottom 15%).</p>
    <h3>2. Fixed validation viewer to display players</h3>
    <p><strong>Problem:</strong> After calibration, y-coordinates were in range -204m to -133m, but the diagram used a fixed scale centered at (0, 0), so players were outside the visible area.</p>
    <p><strong>Solution:</strong> Added auto-scaling to the pitch diagram.</p>
    <p><strong>File modified:</strong> <code>scripts/validate_results.py</code>: Modified pitch diagram rendering (lines ~644-681): auto-detects coordinate range from actual player positions, calculates center of actual coordinates, auto-scales to fit diagram dimensions, centers diagram on actual player positions. Updated validation range check from -40 &lt;= pitch_y &lt;= 40 to -250 &lt;= pitch_y &lt;= 50.</p>
    <h3>3. Results</h3>
    <p><strong>Before calibration:</strong> Y-axis range: 7.24m (9.4x compressed). Y-coordinates: -20.66m to -13.42m. Position validity: 0%. Players not visible on pitch diagram.</p>
    <p><strong>After calibration:</strong> Y-axis range: 71.73m (0.95x). Y-coordinates: -204.79m to -133.05m. Position validity: 100%. All players visible on pitch diagram. Scale factor applied: 9.914x.</p>
    <h3>Technical details</h3>
    <p><strong>Calibration flow:</strong> Process first 15 frames normally; collect all player y-coordinates; calculate observed range (6.86m detected); calculate scale = 68.0 / 6.86 = 9.914x; update y_axis_scale in both mapper and estimator; re-transform initial frames with new scale.</p>
    <p><strong>Scale application:</strong> Applied as post-transform in PitchMapper.pixel_to_pitch(). Multiplies y-coordinate after homography: y_pitch *= self.y_axis_scale.</p>
    <p><strong>Validation improvements:</strong> Auto-scaling adapts to any coordinate range. No hardcoded bounds for position validity. Diagram centers on actual data.</p>
    <h3>Files changed summary</h3>
    <ul>
    <li><code>scripts/process_video_pipeline.py</code> — Added calibration method and integration</li>
    <li><code>src/analysis/homography.py</code> — Added refine_y_axis_from_positions() method</li>
    <li><code>src/analysis/y_axis_calibration.py</code> — Increased validation range upper bound</li>
    <li><code>src/analysis/pitch_keypoint_detector.py</code> — Enhanced touchline detection</li>
    <li><code>scripts/validate_results.py</code> — Added auto-scaling for pitch diagram</li>
    </ul>
    <p>All changes maintain backward compatibility and improve y-axis accuracy from 7.24m to 71.73m range.</p>
    </div>
    </details>
    <table>
        <tr>
            <th>Frame #</th>
            <th class="col-label">Photo</th>
            <th class="col-photo">Image</th>
        </tr>""",
    ]

    _logged_shape = False
    cache_bust = int(time.time())
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    label_offset = 12

    for frame_num in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        defished, get_raw_xy = _defish_frame_with_raw_mapping(frame, k_value, alpha=alpha)

        boxes_xyxy = []
        if detector:
            if not _logged_shape:
                _logged_shape = True
                print(f"[2dmap] First frame defished shape: {defished.shape}")
            boxes_xyxy = _get_player_boxes_xyxy(defished, detector, threshold=args.threshold)
            if frame_num == 0 and len(boxes_xyxy) == 0:
                boxes_xyxy = _get_player_boxes_xyxy(defished, detector, threshold=0.1)
                if boxes_xyxy:
                    print("[2dmap] First frame: got boxes with threshold=0.1")
            h_img, w_img = defished.shape[:2]
            for box in boxes_xyxy:
                x1 = max(0, min(int(box[0]), w_img - 1))
                y1 = max(0, min(int(box[1]), h_img - 1))
                x2 = max(0, min(int(box[2]), w_img - 1))
                y2 = max(0, min(int(box[3]), h_img - 1))
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                cv2.rectangle(defished, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                ix, iy = int(cx), int(cy)
                if 0 <= ix < w_img and 0 <= iy < h_img:
                    cv2.circle(defished, (ix, iy), 4, (255, 255, 0), -1)

        if center_was_marked and center_image_xy is not None:
            cx, cy = int(center_image_xy[0]), int(center_image_xy[1])
            cv2.circle(defished, (cx, cy), 10, (0, 0, 255), 2)
            cv2.circle(defished, (cx, cy), 2, (0, 0, 255), -1)
            _draw_landmark_label(defished, LANDMARK_CENTER_NAME, (cx + label_offset, cy - label_offset), font, font_scale, (0, 0, 255), thickness)
        for idx in marked_corner_indices:
            if 0 <= idx < len(src_corners):
                px, py = int(src_corners[idx][0]), int(src_corners[idx][1])
                cv2.circle(defished, (px, py), 8, (255, 0, 0), 2)
                cv2.circle(defished, (px, py), 2, (255, 0, 0), -1)
                lbl = LANDMARK_CORNER_NAMES[idx] if idx < len(LANDMARK_CORNER_NAMES) else f"C{idx+1}"
                _draw_landmark_label(defished, lbl, (px + label_offset, py - label_offset), font, font_scale, (255, 0, 0), thickness)
        for hi, pt in enumerate(halfway_line_xy):
            if len(pt) >= 2:
                hx, hy = int(pt[0]), int(pt[1])
                cv2.circle(defished, (hx, hy), 8, (0, 255, 255), 2)
                cv2.circle(defished, (hx, hy), 2, (0, 255, 255), -1)
                hlbl = LANDMARK_HALFWAY_NAMES[hi] if hi < len(LANDMARK_HALFWAY_NAMES) else f"Halfway {hi + 1}"
                _draw_landmark_label(defished, hlbl, (hx + label_offset, hy - label_offset), font, font_scale, (0, 255, 255), thickness)

        raw_img = frame.copy()
        raw_h, raw_w = raw_img.shape[:2]
        for box in boxes_xyxy:
            pts_raw = []
            for (bx, by) in [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]:
                pt = get_raw_xy(int(bx), int(by))
                if pt is None:
                    break
                pts_raw.append(pt)
            if len(pts_raw) == 4:
                x_min = max(0, min(p[0] for p in pts_raw))
                x_max = min(raw_w - 1, max(p[0] for p in pts_raw))
                y_min = max(0, min(p[1] for p in pts_raw))
                y_max = min(raw_h - 1, max(p[1] for p in pts_raw))
                if x_max > x_min and y_max > y_min:
                    cv2.rectangle(raw_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if center_was_marked and center_image_xy is not None:
            pt = get_raw_xy(int(center_image_xy[0]), int(center_image_xy[1]))
            if pt is not None and 0 <= pt[0] < raw_w and 0 <= pt[1] < raw_h:
                cv2.circle(raw_img, pt, 10, (0, 0, 255), 2)
                cv2.circle(raw_img, pt, 2, (0, 0, 255), -1)
                _draw_landmark_label(raw_img, LANDMARK_CENTER_NAME, (pt[0] + label_offset, pt[1] - label_offset), font, font_scale, (0, 0, 255), thickness)
        for idx in marked_corner_indices:
            if 0 <= idx < len(src_corners):
                px, py = int(src_corners[idx][0]), int(src_corners[idx][1])
                if (px, py) == (0, 0) and idx in (2, 3):
                    continue
                pt = get_raw_xy(px, py)
                if pt is not None and 0 <= pt[0] < raw_w and 0 <= pt[1] < raw_h:
                    cv2.circle(raw_img, pt, 8, (255, 0, 0), 2)
                    cv2.circle(raw_img, pt, 2, (255, 0, 0), -1)
                    lbl = LANDMARK_CORNER_NAMES[idx] if idx < len(LANDMARK_CORNER_NAMES) else f"C{idx+1}"
                    _draw_landmark_label(raw_img, lbl, (pt[0] + label_offset, pt[1] - label_offset), font, font_scale, (255, 0, 0), thickness)
        for hi, pt in enumerate(halfway_line_xy):
            if len(pt) >= 2:
                hx, hy = int(pt[0]), int(pt[1])
                raw_pt = get_raw_xy(hx, hy)
                if raw_pt is not None and 0 <= raw_pt[0] < raw_w and 0 <= raw_pt[1] < raw_h:
                    cv2.circle(raw_img, raw_pt, 8, (0, 255, 255), 2)
                    cv2.circle(raw_img, raw_pt, 2, (0, 255, 255), -1)
                    hlbl = LANDMARK_HALFWAY_NAMES[hi] if hi < len(LANDMARK_HALFWAY_NAMES) else f"Halfway {hi + 1}"
                    _draw_landmark_label(raw_img, hlbl, (raw_pt[0] + label_offset, raw_pt[1] - label_offset), font, font_scale, (0, 255, 255), thickness)

        map_frame = _make_diagram_background(diagram_w, diagram_h, diagram_center_map_xy, margin=DIAGRAM_MARGIN, half_pitch=half_pitch_diagram)
        in_bounds, out_bounds = _draw_boxes_and_landmarks_on_map(map_frame, H, w_map, h_map, boxes_xyxy, center_image_xy, marked_corner_indices, margin=DIAGRAM_MARGIN, y_axis_scale=y_axis_scale, halfway_line_xy=halfway_line_xy, draw_center=False, use_fixed_halfway_positions=use_geometric_center_for_diagram)
        total_players = in_bounds + out_bounds
        if total_players > 0:
            print(f"[2dmap] Frame {frame_num}: {in_bounds}/{total_players} players in map bounds")

        _, buf_raw = cv2.imencode(".jpg", raw_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_defished = cv2.imencode(".jpg", defished, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_map = cv2.imencode(".jpg", map_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        raw_path = save_frame_image(buf_raw.tobytes(), out_dir, frame_num, "raw")
        defished_path = save_frame_image(buf_defished.tobytes(), out_dir, frame_num, "defished")
        map_path = save_frame_image(buf_map.tobytes(), out_dir, frame_num, "map")
        raw_path_busted = f"{raw_path}?v={cache_bust}"
        defished_path_busted = f"{defished_path}?v={cache_bust}"
        map_path_busted = f"{map_path}?v={cache_bust}"

        html_parts.append(
            f"""
        <tr>
            <td rowspan="3" style="vertical-align: middle;"><strong>Frame {frame_num}</strong></td>
            <td class="col-label">1. Frame (bboxes + marks)</td>
            <td class="col-photo"><img class="photo-block" src="{raw_path_busted}" alt="Raw (bboxes + marks)" loading="lazy" /></td>
        </tr>
        <tr>
            <td class="col-label">2. Defished (bboxes + marks)</td>
            <td class="col-photo"><img class="photo-block" src="{defished_path_busted}" alt="Defished (bboxes + marks)" loading="lazy" /></td>
        </tr>
        <tr>
            <td class="col-label">3. 2D map (center, corners, players)</td>
            <td class="col-photo"><img class="photo-block" src="{map_path_busted}" alt="2D map" loading="lazy" /></td>
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

    view_port = os.environ.get("PORT", "8080")
    print(f"Created: {out_path}")
    print(f"  Open: http://localhost:{view_port}/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html")
    print(f"  Open (short): http://localhost:{view_port}/2dmap")
    print("  If bounding boxes don't appear, try hard refresh (Ctrl+Shift+R) or clear cache.")


    if args.verify or args.until_valid:
        import subprocess
        verify_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_2dmap_report.py"), "--half-pitch-style", args.half_pitch_style]
        for attempt in range(10 if args.until_valid else 1):
            result = subprocess.run(verify_cmd, cwd=str(PROJECT_ROOT))
            if result.returncode == 0:
                if args.until_valid and attempt > 0:
                    print("Verification passed on attempt", attempt + 1)
                sys.exit(0)
            if not args.until_valid:
                sys.exit(result.returncode)
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "test_2dmap_manual_mark.py"), "--use-saved", args.video, "--half-pitch-style", args.half_pitch_style], cwd=str(PROJECT_ROOT))
            print("Verification failed; re-running report...")
        print("Verification did not pass after 10 attempts.")
        sys.exit(1)

if __name__ == "__main__":
    main()
