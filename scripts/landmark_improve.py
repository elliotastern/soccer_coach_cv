"""
Shared improved landmark logic: center circle (Hough), center line (vertical / circle center),
touchline filtering. Used by test_landmarks.py and find_landmarks.py.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.pitch_keypoint_detector import PitchKeypoint

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_CIRCLE_RADIUS_M = 9.15


def detect_center_circle_hough(frame):
    """
    Find the center circle via HoughCircles (circle closest to image center).
    Returns (cx, cy, radius_px) or None.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    center_x, center_y = w / 2, h / 2
    mn = min(h, w)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=int(mn / 3),
        param1=50,
        param2=20,
        minRadius=int(mn / 25),
        maxRadius=int(mn / 8),
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    best = None
    best_dist = float("inf")
    for c in circles[0]:
        cx, cy, r = c
        d = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if d < best_dist:
            best_dist = d
            best = (float(cx), float(cy), float(r))
    return best


def center_circle_keypoints_from_hough(cx, cy, r, num_points=8, pitch_radius=CENTER_CIRCLE_RADIUS_M):
    """Generate center_circle PitchKeypoints: center + points on circle (pitch radius 9.15m)."""
    kps = [
        PitchKeypoint(
            image_point=(cx, cy),
            pitch_point=(0.0, 0.0),
            landmark_type="center_circle",
            confidence=0.85,
        )
    ]
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        img_x = cx + r * np.sin(angle)
        img_y = cy - r * np.cos(angle)
        pitch_x = pitch_radius * np.sin(angle)
        pitch_y = -pitch_radius * np.cos(angle)
        kps.append(
            PitchKeypoint(
                image_point=(float(img_x), float(img_y)),
                pitch_point=(pitch_x, pitch_y),
                landmark_type="center_circle",
                confidence=0.75,
            )
        )
    return kps


def detect_dominant_vertical_line(frame, min_length_ratio=0.2):
    """
    Detect the dominant vertical line (center line of the field) via Hough lines.
    Returns (center_x, [(y1, y2), ...]) for the longest near-vertical line, or (None, []) if not found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(min(h, w) * 0.15), maxLineGap=20)
    if lines is None:
        return None, []
    vertical_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < min(h, w) * min_length_ratio:
            continue
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(abs(angle) - 90) < 10:
            x_avg = (x1 + x2) / 2
            vertical_segments.append((x_avg, length, (min(y1, y2), max(y1, y2))))
    if not vertical_segments:
        return None, []
    vertical_segments.sort(key=lambda s: (-s[1], abs(s[0] - w / 2)))
    best_x = vertical_segments[0][0]
    best_ys = [(vertical_segments[0][2][0], vertical_segments[0][2][1])]
    for x_avg, length, ys in vertical_segments[1:]:
        if abs(x_avg - best_x) < w * 0.05:
            best_ys.append(ys)
    y_min = min(ys[0] for ys in best_ys)
    y_max = max(ys[1] for ys in best_ys)
    return best_x, [(y_min, y_max)]


def center_line_keypoints_from_vertical(center_x, y_range, h, w, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, num_samples=8):
    """
    Generate center_line PitchKeypoints along the dominant vertical line.
    pitch_point = (0, y_pitch) with y_pitch in [-pitch_width/2, pitch_width/2]; map image y linearly.
    """
    y_lo, y_hi = y_range[0][0], y_range[0][1]
    keypoints = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        y_img = y_lo + t * (y_hi - y_lo)
        if not (0 <= y_img < h):
            continue
        y_norm = (y_img - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.5
        y_pitch = -pitch_width / 2 + y_norm * pitch_width
        keypoints.append(
            PitchKeypoint(
                image_point=(float(center_x), float(y_img)),
                pitch_point=(0.0, y_pitch),
                landmark_type="center_line",
                confidence=0.9,
            )
        )
    return keypoints


def filter_touchlines(keypoints, center_x, h, w, min_dist_from_center_ratio=0.2, exclude_bottom_ratio=0.15, max_per_side=4):
    """Keep touchlines sufficiently far from center, above the debris zone, and cap per side."""
    if center_x is None:
        return keypoints
    min_dist = min(w, h) * min_dist_from_center_ratio
    y_cut = h * (1 - exclude_bottom_ratio)
    touchlines = [kp for kp in keypoints if kp.landmark_type == "touchline"]
    others = [kp for kp in keypoints if kp.landmark_type != "touchline"]
    left = [kp for kp in touchlines if kp.image_point[0] < center_x - min_dist and kp.image_point[1] < y_cut]
    right = [kp for kp in touchlines if kp.image_point[0] > center_x + min_dist and kp.image_point[1] < y_cut]

    def take_best(n, pts, key):
        return sorted(pts, key=key)[:n]

    left = take_best(max_per_side, left, lambda kp: kp.image_point[0])
    right = take_best(max_per_side, right, lambda kp: -kp.image_point[0])
    return others + left + right


def apply_center_line_and_filter(defished, raw_keypoints, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """
    Find center line and center circle; add keypoints and filter touchlines.
    - If center circle is found (HoughCircles): use its center x as center line, add center_circle + center_line keypoints.
    - Else: find dominant vertical line (Hough), add center_line keypoints only.
    Removes detector's center_line and center_circle when we supply our own.
    """
    h, w = defished.shape[:2]
    other = [kp for kp in raw_keypoints if kp.landmark_type not in ("center_line", "center_circle")]
    center_x = None
    extra_kps = []

    circle = detect_center_circle_hough(defished)
    if circle is not None:
        cx, cy, r = circle
        center_x = cx
        extra_kps.extend(center_circle_keypoints_from_hough(cx, cy, r))
        y_ranges = [(max(0, cy - h * 0.4), min(h, cy + h * 0.4))]
        extra_kps.extend(
            center_line_keypoints_from_vertical(center_x, y_ranges, h, w, pitch_length, pitch_width)
        )
    else:
        center_x, y_ranges = detect_dominant_vertical_line(defished)
        if center_x is not None and y_ranges:
            extra_kps.extend(
                center_line_keypoints_from_vertical(center_x, y_ranges, h, w, pitch_length, pitch_width)
            )

    combined = other + extra_kps
    combined = filter_touchlines(combined, center_x, h, w)
    return combined
