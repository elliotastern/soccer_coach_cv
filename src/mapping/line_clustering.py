"""
Line clustering and virtual keypoint generation from Hough segments.
Vanishing point (VP) estimation, DBSCAN in (rho, theta), merge to line equations,
then virtual keypoints from longitudinal-transversal intersections.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@dataclass
class MergedLine:
    """Single line equation ax + by + c = 0 (normalized so a^2+b^2=1)."""
    a: float
    b: float
    c: float
    is_longitudinal: bool  # True = converges to VP (lengthwise), False = transversal


def _segment_to_rho_theta(segment: np.ndarray) -> Tuple[float, float]:
    """Convert segment [x1,y1,x2,y2] to Hough (rho, theta)."""
    x1, y1, x2, y2 = segment[0]
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return 0.0, 0.0
    # Normal direction: (-dy, dx) normalized
    nx, ny = -dy / length, dx / length
    theta = np.arctan2(ny, nx)  # angle of normal
    if theta < 0:
        theta += np.pi
    rho = x1 * nx + y1 * ny
    return float(rho), float(theta)


def _segment_to_line_equation(segment: np.ndarray) -> Tuple[float, float, float]:
    """Convert segment to line equation ax + by + c = 0 (normalized)."""
    x1, y1, x2, y2 = segment[0]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    n = np.sqrt(a * a + b * b)
    if n < 1e-10:
        return 0.0, 0.0, 0.0
    a, b, c = a / n, b / n, c / n
    return float(a), float(b), float(c)


def _line_intersection(l1: MergedLine, l2: MergedLine) -> Optional[Tuple[float, float]]:
    """Intersection of two lines; returns (x, y) or None if parallel."""
    det = l1.a * l2.b - l1.b * l2.a
    if abs(det) < 1e-10:
        return None
    x = (l1.b * l2.c - l1.c * l2.b) / det
    y = (l1.c * l2.a - l1.a * l2.c) / det
    return (float(x), float(y))


def _estimate_vanishing_point_ransac(
    lines: List[np.ndarray],
    w: int,
    h: int,
    n_trials: int = 200,
    inlier_threshold: float = 50.0,
) -> Optional[Tuple[float, float]]:
    """
    Estimate primary vanishing point (longitudinal lines) via RANSAC.
    Sample pairs of segments, compute intersection, return VP with highest inlier count.
    """
    if len(lines) < 2:
        return None
    n = len(lines)
    intersections = []
    for i in range(n):
        x1, y1, x2, y2 = lines[i][0]
        for j in range(i + 1, n):
            x3, y3, x4, y4 = lines[j][0]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            intersections.append((ix, iy))
    if not intersections:
        return None
    pts = np.array(intersections, dtype=np.float64)
    # VP can be far outside image; use median or RANSAC on distance to lines
    best_vp = None
    best_inliers = 0
    for _ in range(n_trials):
        i, j = np.random.randint(0, len(pts), 2)
        if i == j:
            continue
        vpx, vpy = (pts[i] + pts[j]) / 2.0
        inliers = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Distance from VP to line
            a, b, c = _segment_to_line_equation(line)
            dist = abs(a * vpx + b * vpy + c)
            if dist < inlier_threshold:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_vp = (float(vpx), float(vpy))
    if best_vp is not None:
        return best_vp
    # Fallback: median of intersections (often VP is near there)
    return (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))


def _classify_longitudinal_transversal(
    lines: List[np.ndarray],
    vp: Tuple[float, float],
    w: int,
    h: int,
    vp_distance_threshold: float = 150.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Classify segments as longitudinal (pass near VP) or transversal.
    Longitudinal: line extended passes close to VP.
    """
    vpx, vpy = vp
    longitudinal = []
    transversal = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        a, b, c = _segment_to_line_equation(seg)
        dist_vp_to_line = abs(a * vpx + b * vpy + c)
        if dist_vp_to_line < vp_distance_threshold:
            longitudinal.append(seg)
        else:
            transversal.append(seg)
    return longitudinal, transversal


def _merge_segments_dbscan(
    segments: List[np.ndarray],
    is_longitudinal: bool = True,
    eps_rho: float = 20.0,
    eps_theta: float = 0.05,
    min_samples: int = 1,
) -> List[MergedLine]:
    """
    Cluster segments in (rho, theta) with DBSCAN, fit one line per cluster.
    """
    if not _SKLEARN_AVAILABLE or not segments:
        return []
    features = np.array([_segment_to_rho_theta(s) for s in segments])
    rho, theta = features[:, 0], features[:, 1]
    scale_theta = 500.0
    X = np.column_stack([rho, theta * scale_theta])
    eps = np.sqrt(eps_rho ** 2 + (eps_theta * scale_theta) ** 2)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    merged = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_segs = [segments[i] for i in range(len(segments)) if mask[i]]
        if not cluster_segs:
            continue
        a_sum, b_sum, c_sum = 0.0, 0.0, 0.0
        for s in cluster_segs:
            a, b, c = _segment_to_line_equation(s)
            a_sum += a
            b_sum += b
            c_sum += c
        n = len(cluster_segs)
        a, b, c = a_sum / n, b_sum / n, c_sum / n
        norm = np.sqrt(a * a + b * b)
        if norm < 1e-10:
            continue
        a, b, c = a / norm, b / norm, c / norm
        merged.append(MergedLine(a=a, b=b, c=c, is_longitudinal=is_longitudinal))
    return merged


def merged_lines_and_intersection_keypoints(
    segments: List[np.ndarray],
    image_width: int,
    image_height: int,
    min_distance_between_keypoints: float = 15.0,
    extend_bounds_factor: float = 1.5,
) -> Tuple[List[MergedLine], List[MergedLine], List[Tuple[float, float]]]:
    """
    From Hough segments: estimate VP, classify longitudinal/transversal,
    merge each group with DBSCAN, compute intersections between merged longitudinal
    and merged transversal lines. Returns validated image keypoints.

    Returns:
        merged_longitudinal: list of MergedLine (lengthwise)
        merged_transversal: list of MergedLine (widthwise)
        intersection_points: list of (x, y) image points (validated)
    """
    if segments is None or len(segments) < 2:
        return [], [], []
    lines = [s for s in segments if s is not None and len(s) > 0]
    if len(lines) < 2:
        return [], [], []
    w, h = image_width, image_height

    vp = _estimate_vanishing_point_ransac(lines, w, h)
    if vp is None:
        return [], [], []

    long_segs, trans_segs = _classify_longitudinal_transversal(lines, vp, w, h)

    merged_long = _merge_segments_dbscan(long_segs, is_longitudinal=True) if long_segs else []
    merged_trans = _merge_segments_dbscan(trans_segs, is_longitudinal=False) if trans_segs else []

    intersection_points = []
    for l_line in merged_long:
        for t_line in merged_trans:
            pt = _line_intersection(l_line, t_line)
            if pt is None:
                continue
            ix, iy = pt
            # Validate: within extended image bounds
            margin_w = w * (extend_bounds_factor - 1.0) / 2
            margin_h = h * (extend_bounds_factor - 1.0) / 2
            if ix < -margin_w or ix > w + margin_w or iy < -margin_h or iy > h + margin_h:
                continue
            # Min distance from existing points
            too_close = False
            for (qx, qy) in intersection_points:
                if np.sqrt((ix - qx) ** 2 + (iy - qy) ** 2) < min_distance_between_keypoints:
                    too_close = True
                    break
            if not too_close:
                intersection_points.append((float(ix), float(iy)))

    return merged_long, merged_trans, intersection_points
