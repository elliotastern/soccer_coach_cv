"""Match 3 camera mosaic / pitch-ordered stitch views for Phase 1 review."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_GS = ROOT / "scripts" / "gold_set"
if str(_GS) not in sys.path:
    sys.path.insert(0, str(_GS))

from raw_cam_id import cam_id_from_raw_name, load_match_raw  # noqa: E402

from src.mapping.match3_xy import (  # noqa: E402
    apply_H,
    calib_undistort_params,
    load_calib,
    map_ball_box,
    scale_px,
    undistort_px,
)
from src.review.frame_sync import draw_det_boxes, keep_top1_ball  # noqa: E402
from src.review.pitch1_panel import PITCH_LEN_M, PITCH_WID_M  # noqa: E402

# Coach mosaic — compass rotated 90° CW from north-up (+y left on pitch):
#   Top: Left touchline   Bottom: Right touchline
#   Left side: South      Right side: North
#   Grid: top P10|P9 (left sideline) · bottom P7|P8 (right sideline)
QUAD_GRID = [
    ["P10", "P9"],
    ["P7", "P8"],
]
QUAD_ROTATE_180 = frozenset({"P10", "P9"})
MOSAIC_MAP_ORIENT = "cw90"
MOSAIC_COMPASS_TOP = "Left"
MOSAIC_COMPASS_BOTTOM = "Right"
MOSAIC_COMPASS_WEST = "South"
MOSAIC_COMPASS_EAST = "North"
MOSAIC_SIDE_W = 34
MOSAIC_BAN_H = 26
MOSAIC_NORTH_H = 16
MOSAIC_SOUTH_H = 16
MOSAIC_CAM_H = 14
MOSAIC_CHROME_BG = (28, 42, 32)
MOSAIC_GAP_GREEN = (40, 90, 50)


def mosaic_grid_size(tile_w: int, tile_h: int, gap: int = 2) -> tuple[int, int]:
    """Video grid px (w, h) inside mosaic left/right chrome."""
    return 2 * tile_w + gap, 2 * tile_h + gap


def mosaic_total_width(tile_w: int, gap: int = 2) -> int:
    grid_w, _ = mosaic_grid_size(tile_w, 0, gap)
    return MOSAIC_SIDE_W + grid_w + MOSAIC_SIDE_W


def pitch_stack_metrics(
    grid_w: int,
    grid_h: int,
    *,
    drop_top: bool = True,
    scale: float = 0.46,
    map_orient: str = MOSAIC_MAP_ORIENT,
) -> dict[str, int]:
    """Pitch band below mosaic — cw90: south left, north right, left top, right bottom."""
    from src.review.pitch1_panel import (
        PITCH_LEN_M,
        PITCH_WID_M,
        _TIGHT_NORTH_H,
        _TIGHT_SOUTH_H,
        _TIGHT_STAT_H,
    )

    if map_orient == "cw90":
        field_w = max(96, int(round(grid_w * scale)))
        field_h = max(48, int(round(field_w * PITCH_WID_M / PITCH_LEN_M)))
    else:
        field_h = max(72, int(round(grid_h * scale)))
        field_w = max(48, int(round(field_h * PITCH_WID_M / PITCH_LEN_M)))
    top_h = 0 if drop_top else _TIGHT_NORTH_H
    panel_h = _TIGHT_STAT_H + top_h + field_h + _TIGHT_SOUTH_H
    panel_w = MOSAIC_SIDE_W + grid_w + MOSAIC_SIDE_W
    return {
        "panel_w": panel_w,
        "panel_h": panel_h,
        "field_w": field_w,
        "field_h": field_h,
        "band_w": grid_w,
        "map_orient": map_orient,
    }


def pitch_panel_height_for_stack(grid_w: int, drop_north: bool = True) -> int:
    """Legacy helper — use pitch_stack_metrics when grid_h is known."""
    from src.review.pitch1_panel import PITCH_LEN_M, PITCH_WID_M, _TIGHT_SOUTH_H, _TIGHT_STAT_H

    field_h = int(round(grid_w * PITCH_WID_M / PITCH_LEN_M))
    north_h = 0 if drop_north else MOSAIC_NORTH_H
    return _TIGHT_STAT_H + north_h + field_h + _TIGHT_SOUTH_H


def compose_coach_stack(
    mosaic: np.ndarray,
    pitch: np.ndarray,
    *,
    connect: bool = True,
) -> np.ndarray:
    """Stack mosaic over pitch; trim duplicate N/S chrome and align widths."""
    if pitch.shape[1] != mosaic.shape[1]:
        pitch = cv2.resize(
            pitch,
            (mosaic.shape[1], int(pitch.shape[0] * mosaic.shape[1] / pitch.shape[1])),
        )
    top = mosaic
    if connect and top.shape[0] > MOSAIC_SOUTH_H:
        top = top[:-MOSAIC_SOUTH_H, :]
    parts = [top]
    if connect:
        seam = np.zeros((3, mosaic.shape[1], 3), dtype=np.uint8)
        seam[:] = MOSAIC_GAP_GREEN
        cv2.line(
            seam, (MOSAIC_SIDE_W, 1), (mosaic.shape[1] - MOSAIC_SIDE_W, 1),
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        parts.append(seam)
    parts.append(pitch)
    return np.vstack(parts)


ENDS = ["P1", "P6"]  # south / north
GOALS = ["P_Goal1", "P_Goal2"]
ALL_CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]

VIEW_OPTIONS = [
    "Whole pitch (Left top · South left · P10|P9 / P7|P8)",
    "P1 + P6 (ends)",
    "Goals (Goal1 + Goal2)",
    "Best camera (ball)",
] + [f"Only {c}" for c in ALL_CAMS]

# Corner names a coach understands
COACH_CORNER = {
    "P10": "South · left · P10 (180°)",
    "P9": "North · left · P9 (180°)",
    "P7": "South · right · P7",
    "P8": "North · right · P8",
}

DetectFn = Callable[[str, np.ndarray], list]


def match3_videos(repo_root: Path) -> dict[str, Path]:
    folder = repo_root / "data/raw/Match 3"
    if not folder.is_dir():
        return {}
    try:
        return load_match_raw(folder)
    except Exception:
        out = {}
        for p in folder.glob("*.mp4"):
            if p.name.startswith("._"):
                continue
            try:
                out[cam_id_from_raw_name(p.name)] = p
            except ValueError:
                continue
        return out


def read_frame_bgr(video: Path, frame_id: int) -> Optional[np.ndarray]:
    """Read one mosaic tile frame; retry transient USB/LaCie EIO, never raise."""
    from src.review.io_retry import call_with_io_retry, is_transient_io

    def _once():
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return None
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fid = int(frame_id)
            if n > 0:
                fid = min(max(0, fid), n - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
        finally:
            cap.release()
        return frame if ok else None

    try:
        return call_with_io_retry(_once, tries=4, label=f"mosaic:{video.name}")
    except Exception as exc:  # noqa: BLE001 — tile soft-fail
        if is_transient_io(exc):
            return None
        return None


def _letterbox_meta(frame: np.ndarray, tw: int, th: int) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:] = (24, 28, 26)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def _letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    return _letterbox_meta(frame, tw, th)[0]


def _scale_dets(dets: list, scale: float, x0: int, y0: int) -> list:
    from src.state.types import Detection

    out = []
    for d in dets:
        x, y, w, h = [float(v) for v in d.bbox]
        out.append(
            Detection(
                int(d.class_id),
                float(d.confidence),
                (x * scale + x0, y * scale + y0, w * scale, h * scale),
                getattr(d, "class_name", "") or "",
            )
        )
    return out


def _label_tile(tile: np.ndarray, text: str, missing: bool = False) -> np.ndarray:
    """Error/missing tile only — normal cells use header labels, not on-video overlay."""
    out = tile.copy()
    if missing:
        color = (0, 165, 255)
        cv2.putText(
            out, text, (10, out.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )
        cv2.putText(
            out, "NO CAMERA", (10, out.shape[0] // 2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 200), 2,
        )
    return out


def _is_ball_det(det) -> bool:
    name = str(getattr(det, "class_name", "") or "").lower()
    return name == "ball" or int(getattr(det, "class_id", -1)) == 1


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _filter_coach_dets(
    dets: list,
    calib: dict | None,
    frame_wh: tuple[int, int],
    *,
    already_defished: bool,
) -> list:
    """Coach video boxes: keep solid player dets; balls still pitch-gated.

    Players are drawn on every camera tile when conf/size pass ``player_det_ok``.
    Pitch 1 mapping still drops unmappable feet in ``fuse_live_dets_for_pitch``.
    """
    if not dets:
        return []
    from src.review.multicam_fuse import player_det_ok

    # Video: show bodies on all cams (do not hide boxes that fail H/hull)
    players = [d for d in dets if not _is_ball_det(d) and player_det_ok(d)]
    out = list(players)
    balls = [d for d in dets if _is_ball_det(d)]
    kept_balls = []
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    edge_x, edge_y = 0.06 * fw, 0.06 * fh
    for d in balls:
        conf = float(d.confidence)
        # Review coverage: allow weaker balls if they still map onto Pitch 1
        if conf < 0.18:
            continue
        bx, by, bw, bh = [float(v) for v in d.bbox]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        near_edge = cx < edge_x or cy < edge_y or cx > fw - edge_x or cy > fh - edge_y
        # Edge FPs common; keep only if fairly confident
        if near_edge and conf < 0.42:
            continue
        if any(_bbox_iou(d.bbox, p.bbox) >= 0.12 for p in players):
            continue
        if calib is not None:
            mapped = map_ball_box(
                calib,
                d.bbox,
                conf,
                frame_wh=frame_wh,
                apply_undistort=not already_defished,
            )
            if mapped is None:
                continue
        kept_balls.append(d)
    out.extend(keep_top1_ball(kept_balls))
    return out


def _annotate(
    frame: np.ndarray,
    dets: list | None,
    coach_simple: bool = True,
    draw_labels: bool = True,
    min_ball_side: int = 32,
    min_ball_conf: float = 0.30,
) -> np.ndarray:
    if not dets:
        return frame
    dets = keep_top1_ball(list(dets))
    if not coach_simple:
        return draw_det_boxes(frame, dets)
    # Coach mode: thick boxes; ball gets a visible min-size box centered on the det
    vis = frame.copy()
    for det in dets:
        is_ball = getattr(det, "class_name", "") == "ball" or int(det.class_id) == 1
        if is_ball and float(det.confidence) < float(min_ball_conf):
            continue
        x, y, w, h = [float(v) for v in det.bbox]
        if is_ball:
            cx, cy = x + w / 2.0, y + h / 2.0
            side = max(float(min_ball_side), w, h)
            # Slight pad so the orange ring clearly sits on the ball
            side = max(side * 1.15, float(min_ball_side))
            x, y, w, h = cx - side / 2.0, cy - side / 2.0, side, side
        xi, yi = int(round(x)), int(round(y))
        wi, hi = max(1, int(round(w))), max(1, int(round(h)))
        # Clip to frame
        xi = max(0, min(xi, vis.shape[1] - 1))
        yi = max(0, min(yi, vis.shape[0] - 1))
        wi = max(1, min(wi, vis.shape[1] - xi))
        hi = max(1, min(hi, vis.shape[0] - yi))
        color = (0, 165, 255) if is_ball else (0, 220, 0)
        thick = 5 if is_ball else 4
        cv2.rectangle(vis, (xi, yi), (xi + wi, yi + hi), color, thick)
        if draw_labels and is_ball:
            label = "BALL"
            ty = max(32, yi - 8)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.rectangle(vis, (xi, ty - th - 6), (xi + tw + 8, ty + 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (xi + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    return vis


def _rotate_dets_180(dets: list, w: int, h: int) -> list:
    from src.state.types import Detection

    out = []
    for d in dets:
        x, y, bw, bh = [float(v) for v in d.bbox]
        out.append(
            Detection(
                int(d.class_id),
                float(d.confidence),
                (w - x - bw, h - y - bh, bw, bh),
                getattr(d, "class_name", "") or "",
            )
        )
    return out


def _remap_bbox_undistort(
    bbox,
    calib: dict,
    wh,
    alpha_override: float | None = None,
) -> tuple[float, float, float, float]:
    """Undistort AABB corners with the SAME alpha as undistort_bgr for this tile."""
    params = calib_undistort_params(calib)
    if not params:
        return tuple(float(v) for v in bbox)
    if alpha_override is not None:
        params = {**params, "alpha": float(alpha_override)}
    x, y, bw, bh = [float(v) for v in bbox]
    cw, ch = float(wh[0]), float(wh[1])
    corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    pts = [undistort_px(u, v, cw, ch, params) for u, v in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    return (x0, y0, max(xs) - x0, max(ys) - y0)


# Use the same Brown alpha as locked landmarks so remap ↔ pixels match.
MOSAIC_DEFISH_ALPHA = None  # None → calib undistort.alpha


def _tile(
    videos: dict[str, Path],
    cam: str,
    frame_id: int,
    tw: int,
    th: int,
    dets_by_cam: dict[str, list] | None = None,
    detect_fn: DetectFn | None = None,
    rotate_180: bool = False,
    apply_defish: bool = True,
) -> np.ndarray:
    """One mosaic cell.

    Box lock rule: draw boxes only after the final pixel warp for that tile.
    - defish off: detect/raw dets → rotate → draw
    - defish on: warp image first, then detect_fn on warped pixels (or skip boxes)
    """
    from src.state.types import Detection

    path = videos.get(cam)
    if path is None or not path.is_file():
        blank = np.zeros((th, tw, 3), dtype=np.uint8)
        blank[:] = (40, 40, 40)
        return _label_tile(blank, f"{COACH_CORNER.get(cam, cam)} (missing)", missing=True)
    frame = read_frame_bgr(path, frame_id)
    if frame is None:
        blank = np.zeros((th, tw, 3), dtype=np.uint8)
        blank[:] = (40, 40, 40)
        return _label_tile(blank, f"{COACH_CORNER.get(cam, cam)} (no frame)", missing=True)

    work = frame
    # Always load calib when present so ball boxes can be pitch-gated
    calib = load_calib(cam)
    do_defish = bool(
        apply_defish and calib is not None and calib_undistort_params(calib)
    )

    if do_defish:
        cw, ch = [int(v) for v in (calib.get("image_wh") or [work.shape[1], work.shape[0]])]
        if work.shape[1] != cw or work.shape[0] != ch:
            work = cv2.resize(work, (cw, ch))
        params = calib_undistort_params(calib) or {}
        alpha = (
            float(MOSAIC_DEFISH_ALPHA)
            if MOSAIC_DEFISH_ALPHA is not None
            else float(params.get("alpha", 0.8))
        )
        work = undistort_bgr(work, calib, alpha_override=alpha)
        # Prefer live detect on defished pixels; else reuse bag (already detect-after-defish)
        if detect_fn is not None:
            dets = list(detect_fn(cam, work) or [])
        elif dets_by_cam is not None and cam in dets_by_cam:
            dets = list(dets_by_cam[cam] or [])
        else:
            dets = None
    else:
        dets = None
        if dets_by_cam is not None and cam in dets_by_cam:
            dets = list(dets_by_cam[cam])
        elif detect_fn is not None:
            dets = list(detect_fn(cam, work) or [])

    if dets:
        # Skip re-filter if bag already pruned (has __wh from ensure pass)
        already_pruned = bool(
            dets_by_cam is not None and f"{cam}__wh" in dets_by_cam and detect_fn is None
        )
        if not already_pruned:
            dets = _filter_coach_dets(
                list(dets),
                calib,
                (int(work.shape[1]), int(work.shape[0])),
                already_defished=do_defish,
            )

    # Keep pre-rotate dets for Pitch 1 mapping (rotation is display-only)
    if dets_by_cam is not None and dets:
        dets_by_cam[cam] = list(dets)
        dets_by_cam[f"{cam}__wh"] = (int(work.shape[1]), int(work.shape[0]))
        dets_by_cam[f"{cam}__bgr"] = work

    if rotate_180:
        h, w = work.shape[:2]
        work = cv2.rotate(work, cv2.ROTATE_180)
        if dets:
            dets = _rotate_dets_180(dets, w, h)

    tile, scale, x0, y0 = _letterbox_meta(work, tw, th)
    if dets:
        tile = _annotate(
            tile,
            _scale_dets(dets, scale, x0, y0),
            coach_simple=True,
            min_ball_conf=0.18,
        )
    return tile


def _mosaic_compass_row(
    width: int, height: int, text: str, bg: tuple[int, int, int] = (28, 42, 32),
) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = bg
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.46, 1
    (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
    cv2.putText(
        strip, text, ((width - tw) // 2, (height + th) // 2 - 1),
        font, fs, (210, 210, 210), thick, cv2.LINE_AA,
    )
    return strip


def _mosaic_side_label(width: int, height: int, text: str) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (28, 42, 32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.44, 1
    (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
    cv2.putText(
        strip, text, (max(2, (width - tw) // 2), (height + th) // 2),
        font, fs, (210, 210, 210), thick, cv2.LINE_AA,
    )
    return strip


def _mosaic_cam_pair_row(
    width: int,
    height: int,
    left_cam: str,
    right_cam: str,
    left_tile_w: int,
    gap: int,
) -> np.ndarray:
    """Camera ids centered over each mosaic column (outside the video tiles)."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (28, 42, 32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.44, 1
    split = left_tile_w + gap
    for cam, cx in ((left_cam, left_tile_w // 2), (right_cam, split + (width - split) // 2)):
        (tw, th), _ = cv2.getTextSize(cam, font, fs, thick)
        cv2.putText(
            strip, cam, (max(0, cx - tw // 2), (height + th) // 2 - 1),
            font, fs, (170, 220, 255), thick, cv2.LINE_AA,
        )
    return strip


def _mosaic_banner(col_w: int, ban_h: int = 26) -> np.ndarray:
    strip = np.zeros((ban_h, col_w, 3), dtype=np.uint8)
    strip[:] = (28, 42, 32)
    cv2.putText(
        strip, "WHOLE PITCH", (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        "Green=player  Orange=ball",
        (168, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA,
    )
    return strip


def mosaic_grid(tiles: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    rows = []
    for row in tiles:
        h = max(t.shape[0] for t in row)
        w = sum(t.shape[1] for t in row) + gap * (len(row) - 1)
        strip = np.zeros((h, w, 3), dtype=np.uint8)
        strip[:] = (40, 90, 50)
        x = 0
        for t in row:
            strip[0 : t.shape[0], x : x + t.shape[1]] = t
            x += t.shape[1] + gap
        rows.append(strip)
    max_w = max(r.shape[1] for r in rows)
    out_h = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.zeros((out_h, max_w, 3), dtype=np.uint8)
    out[:] = (40, 90, 50)
    y = 0
    for r in rows:
        out[y : y + r.shape[0], 0 : r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def best_cam_for_frame(output_root: Path, frame_id: int, fallback: str = "P10") -> str:
    """Cam with highest ball confidence at frame_id among sibling exports."""
    from src.review.multicam_fuse import discover_cam_frame_csvs, load_cam_tables

    tables = load_cam_tables(discover_cam_frame_csvs(Path(output_root)))
    best_cam, best_conf = fallback, -1.0
    for cam, df in tables.items():
        balls = df[(df["frame_id"] == int(frame_id)) & (df["Player_ID"] == -1)]
        if len(balls) == 0:
            continue
        conf = float(balls.iloc[0]["confidence"]) if "confidence" in balls.columns else 0.5
        if conf > best_conf:
            best_conf = conf
            best_cam = cam
    return best_cam


def cams_for_view(view: str, output_root: Path, frame_id: int, primary_cam: str) -> list[str]:
    if view.startswith("4 quads") or view.startswith("Whole pitch"):
        return ["P10", "P9", "P7", "P8"]
    if view.startswith("Best camera"):
        # Best-ball is a ball pick on the whole-pitch mosaic — not a single-cam crop.
        return ["P10", "P9", "P7", "P8"]
    if view.startswith("P1 + P6"):
        return list(ENDS)
    if view.startswith("Goals"):
        return list(GOALS)
    if view.startswith("Only "):
        return [view.replace("Only ", "").strip()]
    return [primary_cam]


def _canvas_layout(panel_w: int, panel_h: int, margin: int = 28):
    avail_w = panel_w - 2 * margin
    avail_h = panel_h - 2 * margin
    field_aspect = PITCH_WID_M / PITCH_LEN_M
    if avail_w / max(avail_h, 1) > field_aspect:
        ph = avail_h
        pw = int(round(ph * field_aspect))
    else:
        pw = avail_w
        ph = int(round(pw / field_aspect))
    x0 = margin + (avail_w - pw) // 2
    y0 = margin + (avail_h - ph) // 2
    return x0, y0, pw, ph


def meters_to_canvas_matrix(panel_w: int, panel_h: int, margin: int = 28) -> np.ndarray:
    """3×3: Pitch 1 meters (xm, ym, 1) → canvas pixels (north up, +y left)."""
    x0, y0, pw, ph = _canvas_layout(panel_w, panel_h, margin)
    c1 = pw / PITCH_WID_M
    d2 = ph / PITCH_LEN_M
    c0 = x0 + (PITCH_WID_M / 2.0) * c1
    d0 = y0 + (PITCH_LEN_M / 2.0) * d2
    return np.array(
        [[0.0, -c1, c0], [-d2, 0.0, d0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def undistort_bgr(
    frame: np.ndarray, calib: dict, alpha_override: float | None = None
) -> np.ndarray:
    """Full-frame Brown undistort matching match3_xy / landmark stills."""
    params = calib_undistort_params(calib)
    if not params:
        return frame
    h, w = frame.shape[:2]
    k_mat = np.array(
        [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(
        [
            float(params["k1"]),
            float(params["k2"]),
            float(params["p1"]),
            float(params["p2"]),
            0.0,
        ],
        dtype=np.float64,
    )
    alpha = float(params["alpha"]) if alpha_override is None else float(alpha_override)
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        k_mat, dist, (w, h), alpha, (w, h)
    )
    return cv2.undistort(frame, k_mat, dist, None, new_k)


def prepare_cam_frame(frame: np.ndarray, calib: dict) -> np.ndarray:
    """Resize to calib WH, then undistort so pixels match H."""
    cw, ch = [int(v) for v in (calib.get("image_wh") or [frame.shape[1], frame.shape[0]])]
    if frame.shape[1] != cw or frame.shape[0] != ch:
        frame = cv2.resize(frame, (cw, ch))
    return undistort_bgr(frame, calib)


def det_box_to_canvas(
    bbox,
    calib: dict,
    frame_wh,
    panel_w: int,
    panel_h: int,
) -> list[tuple[int, int]]:
    """Project axis-aligned image bbox corners → Pitch 1 canvas polygon."""
    x, y, bw, bh = [float(v) for v in bbox]
    corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    calib_wh = calib.get("image_wh") or frame_wh
    cw, ch = float(calib_wh[0]), float(calib_wh[1])
    params = calib_undistort_params(calib)
    M = meters_to_canvas_matrix(panel_w, panel_h)
    H = np.asarray(calib["H"], dtype=np.float64)
    out: list[tuple[int, int]] = []
    for u, v in corners:
        px, py = scale_px(u, v, frame_wh, calib_wh)
        if params:
            px, py = undistort_px(px, py, cw, ch, params)
        xy = apply_H(H, px, py)
        if xy is None:
            continue
        vec = M @ np.array([xy[0], xy[1], 1.0], dtype=np.float64)
        if abs(vec[2]) < 1e-8:
            continue
        out.append((int(round(vec[0] / vec[2])), int(round(vec[1] / vec[2]))))
    return out


def draw_dets_on_pitch_canvas(
    canvas: np.ndarray,
    dets: list,
    calib: dict,
    frame_wh,
) -> np.ndarray:
    """Draw player/ball boxes after H-warp (colors stay pure for coach UX + eng-loop)."""
    if not dets:
        return canvas
    h, w = canvas.shape[:2]
    vis = canvas
    for det in keep_top1_ball(list(dets)):
        poly = det_box_to_canvas(det.bbox, calib, frame_wh, w, h)
        if len(poly) < 3:
            continue
        is_ball = getattr(det, "class_name", "") == "ball" or int(det.class_id) == 1
        color = (0, 165, 255) if is_ball else (0, 220, 0)
        thick = 4 if is_ball else 3
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(vis, [pts], True, color, thick)
        # AABB label
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        label = f"{'BALL' if is_ball else 'P'} {float(det.confidence):.2f}"
        ty = max(28, y0 - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x0, ty - th - 4), (x0 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(vis, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def warp_cam_to_pitch(
    frame_bgr: np.ndarray,
    calib: dict,
    panel_w: int,
    panel_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp (defished) cam frame onto Pitch 1 canvas. Returns (bgr, mask)."""
    prepared = prepare_cam_frame(frame_bgr, calib)
    M = meters_to_canvas_matrix(panel_w, panel_h)
    H = np.asarray(calib["H"], dtype=np.float64)
    canvas_h = M @ H
    warped = cv2.warpPerspective(
        prepared, canvas_h, (panel_w, panel_h), flags=cv2.INTER_LINEAR
    )
    ones = np.full(prepared.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(ones, canvas_h, (panel_w, panel_h), flags=cv2.INTER_NEAREST)
    return warped, mask


def _pitch_base(panel_w: int, panel_h: int) -> np.ndarray:
    """Striped Pitch 1 underlay (same north-up layout as pitch1_panel)."""
    vis = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    vis[:] = (32, 48, 36)
    x0, y0, pw, ph = _canvas_layout(panel_w, panel_h)
    for i in range(10):
        ya = y0 + int(i * ph / 10)
        yb = y0 + int((i + 1) * ph / 10)
        color = (55, 130, 55) if i % 2 == 0 else (48, 118, 48)
        cv2.rectangle(vis, (x0, ya), (x0 + pw, yb), color, -1)
    # outline + halfway for orientation
    cv2.rectangle(vis, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (240, 240, 240), 2)
    mid_y = y0 + ph // 2
    cv2.line(vis, (x0, mid_y), (x0 + pw - 1, mid_y), (240, 240, 240), 2)
    cv2.putText(vis, "N", (x0 + pw // 2 - 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, "S", (x0 + pw // 2 - 8, y0 + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def stitch_quads_pitch_order(
    videos: dict[str, Path],
    frame_id: int,
    panel_w: int = 960,
    panel_h: int = 1480,
    dets_by_cam: dict[str, list] | None = None,
    detect_fn: DetectFn | None = None,
) -> np.ndarray:
    """H-warp P8/P9/P10/P7 onto one Pitch 1 canvas (north up, pitch order)."""
    base = _pitch_base(panel_w, panel_h).astype(np.float32)
    acc = np.zeros((panel_h, panel_w, 3), dtype=np.float32)
    wsum = np.zeros((panel_h, panel_w), dtype=np.float32)
    pending_boxes: list[tuple[list, dict, tuple[int, int]]] = []
    for cam in ["P10", "P9", "P7", "P8"]:
        path = videos.get(cam)
        calib = load_calib(cam)
        if path is None or not path.is_file() or calib is None:
            continue
        frame = read_frame_bgr(path, frame_id)
        if frame is None:
            continue
        dets = None
        if dets_by_cam is not None and cam in dets_by_cam:
            dets = dets_by_cam[cam]
        elif detect_fn is not None:
            dets = detect_fn(cam, frame)
        if dets:
            pending_boxes.append((list(dets), calib, (frame.shape[1], frame.shape[0])))
        # Warp raw imagery (boxes drawn after blend so colors stay coach-visible)
        warped, mask = warp_cam_to_pitch(frame, calib, panel_w, panel_h)
        m = (mask.astype(np.float32) / 255.0)
        acc += warped.astype(np.float32) * m[..., None]
        wsum += m
    out = base.copy()
    hit = wsum > 0.05
    if np.any(hit):
        blended = acc / np.maximum(wsum[..., None], 1e-6)
        out[hit] = 0.82 * blended[hit] + 0.18 * base[hit]
    label = out.astype(np.uint8)
    for dets, calib, fwh in pending_boxes:
        label = draw_dets_on_pitch_canvas(label, dets, calib, fwh)
    cv2.rectangle(label, (0, 0), (panel_w - 1, 40), (0, 0, 0), -1)
    cv2.putText(
        label,
        "4 quads · pitch-ordered H-stitch (N up)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return label


def _iter_cam_dets(dets_by_cam: dict):
    """Yield (cam_id, det_list) — skip meta keys like ``__wh`` / ``__bgr`` frames."""
    for cam, dets in list(dets_by_cam.items()):
        if not isinstance(cam, str):
            continue
        if cam.endswith("__wh") or cam.endswith("__bgr"):
            continue
        if not isinstance(dets, (list, tuple)):
            continue
        yield cam, dets


def _keep_single_mosaic_ball(dets_by_cam: dict, *, apply_defish: bool = True) -> str | None:
    """One pitch → one orange ball. Keep highest-conf mapped ball; strip others.

    Matches the Pitch 1 panel (single yellow ball). Returns winning cam id or None.
    """
    if not dets_by_cam:
        return None
    candidates = []
    for cam, dets in _iter_cam_dets(dets_by_cam):
        if not dets:
            continue
        calib = load_calib(cam)
        wh = dets_by_cam.get(f"{cam}__wh")
        already = bool(
            apply_defish and calib is not None and calib_undistort_params(calib)
        )
        for d in dets:
            if not _is_ball_det(d):
                continue
            mapped = None
            if calib is not None and wh is not None:
                mapped = map_ball_box(
                    calib,
                    d.bbox,
                    float(d.confidence),
                    frame_wh=wh,
                    apply_undistort=not already,
                )
            candidates.append((float(d.confidence), cam, mapped is not None))
    if not candidates:
        return None
    mapped_ok = [c for c in candidates if c[2]]
    pool = mapped_ok if mapped_ok else candidates
    best_cam = max(pool, key=lambda c: c[0])[1]
    for cam, dets in _iter_cam_dets(dets_by_cam):
        if not dets:
            continue
        if cam == best_cam:
            dets_by_cam[cam] = keep_top1_ball(list(dets))
        else:
            dets_by_cam[cam] = [d for d in dets if not _is_ball_det(d)]
    return best_cam


def _ensure_cam_dets(
    videos: dict[str, Path],
    cam: str,
    frame_id: int,
    dets_by_cam: dict,
    detect_fn: DetectFn | None,
    apply_defish: bool,
) -> None:
    """Fill dets_by_cam[cam] using the same detect-after-defish path as _tile."""
    if detect_fn is None:
        return
    path = videos.get(cam)
    if path is None or not path.is_file():
        return
    frame = read_frame_bgr(path, frame_id)
    if frame is None:
        return
    work = frame
    calib = load_calib(cam)
    do_defish = bool(
        apply_defish and calib is not None and calib_undistort_params(calib)
    )
    if do_defish:
        cw, ch = [int(v) for v in (calib.get("image_wh") or [work.shape[1], work.shape[0]])]
        if work.shape[1] != cw or work.shape[0] != ch:
            work = cv2.resize(work, (cw, ch))
        params = calib_undistort_params(calib) or {}
        alpha = (
            float(MOSAIC_DEFISH_ALPHA)
            if MOSAIC_DEFISH_ALPHA is not None
            else float(params.get("alpha", 0.8))
        )
        work = undistort_bgr(work, calib, alpha_override=alpha)
    dets = list(detect_fn(cam, work) or [])
    dets = _filter_coach_dets(
        dets,
        calib,
        (int(work.shape[1]), int(work.shape[0])),
        already_defished=do_defish,
    )
    dets_by_cam[cam] = dets
    dets_by_cam[f"{cam}__wh"] = (int(work.shape[1]), int(work.shape[0]))
    # Detect-space BGR for live jersey team labeling on Pitch 1
    dets_by_cam[f"{cam}__bgr"] = work


def mosaic_quads_coach(
    videos: dict[str, Path],
    frame_id: int,
    tile_w: int = 960,
    tile_h: int = 540,
    dets_by_cam: dict[str, list] | None = None,
    detect_fn: DetectFn | None = None,
    apply_defish: bool = True,
) -> np.ndarray:
    """Coach mosaic: whole pitch at a glance. Plain labels + PLAYER/BALL boxes."""
    bag: dict = dict(dets_by_cam or {})
    # Pass 1: detect all quads into bag
    if detect_fn is not None:
        for cam in ["P10", "P9", "P7", "P8"]:
            _ensure_cam_dets(videos, cam, frame_id, bag, detect_fn, apply_defish)
        # Pass 1b: one ball for the whole pitch (matches Pitch 1 panel)
        _keep_single_mosaic_ball(bag, apply_defish=apply_defish)
        if dets_by_cam is not None:
            dets_by_cam.clear()
            dets_by_cam.update(bag)

    def cell(cam: str) -> np.ndarray:
        return _tile(
            videos,
            cam,
            frame_id,
            tile_w,
            tile_h,
            dets_by_cam=bag,
            detect_fn=None,  # already filled + single-ball pruned
            rotate_180=(cam in QUAD_ROTATE_180),
            apply_defish=apply_defish,
        )

    row_top = [cell(c) for c in QUAD_GRID[0]]
    row_bot = [cell(c) for c in QUAD_GRID[1]]
    gap = 2
    mosaic_body = mosaic_grid([row_top, row_bot], gap=gap)
    grid_w, grid_h = mosaic_body.shape[1], mosaic_body.shape[0]
    side_w = MOSAIC_SIDE_W
    north_h = MOSAIC_NORTH_H
    south_h = MOSAIC_SOUTH_H
    cam_h = MOSAIC_CAM_H
    ban_h = MOSAIC_BAN_H
    tile_w = row_top[0].shape[1]
    total_w = side_w + grid_w + side_w
    total_h = ban_h + north_h + cam_h + grid_h + cam_h + south_h
    out = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    out[:] = MOSAIC_CHROME_BG
    out[0:ban_h, 0:total_w] = _mosaic_banner(total_w, ban_h)
    y = ban_h
    out[y : y + north_h, 0:total_w] = _mosaic_compass_row(total_w, north_h, MOSAIC_COMPASS_TOP)
    y += north_h
    out[y : y + cam_h, side_w : side_w + grid_w] = _mosaic_cam_pair_row(
        grid_w, cam_h, QUAD_GRID[0][0], QUAD_GRID[0][1], tile_w, gap,
    )
    y += cam_h
    out[y : y + grid_h, side_w : side_w + grid_w] = mosaic_body
    out[y : y + grid_h, 0:side_w] = _mosaic_side_label(side_w, grid_h, MOSAIC_COMPASS_WEST)
    out[y : y + grid_h, side_w + grid_w : total_w] = _mosaic_side_label(
        side_w, grid_h, MOSAIC_COMPASS_EAST,
    )
    y += grid_h
    out[y : y + cam_h, side_w : side_w + grid_w] = _mosaic_cam_pair_row(
        grid_w, cam_h, QUAD_GRID[1][0], QUAD_GRID[1][1], tile_w, gap,
    )
    y += cam_h
    out[y : y + south_h, 0:total_w] = _mosaic_compass_row(total_w, south_h, MOSAIC_COMPASS_BOTTOM)
    return out


def fill_quad_dets_for_pitch(
    videos: dict[str, Path],
    frame_id: int,
    dets_by_cam: dict,
    detect_fn: DetectFn | None,
    apply_defish: bool = True,
    *,
    single_ball: bool = True,
) -> dict:
    """Ensure P10/P9/P7/P8 player+ball dets are in ``dets_by_cam`` for Pitch 1 fuse.

    Used when the stage shows Best-ball / single cam so the pitch map still
    gets every mappable player across the quads (one orange ball if single_ball).
    """
    if detect_fn is None or dets_by_cam is None:
        return dets_by_cam or {}
    bag: dict = dict(dets_by_cam)
    for cam in ["P10", "P9", "P7", "P8"]:
        _ensure_cam_dets(videos, cam, frame_id, bag, detect_fn, apply_defish)
    if single_ball:
        _keep_single_mosaic_ball(bag, apply_defish=apply_defish)
    dets_by_cam.clear()
    dets_by_cam.update(bag)
    return dets_by_cam


def build_cam_view(
    repo_root: Path,
    view: str,
    frame_id: int,
    output_root: Path,
    primary_cam: str = "P10",
    tile_w: int = 640,
    tile_h: int = 360,
    dets_by_cam: dict[str, list] | None = None,
    detect_fn: DetectFn | None = None,
    stitch_w: int = 960,
    stitch_h: int = 1480,
    apply_defish: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return BGR view + list of cams used. Boxes drawn when dets/detect_fn given."""
    videos = match3_videos(repo_root)
    cams = cams_for_view(view, output_root, frame_id, primary_cam)

    if (
        view.startswith("4 quads")
        or view.startswith("Whole pitch")
        or view.startswith("Best camera")
    ):
        img = mosaic_quads_coach(
            videos,
            frame_id,
            tile_w=max(tile_w, 640),
            tile_h=max(tile_h, 360),
            dets_by_cam=dets_by_cam,
            detect_fn=detect_fn,
            apply_defish=apply_defish,
        )
        # Annotate which cam won the single-ball pick (Best-ball filter).
        if view.startswith("Best camera") and dets_by_cam is not None:
            win = None
            for c in ("P10", "P9", "P7", "P8"):
                dets = dets_by_cam.get(c) or []
                if any(_is_ball_det(d) for d in dets):
                    win = c
                    break
            tag = f"BEST BALL · {win or best_cam_for_frame(output_root, frame_id, primary_cam)}"
            cv2.rectangle(img, (0, img.shape[0] - 36), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(
                img,
                tag,
                (12, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        return img, cams

    if view.startswith("P1 + P6") or view.startswith("Goals"):
        pair = [
            _tile(
                videos, c, frame_id, tile_w, tile_h, dets_by_cam, detect_fn,
                apply_defish=apply_defish,
            )
            for c in cams
        ]
        return mosaic_grid([pair]), cams

    cam = cams[0]
    big_w, big_h = tile_w * 2, tile_h * 2
    return (
        _tile(
            videos, cam, frame_id, big_w, big_h, dets_by_cam, detect_fn,
            apply_defish=apply_defish,
        ),
        cams,
    )
