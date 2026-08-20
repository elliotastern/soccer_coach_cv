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
    scale_px,
    undistort_px,
)
from src.review.frame_sync import draw_det_boxes, keep_top1_ball  # noqa: E402
from src.review.pitch1_panel import PITCH_LEN_M, PITCH_WID_M  # noqa: E402

# Coach mosaic layout (screen positions):
#   P10 top-left (180°) | P9 top-right (180°)
#   P7  bottom-left     | P8 bottom-right
QUAD_GRID = [
    ["P10", "P9"],  # top (both rotated 180°)
    ["P7", "P8"],   # bottom
]
QUAD_ROTATE_180 = frozenset({"P9", "P10"})
ENDS = ["P1", "P6"]  # south / north
GOALS = ["P_Goal1", "P_Goal2"]
ALL_CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]

VIEW_OPTIONS = [
    "4 quads (whole pitch · N-up mosaic)",
    "P1 + P6 (ends)",
    "Goals (Goal1 + Goal2)",
    "Best camera (ball)",
] + [f"Only {c}" for c in ALL_CAMS]

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
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fid = int(frame_id)
    if n > 0:
        fid = min(max(0, fid), n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


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
    out = tile.copy()
    color = (0, 165, 255) if missing else (0, 255, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    if missing:
        cv2.putText(
            out, "NO FRAME", (10, out.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 200), 2,
        )
    return out


def _annotate(frame: np.ndarray, dets: list | None) -> np.ndarray:
    if not dets:
        return frame
    return draw_det_boxes(frame, keep_top1_ball(list(dets)))


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
    path = videos.get(cam)
    if path is None or not path.is_file():
        blank = np.zeros((th, tw, 3), dtype=np.uint8)
        blank[:] = (40, 40, 40)
        return _label_tile(blank, f"{cam} (missing video)", missing=True)
    frame = read_frame_bgr(path, frame_id)
    if frame is None:
        blank = np.zeros((th, tw, 3), dtype=np.uint8)
        blank[:] = (40, 40, 40)
        return _label_tile(blank, f"{cam} (no frame {frame_id})", missing=True)
    dets = None
    if dets_by_cam is not None and cam in dets_by_cam:
        dets = dets_by_cam[cam]
    elif detect_fn is not None:
        dets = detect_fn(cam, frame)

    # Detect on raw; draw on full-res; then Brown defish so coach sees straight lines
    # and boxes stay glued (undistort remaps the painted pixels).
    work = frame
    calib = load_calib(cam) if apply_defish else None
    did_defish = False
    if calib is not None and calib_undistort_params(calib):
        cw, ch = [int(v) for v in (calib.get("image_wh") or [work.shape[1], work.shape[0]])]
        if work.shape[1] != cw or work.shape[0] != ch:
            sx = cw / float(work.shape[1])
            sy = ch / float(work.shape[0])
            work = cv2.resize(work, (cw, ch))
            if dets:
                from src.state.types import Detection
                dets = [
                    Detection(
                        int(d.class_id),
                        float(d.confidence),
                        (
                            float(d.bbox[0]) * sx,
                            float(d.bbox[1]) * sy,
                            float(d.bbox[2]) * sx,
                            float(d.bbox[3]) * sy,
                        ),
                        getattr(d, "class_name", "") or "",
                    )
                    for d in dets
                ]
        if dets:
            work = _annotate(work, dets)
            dets = None  # already painted
        work = undistort_bgr(work, calib)
        did_defish = True

    tile, scale, x0, y0 = _letterbox_meta(work, tw, th)
    if dets:
        tile = _annotate(tile, _scale_dets(dets, scale, x0, y0))
    if rotate_180:
        tile = cv2.rotate(tile, cv2.ROTATE_180)
    bits = [cam]
    if did_defish:
        bits.append("defish")
    if rotate_180:
        bits.append("180°")
    return _label_tile(tile, " · ".join(bits))


def mosaic_grid(tiles: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    rows = []
    for row in tiles:
        h = max(t.shape[0] for t in row)
        w = sum(t.shape[1] for t in row) + gap * (len(row) - 1)
        strip = np.zeros((h, w, 3), dtype=np.uint8)
        strip[:] = (12, 12, 12)
        x = 0
        for t in row:
            strip[0 : t.shape[0], x : x + t.shape[1]] = t
            x += t.shape[1] + gap
        rows.append(strip)
    max_w = max(r.shape[1] for r in rows)
    out_h = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.zeros((out_h, max_w, 3), dtype=np.uint8)
    out[:] = (12, 12, 12)
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
    if view.startswith("4 quads"):
        return ["P10", "P9", "P7", "P8"]
    if view.startswith("P1 + P6"):
        return list(ENDS)
    if view.startswith("Goals"):
        return list(GOALS)
    if view.startswith("Best camera"):
        return [best_cam_for_frame(output_root, frame_id, fallback=primary_cam)]
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


def undistort_bgr(frame: np.ndarray, calib: dict) -> np.ndarray:
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
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        k_mat, dist, (w, h), float(params["alpha"]), (w, h)
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
    for cam in ["P8", "P9", "P10", "P7"]:
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


def mosaic_quads_coach(
    videos: dict[str, Path],
    frame_id: int,
    tile_w: int = 960,
    tile_h: int = 540,
    dets_by_cam: dict[str, list] | None = None,
    detect_fn: DetectFn | None = None,
    apply_defish: bool = True,
) -> np.ndarray:
    """Coach mosaic: P10|P9 on top (180°), P7|P8 on bottom. Boxes in image space."""
    def cell(cam: str) -> np.ndarray:
        return _tile(
            videos,
            cam,
            frame_id,
            tile_w,
            tile_h,
            dets_by_cam,
            detect_fn,
            rotate_180=(cam in QUAD_ROTATE_180),
            apply_defish=apply_defish,
        )

    grid = [
        [cell("P10"), cell("P9")],  # top
        [cell("P7"), cell("P8")],   # bottom — P7 BL, P8 BR
    ]
    mosaic = mosaic_grid(grid, gap=6)
    ban_h = 40
    foot_h = 28
    out = np.zeros((mosaic.shape[0] + ban_h + foot_h, mosaic.shape[1], 3), dtype=np.uint8)
    out[:] = (18, 18, 18)
    out[ban_h : ban_h + mosaic.shape[0]] = mosaic
    defish_note = " · DEFISHED P7–P10" if apply_defish else ""
    cv2.putText(
        out,
        f"TOP    P10 (180) | P9 (180)     mosaic{defish_note} · boxes on",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        out,
        "BOTTOM P7 (BL)   | P8 (BR)",
        (12, out.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    return out


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

    if view.startswith("4 quads"):
        img = mosaic_quads_coach(
            videos,
            frame_id,
            tile_w=max(tile_w, 800),
            tile_h=max(tile_h, 450),
            dets_by_cam=dets_by_cam,
            detect_fn=detect_fn,
            apply_defish=apply_defish,
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
