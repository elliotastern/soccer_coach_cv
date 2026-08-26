"""Streamlit — label fuse mosaic emits (stable frame scrub).

Usage: streamlit run apps/coach_emit_label_dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gold_set.emit_timeline_layout import (  # noqa: E402
    build_timeline_html,
    layout_timeline_events,
    timeline_html_height,
    timeline_events_near as _timeline_events_near,
)
from src.review.cam_mosaic import (  # noqa: E402
    MOSAIC_BAN_H,
    MOSAIC_CAM_H,
    MOSAIC_NORTH_H,
    MOSAIC_SOUTH_H,
    MOSAIC_SIDE_W,
    mosaic_grid_size,
)
from src.review.multicam_fuse import discover_cam_frame_csvs  # noqa: E402
from src.review.pitch1_panel import PITCH_LEN_M, PITCH_WID_M  # noqa: E402

DEFAULT_VIDEO = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_first_90s.mp4"
DEFAULT_EMITS = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json"
DEFAULT_BATCH = ROOT / "data/output/match_4_5min/P10-match4/events.json"
DEFAULT_BATCH_OUT = ROOT / "data/output/match_4_5min"
LABEL_OUT = ROOT / "reports/events_testing/COACH_EMIT_LABELS_mosaic.json"
FRAME_STEP = 10
CLIP_LEAD_S = 1.0
CLIP_TAIL_S = 0.5
QUAD_CAMS = frozenset({"P10", "P9", "P7", "P8"})
LIVE_CAMS = frozenset({"P_Goal1", "P_Goal2", "P1", "P6"})
GOAL_DEPTH_M = 7.0
END_BAND_M = 18.0
ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 0.25, 8.0, 1.0
VIEW_LAYOUTS = ("Best ball cam + pitch", "Quad mosaic + pitch", "Pitch map only")
RENDER_TILE_W, RENDER_TILE_H, RENDER_GAP = 480, 270, 2
VIEWPORT_HEIGHT = 480
BEST_BALL_TILE_SCALE = 1.0
BEST_BALL_VIEWPORT = 480
DISPLAY_BASE_W = 960
DISPLAY_MAX_W = 1200
DISPLAY_MAX_UPSCALE = 1.35
EVENT_BAR_H = 56
EVENT_UI_COLORS = {
    "pass": "#80b4ff",
    "shot": "#4040ff",
    "recovery": "#50dc78",
    "dribble": "#c8a050",
    "movement": "#b4b4b4",
}


def mosaic_split_h(tile_w: int = RENDER_TILE_W, tile_h: int = RENDER_TILE_H) -> int:
    _, grid_h = mosaic_grid_size(tile_w, tile_h, RENDER_GAP)
    return (
        MOSAIC_BAN_H + MOSAIC_NORTH_H + MOSAIC_CAM_H + grid_h + MOSAIC_CAM_H + MOSAIC_SOUTH_H
    )


def quad_tile_offsets(tile_w: int = RENDER_TILE_W, tile_h: int = RENDER_TILE_H) -> dict:
    y = MOSAIC_BAN_H + MOSAIC_NORTH_H + MOSAIC_CAM_H
    x0 = MOSAIC_SIDE_W
    g = RENDER_GAP
    return {
        "P10": (x0, y),
        "P9": (x0 + tile_w + g, y),
        "P7": (x0, y + tile_h + g),
        "P8": (x0 + tile_w + g, y + tile_h + g),
    }


def split_coach_stack(rgb):
    mh = mosaic_split_h()
    if rgb.shape[0] <= mh + 40:
        return None, rgb, None, {}, RENDER_TILE_W, RENDER_TILE_H
    mosaic = rgb[:mh, :]
    below = rgb[mh:, :]
    events_bar = None
    pitch = below
    if below.shape[0] > EVENT_BAR_H + 80:
        events_bar = below[-EVENT_BAR_H:, :]
        pitch = below[:-EVENT_BAR_H, :]
    return mosaic, pitch, events_bar, quad_tile_offsets(), RENDER_TILE_W, RENDER_TILE_H


def tile_ball_box_rect(tile_rgb) -> tuple[int, int, int, int] | None:
    """Largest orange ball-box in tile (x, y, w, h) or None."""
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(bgr, np.array([175, 85, 0]), np.array([255, 215, 95]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        ratio = w / max(h, 1)
        if 0.35 < ratio < 3.0 and area > best_area:
            best_area = area
            best_rect = (x, y, w, h)
    return best_rect


def tile_ball_box_score(tile_rgb) -> float:
    """Largest orange ball-box contour on a mosaic quad tile."""
    rect = tile_ball_box_rect(tile_rgb)
    if rect is None:
        return 0.0
    return float(rect[2] * rect[3])


def crop_tile_on_ball(tile_rgb, rect: tuple[int, int, int, int], min_frac: float = 1.0):
    """Keep full tile by default — micro-crops look weird once display-scaled."""
    if min_frac >= 0.98:
        return tile_rgb
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    th, tw = tile_rgb.shape[:2]
    crop_w = min(tw, max(int(tw * min_frac), int(w * 6), 240))
    crop_h = min(th, max(int(th * min_frac), int(h * 6), 180))
    x0 = int(max(0, min(tw - crop_w, cx - crop_w / 2.0)))
    y0 = int(max(0, min(th - crop_h, cy - crop_h / 2.0)))
    return tile_rgb[y0:y0 + crop_h, x0:x0 + crop_w]


def batch_has_quad_cams(batch_root: Path) -> bool:
    if not batch_root.is_dir():
        return False
    cams = _cached_cam_ids(str(batch_root))
    return sum(1 for c in cams if c in {"P10", "P9", "P7", "P8"}) >= 2


@st.cache_data(show_spinner=False)
def _cached_cam_ids(batch_root_str: str) -> tuple[str, ...]:
    return tuple(discover_cam_frame_csvs(Path(batch_root_str)).keys())


@st.cache_resource(show_spinner=False)
def cached_cam_tables(batch_root_str: str):
    from src.review.multicam_fuse import load_cam_tables

    return load_cam_tables(discover_cam_frame_csvs(Path(batch_root_str)))


def fuse_ball_xy_batch(batch_root: Path, src_fr: int) -> tuple[float, float] | None:
    if not batch_has_quad_cams(batch_root):
        return None
    try:
        from src.review.multicam_fuse import fuse_ball_at_frame

        tables = cached_cam_tables(str(batch_root))
        xy = fuse_ball_at_frame(tables, int(src_fr))
        if xy is None:
            return None
        return float(xy[0]), float(xy[1])
    except Exception:
        return None


def pick_cam_for_pitch_xy(bx: float, by: float) -> str:
    """Goal / endline cams in deep zones; quads in midfield (+x north, +y left)."""
    south_line = -PITCH_LEN_M / 2.0
    north_line = PITCH_LEN_M / 2.0
    if bx <= south_line + GOAL_DEPTH_M:
        return "P_Goal1"
    if bx <= south_line + END_BAND_M:
        return "P1"
    if bx >= north_line - GOAL_DEPTH_M:
        return "P_Goal2"
    if bx >= north_line - END_BAND_M:
        return "P6"
    north = bx >= -0.5
    left = by >= -0.5
    if north and left:
        return "P9"
    if north and not left:
        return "P8"
    if not north and left:
        return "P10"
    return "P7"


def load_off_quad_cam_tile(cam: str, src_fr: int, batch_root: Path) -> np.ndarray | None:
    """P1/P6/goal cams are not in the mosaic MP4 — read live from raw video."""
    from src.review.cam_mosaic import build_cam_view

    try:
        img_bgr, _ = build_cam_view(
            ROOT,
            f"Only {cam}",
            int(src_fr),
            batch_root,
            tile_w=RENDER_TILE_W,
            tile_h=RENDER_TILE_H,
            apply_defish=True,
        )
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def ball_xy_from_pitch_rgb(pitch_rgb) -> tuple[float, float] | None:
    """Yellow fuse ball on rendered cw90 pitch panel → pseudo meters for cam pick."""
    bgr = cv2.cvtColor(pitch_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(bgr, np.array([0, 200, 200]), np.array([60, 255, 255]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_c = None
    best_a = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 40 or a > 2500:
            continue
        if a > best_a:
            best_a = a
            best_c = c
    if best_c is None:
        return None
    x, y, w, h = cv2.boundingRect(best_c)
    cx, cy = x + w / 2.0, y + h / 2.0
    ph, pw = pitch_rgb.shape[:2]
    bx = (cx / max(pw, 1) - 0.5) * PITCH_LEN_M
    by = (0.5 - cy / max(ph, 1)) * PITCH_WID_M
    return bx, by


def raw_ball_score(tile_rgb) -> float:
    """Bright small blob on raw/defished video (no orange det box)."""
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < 25 or a > 900:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / max(h, 1)
        if 0.45 < ratio < 1.8:
            best = max(best, a)
    return best


def _live_cams_for_zone(geo_cam: str) -> list[str]:
    zone_order = {
        "P_Goal1": ["P_Goal1", "P1", "P10", "P7"],
        "P1": ["P1", "P_Goal1", "P10", "P7"],
        "P_Goal2": ["P_Goal2", "P6", "P9", "P8"],
        "P6": ["P6", "P_Goal2", "P9", "P8"],
        "P10": ["P10", "P7", "P1", "P_Goal1"],
        "P9": ["P9", "P8", "P6", "P_Goal2"],
        "P8": ["P8", "P9", "P6", "P_Goal2"],
        "P7": ["P7", "P10", "P1", "P_Goal1"],
    }
    return zone_order.get(geo_cam, [geo_cam])


def pick_best_ball_cam(
    tiles: dict,
    src_fr: int,
    batch_root: Path,
    ball_xy: tuple[float, float] | None = None,
) -> tuple[str, bool]:
    """Prefer mosaic quad with orange ball box; else live cam with raw ball visible."""
    scores = {cam: tile_ball_box_score(tiles[cam]) for cam in tiles}
    best_quad = max(scores, key=lambda c: scores[c])
    if scores[best_quad] >= 80:
        return best_quad, False
    geo_cam = pick_cam_for_pitch_xy(ball_xy[0], ball_xy[1]) if ball_xy else None
    if geo_cam is not None:
        for cam in _live_cams_for_zone(geo_cam):
            if cam in QUAD_CAMS and scores.get(cam, 0) >= max(40, scores[best_quad]):
                return cam, scores.get(cam, 0) < 80
            if cam in LIVE_CAMS:
                live = load_off_quad_cam_tile(cam, src_fr, batch_root)
                if live is not None and raw_ball_score(live) >= 35:
                    return cam, False
    if scores[best_quad] >= 40:
        return best_quad, True
    if geo_cam is not None:
        return geo_cam, True
    return "P10", True


def tag_best_cam_tile(tile_rgb, cam: str, no_ball: bool = False):
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, bgr.shape[0] - 34), (bgr.shape[1], bgr.shape[0]), (0, 0, 0), -1)
    label = f"BEST BALL - {cam}"
    if no_ball:
        label += " (no box in tile)"
    cv2.putText(
        bgr,
        label,
        (12, bgr.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def slim_baked_ball_box(tile_rgb, shrink: float = 0.28):
    """Shrink oversized baked orange BALL overlay so the ball is visible."""
    rect = tile_ball_box_rect(tile_rgb)
    if rect is None:
        return tile_rgb
    x, y, w, h = rect
    if w < 22 and h < 22:
        return tile_rgb
    cx, cy = x + w // 2, y + h // 2
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    orange = cv2.inRange(bgr, np.array([175, 85, 0]), np.array([255, 215, 95]))
    ys, xs = np.where(orange > 0)
    if len(xs) == 0:
        return tile_rgb
    color = tuple(int(v) for v in np.median(bgr[ys, xs], axis=0))
    # Black "BALL" label plate above the box.
    ly0, ly1 = max(0, y - 40), max(0, y + 2)
    lx0, lx1 = max(0, x - 4), min(bgr.shape[1], x + max(w, 110) + 4)
    if ly1 > ly0 and lx1 > lx0:
        plate = bgr[ly0:ly1, lx0:lx1]
        dark = (
            (plate[:, :, 0] < 45)
            & (plate[:, :, 1] < 45)
            & (plate[:, :, 2] < 45)
        )
        orange[ly0:ly1, lx0:lx1][dark] = 255
    orange = cv2.dilate(orange, np.ones((3, 3), np.uint8), iterations=1)
    fill = cv2.GaussianBlur(bgr, (31, 31), 0)
    cleaned = bgr.copy()
    cleaned[orange > 0] = fill[orange > 0]
    side = max(12, int(min(w, h) * shrink))
    nx = int(max(0, min(cleaned.shape[1] - side, cx - side // 2)))
    ny = int(max(0, min(cleaned.shape[0] - side, cy - side // 2)))
    cv2.rectangle(cleaned, (nx, ny), (nx + side, ny + side), color, 2)
    cv2.putText(
        cleaned,
        "BALL",
        (nx, max(14, ny - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


def compose_best_ball_stack(rgb, src_fr: int, batch_root: Path):
    mosaic, pitch, events_bar, offsets, tw, th = split_coach_stack(rgb)
    if mosaic is None:
        return rgb, None
    tiles = {cam: mosaic[y:y + th, x:x + tw] for cam, (x, y) in offsets.items()}
    # Orange box on mosaic is enough — skip fuse CSV reload + live raw cams.
    scores = {cam: tile_ball_box_score(tiles[cam]) for cam in tiles}
    best_quad = max(scores, key=lambda c: scores[c])
    if scores[best_quad] >= 80:
        best, no_ball = best_quad, False
    else:
        ball_xy = fuse_ball_xy_batch(batch_root, src_fr)
        if ball_xy is None and pitch is not None:
            ball_xy = ball_xy_from_pitch_rgb(pitch)
        best, no_ball = pick_best_ball_cam(tiles, src_fr, batch_root, ball_xy)
    if best in QUAD_CAMS:
        tile = tiles[best]
    else:
        live = load_off_quad_cam_tile(best, src_fr, batch_root)
        tile = live if live is not None else tiles.get("P10", next(iter(tiles.values())))
        if live is None:
            no_ball = True
    tile = slim_baked_ball_box(tile)
    ball_rect = tile_ball_box_rect(tile)
    if ball_rect is not None:
        tile = crop_tile_on_ball(tile, ball_rect)
        no_ball = False
    tile = tag_best_cam_tile(tile, best, no_ball=no_ball)
    mw = rgb.shape[1]
    cam_w = max(mw, int(mw * BEST_BALL_TILE_SCALE))
    scaled = cv2.resize(
        tile,
        (cam_w, max(1, int(tile.shape[0] * cam_w / tile.shape[1]))),
        interpolation=cv2.INTER_AREA,
    )
    if pitch.shape[1] != cam_w:
        pitch = cv2.resize(
            pitch,
            (cam_w, max(1, int(pitch.shape[0] * cam_w / pitch.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
    parts = [scaled, pitch]
    if events_bar is not None:
        if events_bar.shape[1] != cam_w:
            events_bar = cv2.resize(
                events_bar, (cam_w, EVENT_BAR_H), interpolation=cv2.INTER_AREA,
            )
        parts.append(events_bar)
    return np.vstack(parts), best


def render_label_status(t_match: float, current: dict) -> None:
    cur_t = float(current.get("t_end", 0))
    cur_type = str(current.get("type", ""))
    cur_conf = current.get("confidence", "")
    delta = abs(t_match - cur_t)
    align = "aligned" if delta < 0.3 else f"{delta:.1f}s off"
    st.markdown(
        f"**Labeling:** `{cur_type}` @ **{cur_t:.1f}s** (conf {cur_conf}) · "
        f"scrub **{t_match:.1f}s** ({align})"
    )
    if cur_type == "shot":
        goal_m = 53.90 / 2.0
        st.info(
            "Shot rule: ball **≥15 m/s** toward a goal line (|x| > "
            f"**{goal_m - 5.0:.0f} m** from centre) with a player within 4 m. "
            "Early fuse map glitches often look like shots — if play is midfield / "
            "not toward Goal 1 or Goal 2, mark **Wrong**."
        )


def render_event_timeline_scrub(
    emits: list[dict],
    current: dict,
    t_match: float,
    max_s: float,
    step_s: float,
    playing: bool,
) -> None:
    """Events timeline + seconds slider."""
    cur_t = float(current.get("t_end", 0))
    cur_type = str(current.get("type", ""))
    events = _timeline_events_near(emits, cur_t)
    if current not in events:
        events.append(current)
        events = sorted(events, key=lambda e: float(e["t_end"]))
    placed = layout_timeline_events(events, max_s)
    scrub_pct = 0.0 if max_s <= 0 else min(100.0, max(0.0, t_match / max_s * 100.0))
    page_html = build_timeline_html(
        placed,
        scrub_pct,
        t_match,
        max_s,
        step_s,
        playing,
        cur_t,
        cur_type,
        t_match,
    )
    components.html(page_html, height=timeline_html_height(placed), scrolling=False)


def apply_query_scrub(meta: dict, n_frames: int) -> None:
    qp = st.query_params
    keep_play = qp.get("keep_play") == "1"
    if "nudge_fr" in qp:
        stop_play()
        nudge_frame(int(qp["nudge_fr"]), n_frames, meta)
        del qp["nudge_fr"]
    if "zoom_step" in qp:
        if not keep_play:
            stop_play()
        delta = float(qp["zoom_step"])
        st.session_state.frame_zoom = max(
            ZOOM_MIN, min(ZOOM_MAX, float(st.session_state.frame_zoom) + delta)
        )
        del qp["zoom_step"]
    if "zoom_val" in qp:
        if not keep_play:
            stop_play()
        z = float(qp["zoom_val"])
        st.session_state.frame_zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        del qp["zoom_val"]
    if "scrub_t" in qp:
        if not keep_play:
            stop_play()
        t = float(qp["scrub_t"])
        st.session_state.scrub_frame = vid_idx_for_match_t(meta, t, n_frames)
        sync_scrub_seconds(meta)
        del qp["scrub_t"]
    if "jump_t" in qp:
        stop_play()
        t = float(qp["jump_t"])
        jump_to_match_t(meta, t, n_frames, lead_frames=2)
        del qp["jump_t"]
    if "keep_play" in qp:
        del qp["keep_play"]


def apply_view_layout(rgb, layout: str, src_fr: int, batch_root: Path):
    if layout == "Pitch map only":
        h = rgb.shape[0]
        return rgb[int(h * 0.48):, :], None
    if layout == "Best ball cam + pitch":
        return compose_best_ball_stack(rgb, src_fr, batch_root)
    return rgb, None


@st.cache_data(max_entries=48, show_spinner=False)
def cached_read_frame(video_str: str, idx: int):
    """Random-access frame read (cached). Prefer read_frame_sequential while playing."""
    cap = cv2.VideoCapture(video_str)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_frame_sequential(video_str: str, idx: int):
    """Reuse one open capture for scrub/play; seek only on jumps."""
    path = st.session_state.get("_vcap_path")
    cap = st.session_state.get("_vcap")
    last = int(st.session_state.get("_vcap_idx", -999))
    if path != video_str or cap is None:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(video_str)
        st.session_state._vcap = cap
        st.session_state._vcap_path = video_str
        last = -999
    if idx == last and "_vcap_rgb" in st.session_state:
        return st.session_state._vcap_rgb
    if idx == last + 1:
        ok, bgr = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, bgr = cap.read()
    if not ok:
        return cached_read_frame(video_str, idx)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.session_state._vcap_idx = idx
    st.session_state._vcap_rgb = rgb
    return rgb


def render_frame_view(rgb, zoom: float) -> None:
    """Native Streamlit image — avoid blowing up tiny crops into pixel soup."""
    z = max(ZOOM_MIN, min(ZOOM_MAX, float(zoom)))
    h, w = rgb.shape[:2]
    target_w = int(DISPLAY_BASE_W * z)
    target_w = max(320, min(DISPLAY_MAX_W, target_w, int(w * DISPLAY_MAX_UPSCALE)))
    if w > target_w:
        rgb = cv2.resize(
            rgb,
            (target_w, max(1, int(h * target_w / w))),
            interpolation=cv2.INTER_AREA,
        )
        w = rgb.shape[1]
    st.image(rgb, width=min(target_w, w))


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def mosaic_emits(path: Path) -> list[dict]:
    raw = load_json(path)
    return sorted(raw.get("emits") or [], key=lambda e: float(e.get("t_end", 0)))


def batch_hi(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    ev = load_json(path).get("events") or []
    return [e for e in ev if float(e.get("confidence", 0)) >= 0.8]


def near_batch(t: float, batch: list[dict], tol: float = 1.5) -> list[str]:
    out = []
    for e in batch:
        te = float(e.get("timestamp_end", 0))
        if abs(te - t) <= tol:
            out.append(f"{e.get('type')}@{te:.1f}")
    return out


def priority_rows(emits: list[dict], batch: list[dict]) -> list[dict]:
    rows = []
    for i, e in enumerate(emits, 1):
        t = float(e.get("t_end", 0))
        rare = e.get("type") in ("shot", "recovery")
        early = t <= 90
        conflict = near_batch(t, batch)
        if not (rare or early or conflict):
            continue
        rows.append(
            {
                "i": i,
                "type": e.get("type"),
                "t_end": round(t, 2),
                "conf": round(float(e.get("confidence", 0)), 3),
                "why": "rare" if rare else ("batch_conflict" if conflict else "early"),
                "batch_near": conflict,
            }
        )
    return rows


def load_labels() -> dict:
    if LABEL_OUT.is_file():
        return load_json(LABEL_OUT)
    return {"emits": {}, "updated_at": None}


def save_labels(data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    LABEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    LABEL_OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")


def label_for(data: dict, i: int) -> dict:
    return data.get("emits", {}).get(str(i), {})


def set_label(data: dict, i: int, ok: bool | None, note: str) -> None:
    data.setdefault("emits", {})[str(i)] = {
        "coach_ok": ok,
        "note": note,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def video_props(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 4.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 4.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def load_video_meta(video: Path) -> dict:
    meta_path = video.parent / "meta.json"
    if not meta_path.is_file():
        return {"stride": 15, "src_fps": 60.0, "start": 0, "out_fps": 4.0}
    m = load_json(meta_path)
    frames_src = m.get("frames_src") or [0]
    return {
        "stride": int(m.get("stride", 15)),
        "src_fps": float(m.get("src_fps", 60.0) if "src_fps" in m else 60.0),
        "start": int(frames_src[0]),
        "out_fps": float(m.get("out_fps", 4.0)),
    }


def src_frame(meta: dict, vid_idx: int) -> int:
    return meta["start"] + vid_idx * meta["stride"]


def vid_idx_for_src(meta: dict, src_fr: int) -> int:
    return max(0, (int(src_fr) - meta["start"]) // meta["stride"])


def match_t_for_vid_idx(meta: dict, vid_idx: int) -> float:
    return vid_idx * meta["stride"] / meta["src_fps"]


def vid_idx_for_match_t(meta: dict, t: float, n_frames: int) -> int:
    idx = int(round(t * meta["src_fps"] / meta["stride"]))
    return max(0, min(idx, max(0, n_frames - 1)))


def jump_to_match_t(meta: dict, t: float, n_frames: int, lead_frames: int = 2) -> None:
    idx = vid_idx_for_match_t(meta, t, n_frames)
    st.session_state.scrub_frame = max(0, idx - lead_frames)
    sync_scrub_seconds(meta)


def reviewed_count(data: dict, indices: list[int]) -> int:
    labeled = data.get("emits", {})
    return sum(1 for i in indices if labeled.get(str(i), {}).get("coach_ok") is not None)


def stop_play() -> None:
    st.session_state.playing = False
    st.session_state.play_slow = False
    st.session_state.play_clip_end = None


def emit_clip_range(emit: dict, meta: dict, n_frames: int) -> tuple[int, int]:
    t0 = float(emit.get("t_start", emit.get("t_end", 0)))
    t1 = float(emit.get("t_end", 0))
    start = vid_idx_for_match_t(meta, max(0.0, t0 - CLIP_LEAD_S), n_frames)
    end = vid_idx_for_match_t(meta, t1 + CLIP_TAIL_S, n_frames)
    return start, max(start, end)


def clip_queue_next(current_i: int, queue: list[int]) -> int | None:
    if current_i in queue:
        pos = queue.index(current_i)
        if pos + 1 < len(queue):
            return queue[pos + 1]
    for qi in queue:
        if qi > current_i:
            return qi
    return None


def start_emit_clip(emit: dict, meta: dict, n_frames: int, slow: bool = False) -> None:
    start, end = emit_clip_range(emit, meta, n_frames)
    st.session_state.scrub_frame = start
    st.session_state.play_clip_end = end
    st.session_state.playing = True
    st.session_state.play_slow = slow
    sync_scrub_seconds(meta)


def jump_emit_clip(emit_i: int, emits: list[dict], meta: dict, n_frames: int) -> None:
    emit = emits[emit_i - 1]
    start, _ = emit_clip_range(emit, meta, n_frames)
    stop_play()
    st.session_state.scrub_frame = start
    st.session_state.emit_jump = emit_i
    sync_scrub_seconds(meta)


def advance_emit_clip(
    emit_i: int,
    clip_queue: list[int],
    emits: list[dict],
    meta: dict,
    n_frames: int,
    auto_play: bool,
) -> None:
    nxt = clip_queue_next(emit_i, clip_queue)
    if nxt is None:
        stop_play()
        st.toast("End of clip queue")
        return
    jump_emit_clip(nxt, emits, meta, n_frames)
    if auto_play:
        start_emit_clip(emits[nxt - 1], meta, n_frames)


def sync_scrub_seconds(meta: dict) -> None:
    st.session_state.scrub_slider = match_t_for_vid_idx(meta, st.session_state.scrub_frame)


def nudge_frame(delta: int, n_frames: int, meta: dict) -> None:
    st.session_state.scrub_frame = int(
        max(0, min(st.session_state.scrub_frame + delta, max(0, n_frames - 1)))
    )
    sync_scrub_seconds(meta)


def render_stride_note(meta: dict) -> None:
    s = meta["stride"]
    if s == 1:
        st.sidebar.success("Events: every source frame (stride 1)")
    else:
        st.sidebar.warning(
            f"Events in this MP4 ran every **{s}** source frames "
            f"(not every frame). Batch single-cam uses stride 1; "
            f"product fuse gate should use stride 1."
        )


def render_playback_row(
    n_frames: int,
    playing: bool,
    play_slow: bool,
    meta: dict,
    emit: dict,
    clip_mode: bool,
) -> None:
    """Playback row — clip mode plays emit window then auto-pauses."""
    cplay, cslow, cnext, cback, cfwd, czout, czin = st.columns(7, gap="small")
    with cplay:
        if clip_mode:
            lbl = "Pause clip" if playing else "Play clip"
        else:
            lbl = "Pause" if playing and not play_slow else "Play"
        if st.button(f"▶ {lbl}", type="primary", use_container_width=True, key="btn_play"):
            if playing:
                stop_play()
            elif clip_mode:
                start_emit_clip(emit, meta, n_frames, slow=False)
            else:
                st.session_state.playing = True
                st.session_state.play_slow = False
            st.rerun()
    with cslow:
        lbl = "Pause" if playing and play_slow else "Slow"
        if st.button(f"Slow {lbl}", use_container_width=True, key="btn_slow"):
            if playing and play_slow:
                stop_play()
            elif clip_mode:
                start_emit_clip(emit, meta, n_frames, slow=True)
            else:
                st.session_state.playing = True
                st.session_state.play_slow = True
            st.rerun()
    with cnext:
        if clip_mode:
            if st.button("Next clip", use_container_width=True, key="btn_next_clip"):
                advance_emit_clip(
                    int(st.session_state.get("label_emit_i", 1)),
                    st.session_state.get("clip_queue", []),
                    st.session_state.get("clip_emits", []),
                    meta,
                    n_frames,
                    auto_play=False,
                )
                st.rerun()
    with cback:
        if st.button(
            f"−{FRAME_STEP} fr", use_container_width=True, disabled=playing, key="btn_back"
        ):
            nudge_frame(-FRAME_STEP, n_frames, meta)
            stop_play()
            st.rerun()
    with cfwd:
        if st.button(
            f"+{FRAME_STEP} fr", use_container_width=True, disabled=playing, key="btn_fwd"
        ):
            nudge_frame(FRAME_STEP, n_frames, meta)
            stop_play()
            st.rerun()
    with czout:
        if st.button("Zoom −", use_container_width=True, key="btn_zout"):
            z = max(ZOOM_MIN, float(st.session_state.frame_zoom) - 0.2)
            st.session_state.frame_zoom = z
            st.session_state.zoom_slider_widget = z
            st.rerun()
    with czin:
        if st.button("Zoom +", use_container_width=True, key="btn_zin"):
            z = min(ZOOM_MAX, float(st.session_state.frame_zoom) + 0.2)
            st.session_state.frame_zoom = z
            st.session_state.zoom_slider_widget = z
            st.rerun()
    if "zoom_slider_widget" not in st.session_state:
        st.session_state.zoom_slider_widget = float(st.session_state.frame_zoom)
    z = st.slider(
        "Zoom",
        min_value=float(ZOOM_MIN),
        max_value=float(ZOOM_MAX),
        step=0.05,
        format="%.2fx",
        key="zoom_slider_widget",
    )
    st.session_state.frame_zoom = float(z)


def render_video_player(
    video: Path,
    meta: dict,
    fps: float,
    n_frames: int,
    batch_root: Path,
    emits: list[dict],
    current_emit: dict,
    clip_mode: bool,
) -> None:
    apply_query_scrub(meta, n_frames)
    layout = str(st.session_state.view_layout)
    playing = bool(st.session_state.playing)
    play_slow = bool(st.session_state.play_slow)
    if playing:
        sync_scrub_seconds(meta)
    step_s = meta["stride"] / meta["src_fps"]
    max_s = match_t_for_vid_idx(meta, max(0, n_frames - 1))
    idx = int(st.session_state.scrub_frame)
    t_match = match_t_for_vid_idx(meta, idx)

    render_label_status(t_match, current_emit)
    if playing:
        st.caption(f"Playing · scrub **{t_match:.1f}s** (timeline paused for speed)")
    else:
        render_event_timeline_scrub(emits, current_emit, t_match, max_s, step_s, playing)
    render_playback_row(n_frames, playing, play_slow, meta, current_emit, clip_mode)
    idx = int(st.session_state.scrub_frame)
    src_fr = src_frame(meta, idx)
    t_match = match_t_for_vid_idx(meta, idx)

    rgb = read_frame_sequential(str(video), idx)
    if rgb is None:
        st.error("Could not read frame")
        return
    src_fr = src_frame(meta, idx)
    rgb, best_cam = apply_view_layout(rgb, layout, src_fr, batch_root)
    render_frame_view(rgb, float(st.session_state.frame_zoom))
    st.caption(
        f"**{t_match:.1f}s** · zoom **{float(st.session_state.frame_zoom):.1f}x** · "
        f"cam **{best_cam or '—'}**"
    )

    if playing:
        play_fps = 0.75 if play_slow else min(4.0, float(meta["out_fps"]))
        clip_end = st.session_state.get("play_clip_end")
        if clip_end is not None and idx >= int(clip_end):
            stop_play()
            st.toast("Clip ended — remark below, then Next clip or Play clip")
            st.rerun()
        nxt = idx + 1
        if nxt >= n_frames:
            stop_play()
            st.toast("Playback finished")
            st.rerun()
        st.session_state.scrub_frame = nxt
        time.sleep(1.0 / max(play_fps, 0.25))
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Fuse emit labels", layout="wide")
    if "scrub_frame" not in st.session_state:
        st.session_state.scrub_frame = 0
    if "scrub_slider" not in st.session_state:
        st.session_state.scrub_slider = 0.0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "play_slow" not in st.session_state:
        st.session_state.play_slow = False
    if "frame_zoom" not in st.session_state:
        st.session_state.frame_zoom = ZOOM_DEFAULT
    if "play_clip_end" not in st.session_state:
        st.session_state.play_clip_end = None

    video = Path(st.sidebar.text_input("Video", value=str(DEFAULT_VIDEO)))
    emits_path = Path(st.sidebar.text_input("Emits JSON", value=str(DEFAULT_EMITS)))
    if not video.is_file() or not emits_path.is_file():
        st.error("Video or emits JSON missing.")
        st.stop()

    meta = load_video_meta(video)
    fps, n_frames = video_props(video)
    render_stride_note(meta)
    st.sidebar.caption(
        "Emits use **shot gates v2** (kickoff floor, goal-mouth, 2-frame confirm). "
        "Refresh after `emits_rebuild.log` finishes for full re-score."
    )
    st.sidebar.selectbox(
        "View layout",
        VIEW_LAYOUTS,
        key="view_layout",
        help=(
            "Best ball follows fuse ball: P_Goal1/P1 south, P_Goal2/P6 north, "
            "quads midfield; end/goal cams load live from raw video."
        ),
    )
    batch_root = Path(
        st.sidebar.text_input("Batch output (best cam)", value=str(DEFAULT_BATCH_OUT))
    )

    emits = mosaic_emits(emits_path)
    batch = batch_hi(DEFAULT_BATCH)
    show_all = st.sidebar.checkbox("Show all emits", value=False)
    clip_mode = st.sidebar.checkbox(
        "Clip mode (pause after each emit)",
        value=True,
        help="Play clip runs t_start−1s → t_end+0.5s then pauses for labeling.",
    )
    rows = priority_rows(emits, batch)
    if show_all:
        rows = [
            {
                "i": j,
                "type": e["type"],
                "t_end": round(float(e["t_end"]), 2),
                "conf": e["confidence"],
                "why": "all",
                "batch_near": [],
            }
            for j, e in enumerate(emits, 1)
        ]

    labels = load_labels()
    labels.setdefault("video", str(video.relative_to(ROOT)))
    labels.setdefault("emits_source", str(emits_path.relative_to(ROOT)))
    done = reviewed_count(labels, [r["i"] for r in rows])
    st.sidebar.progress(done / max(len(rows), 1), text=f"Reviewed {done}/{len(rows)}")

    jump_i = st.session_state.get("emit_jump")
    default_i = int(jump_i) if jump_i else rows[0]["i"]
    if jump_i:
        st.session_state.emit_jump = None
    emit_i = int(st.sidebar.number_input("Emit #", 1, len(emits), default_i))
    st.session_state.label_emit_i = emit_i
    clip_queue = [r["i"] for r in rows]
    st.session_state.clip_queue = clip_queue
    st.session_state.clip_emits = emits
    emit = emits[emit_i - 1]
    emit_t = float(emit.get("t_end", 0))
    emit_src = int(emit.get("frame_id", 0))
    clip_start, clip_end = emit_clip_range(emit, meta, n_frames)
    clip_t0 = match_t_for_vid_idx(meta, clip_start)
    clip_t1 = match_t_for_vid_idx(meta, clip_end)

    if "boot_scrub" not in st.session_state:
        jump_to_match_t(meta, emit_t, n_frames, lead_frames=4)
        st.session_state.boot_scrub = True

    if st.sidebar.button("Jump to emit time"):
        stop_play()
        jump_to_match_t(meta, emit_t, n_frames, lead_frames=4)
        st.rerun()

    st.markdown(
        f"### Emit **#{emit_i}** · **{emit['type']}** @ **{emit_t:.1f}s** "
        f"(src frame {emit_src})"
    )
    if clip_mode:
        st.caption(
            f"Clip **{clip_t0:.1f}s → {clip_t1:.1f}s** "
            f"(−{CLIP_LEAD_S:.0f}s lead, +{CLIP_TAIL_S:.1f}s tail) · "
            f"queue **{clip_queue.index(emit_i) + 1 if emit_i in clip_queue else '—'}"
            f"/{len(clip_queue)}**"
        )
    render_video_player(
        video, meta, fps, n_frames, batch_root, emits, emit, clip_mode
    )

    cur = label_for(labels, emit_i)
    note = st.text_input("Note", value=cur.get("note", ""), key=f"note_{emit_i}")
    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("✓ Correct", type="primary", use_container_width=True):
        set_label(labels, emit_i, True, note)
        save_labels(labels)
        st.rerun()
    if b2.button("✗ Wrong", use_container_width=True):
        set_label(labels, emit_i, False, note)
        save_labels(labels)
        st.rerun()
    if b3.button("Clear label", use_container_width=True):
        set_label(labels, emit_i, None, note)
        save_labels(labels)
        st.rerun()
    if b4.button("Next unlabeled emit", use_container_width=True):
        for r in rows:
            if label_for(labels, r["i"]).get("coach_ok") is None:
                et = float(emits[r["i"] - 1].get("t_end", 0))
                stop_play()
                jump_to_match_t(meta, et, n_frames, lead_frames=4)
                st.session_state.emit_jump = r["i"]
                break
        st.rerun()
    if b5.button("Label + next clip", use_container_width=True, disabled=not clip_mode):
        if cur.get("coach_ok") is None:
            set_label(labels, emit_i, True, note)
            save_labels(labels)
        advance_emit_clip(emit_i, clip_queue, emits, meta, n_frames, auto_play=True)
        st.rerun()

    if cur.get("coach_ok") is True:
        st.success("Marked correct")
    elif cur.get("coach_ok") is False:
        st.error("Marked wrong")

    with st.sidebar.expander("Emit list"):
        for r in rows:
            lb = label_for(labels, r["i"])
            mark = "✓" if lb.get("coach_ok") is True else ("✗" if lb.get("coach_ok") is False else "·")
            st.text(f"{mark} #{r['i']} {r['type']} {r['t_end']:.1f}s")

    st.caption(
        "Clip mode: Play clip → auto-pause → remark → Next clip or Label + next clip."
    )
    st.caption("Zoom via slider or Zoom ± · native image (no iframe).")


if __name__ == "__main__":
    main()
