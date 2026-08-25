"""Streamlit — label fuse mosaic emits (stable frame scrub).

Usage: streamlit run apps/coach_emit_label_dashboard.py
"""
from __future__ import annotations

import base64
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
from src.review.cam_mosaic import (  # noqa: E402
    MOSAIC_BAN_H,
    MOSAIC_CAM_H,
    MOSAIC_NORTH_H,
    MOSAIC_SOUTH_H,
    MOSAIC_SIDE_W,
    best_cam_for_frame,
    mosaic_grid_size,
)
from src.review.multicam_fuse import discover_cam_frame_csvs  # noqa: E402

DEFAULT_VIDEO = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_first_90s.mp4"
DEFAULT_EMITS = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json"
DEFAULT_BATCH = ROOT / "data/output/match_4_5min/P10-match4/events.json"
DEFAULT_BATCH_OUT = ROOT / "data/output/match_4_5min"
LABEL_OUT = ROOT / "reports/events_testing/COACH_EMIT_LABELS_mosaic.json"
FRAME_STEP = 10
ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 0.5, 5.0, 1.25
VIEW_LAYOUTS = ("Best ball cam + pitch", "Quad mosaic + pitch", "Pitch map only")
RENDER_TILE_W, RENDER_TILE_H, RENDER_GAP = 480, 270, 2
VIEWPORT_HEIGHT = 520
BEST_BALL_TILE_SCALE = 2.2
BEST_BALL_VIEWPORT = 560
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


def tile_ball_box_score(tile_rgb) -> float:
    """Largest orange ball-box contour on a mosaic quad tile."""
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(bgr, np.array([175, 85, 0]), np.array([255, 215, 95]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        ratio = w / max(h, 1)
        if 0.35 < ratio < 3.0:
            best = max(best, area)
    return best


def batch_has_quad_cams(batch_root: Path) -> bool:
    if not batch_root.is_dir():
        return False
    cams = discover_cam_frame_csvs(batch_root)
    return sum(1 for c in cams if c in {"P10", "P9", "P7", "P8"}) >= 2


def pick_best_ball_cam(tiles: dict, src_fr: int, batch_root: Path) -> str:
    if batch_has_quad_cams(batch_root):
        try:
            return best_cam_for_frame(batch_root, int(src_fr), "P10")
        except Exception:
            pass
    scores = {cam: tile_ball_box_score(tiles[cam]) for cam in tiles}
    best = max(scores, key=lambda c: scores[c])
    if scores[best] < 80:
        return "P10"
    return best


def tag_best_cam_tile(tile_rgb, cam: str):
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, bgr.shape[0] - 34), (bgr.shape[1], bgr.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        f"BEST BALL - {cam}",
        (12, bgr.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compose_best_ball_stack(rgb, src_fr: int, batch_root: Path):
    mosaic, pitch, events_bar, offsets, tw, th = split_coach_stack(rgb)
    if mosaic is None:
        return rgb, None
    tiles = {cam: mosaic[y:y + th, x:x + tw] for cam, (x, y) in offsets.items()}
    best = pick_best_ball_cam(tiles, src_fr, batch_root)
    tile = tag_best_cam_tile(tiles[best], best)
    mw = rgb.shape[1]
    cam_w = max(mw, int(mw * BEST_BALL_TILE_SCALE))
    scaled = cv2.resize(
        tile,
        (cam_w, max(1, int(tile.shape[0] * cam_w / tile.shape[1]))),
        interpolation=cv2.INTER_CUBIC,
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


def render_events_strip(emits: list[dict], t_match: float, current: dict) -> None:
    cur_t = float(current.get("t_end", 0))
    cur_type = str(current.get("type", ""))
    cur_conf = current.get("confidence", "")
    st.markdown(
        f"**Labeling:** `{cur_type}` @ **{cur_t:.1f}s** "
        f"(conf {cur_conf}) · scrub **{t_match:.1f}s**"
    )
    window = 15.0
    recent = sorted(
        [e for e in emits if abs(float(e["t_end"]) - cur_t) <= window],
        key=lambda e: float(e["t_end"]),
    )
    if current not in recent:
        recent.append(current)
        recent = sorted(recent, key=lambda e: float(e["t_end"]))
    chips = []
    for em in recent[-8:]:
        tp = str(em.get("type", ""))
        t = float(em.get("t_end", 0))
        col = EVENT_UI_COLORS.get(tp, "#888")
        bold = tp == cur_type and abs(t - cur_t) < 0.2
        weight = "700" if bold else "400"
        border = "2px solid #fff" if bold else "1px solid #333"
        chips.append(
            f'<span style="background:{col};color:#111;padding:5px 12px;margin:3px;'
            f'border-radius:5px;font-weight:{weight};border:{border}">'
            f'{tp} @{t:.1f}s</span>'
        )
    st.markdown(
        '<div style="background:#1e1e22;padding:8px 10px;border-radius:6px;margin-top:4px">'
        + " ".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


def apply_view_layout(rgb, layout: str, src_fr: int, batch_root: Path):
    if layout == "Pitch map only":
        h = rgb.shape[0]
        return rgb[int(h * 0.48):, :], None
    if layout == "Best ball cam + pitch":
        return compose_best_ball_stack(rgb, src_fr, batch_root)
    return rgb, None


@st.cache_data(max_entries=128, show_spinner=False)
def cached_read_frame(video_str: str, idx: int):
    cap = cv2.VideoCapture(video_str)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def jpeg_b64(rgb, max_w: int = 2200, quality: int = 85) -> str:
    h, w = rgb.shape[:2]
    if w > max_w:
        scale = max_w / w
        rgb = cv2.resize(rgb, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def render_scroll_zoom_frame(rgb, start_zoom: float, viewport_h: int = VIEWPORT_HEIGHT) -> None:
    b64 = jpeg_b64(rgb)
    if not b64:
        st.error("Could not encode frame")
        return
    z0 = max(ZOOM_MIN, min(ZOOM_MAX, float(start_zoom)))
    h_px = viewport_h
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0; padding: 0; width: 100%; height: {h_px}px;
    background: #111; overflow: hidden;
  }}
  #viewport {{
    width: 100%; height: {h_px}px; overflow: auto;
    cursor: grab; background: #111;
  }}
  #viewport.grabbing {{ cursor: grabbing; }}
  #img {{ display: block; margin: 0 auto; border: 0; }}
</style>
</head>
<body>
  <div id="viewport">
    <img id="img" src="data:image/jpeg;base64,{b64}">
  </div>
  <script>
    (function () {{
      const vp = document.getElementById("viewport");
      const img = document.getElementById("img");
      let scale = {z0};
      let drag = false, sx = 0, sy = 0, sl = 0, st = 0;

      function applyZoom() {{
        const nw = img.naturalWidth || 800;
        const nh = img.naturalHeight || 600;
        const w = Math.max(320, vp.clientWidth * 0.98) * scale;
        img.style.width = w + "px";
        img.style.height = (nh * w / nw) + "px";
      }}

      function onWheel(e) {{
        e.preventDefault();
        e.stopPropagation();
        scale = Math.max({ZOOM_MIN}, Math.min({ZOOM_MAX},
          scale * (e.deltaY > 0 ? 0.9 : 1.1)));
        applyZoom();
      }}

      img.addEventListener("load", applyZoom);
      if (img.complete) applyZoom();
      setTimeout(applyZoom, 80);
      vp.addEventListener("wheel", onWheel, {{ passive: false }});

      vp.addEventListener("mousedown", (e) => {{
        drag = true;
        sx = e.clientX; sy = e.clientY;
        sl = vp.scrollLeft; st = vp.scrollTop;
        vp.classList.add("grabbing");
        e.preventDefault();
      }});
      window.addEventListener("mousemove", (e) => {{
        if (!drag) return;
        vp.scrollLeft = sl - (e.clientX - sx);
        vp.scrollTop = st - (e.clientY - sy);
      }});
      window.addEventListener("mouseup", () => {{
        drag = false;
        vp.classList.remove("grabbing");
      }});
    }})();
  </script>
</body>
</html>"""
    components.html(html, height=h_px + 12, scrolling=False)


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


def reviewed_count(data: dict, indices: list[int]) -> int:
    labeled = data.get("emits", {})
    return sum(1 for i in indices if labeled.get(str(i), {}).get("coach_ok") is not None)


def stop_play() -> None:
    st.session_state.playing = False
    st.session_state.play_slow = False


def sync_scrub_slider() -> None:
    st.session_state.scrub_slider = int(st.session_state.scrub_frame)


def nudge_frame(delta: int, n_frames: int) -> None:
    st.session_state.scrub_frame = int(
        max(0, min(st.session_state.scrub_frame + delta, max(0, n_frames - 1)))
    )


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


def render_transport_bar(n_frames: int, playing: bool, play_slow: bool) -> None:
    """Playback controls above video — always visible without scrolling."""
    st.markdown("#### Playback")
    cplay, cslow, cback, cfwd, czout, czin = st.columns(6)
    with cplay:
        lbl = "Pause" if playing and not play_slow else "Play"
        if st.button(f"▶ {lbl}", type="primary", use_container_width=True, key="btn_play"):
            if playing and not play_slow:
                stop_play()
            else:
                st.session_state.playing = True
                st.session_state.play_slow = False
            st.rerun()
    with cslow:
        lbl = "Pause" if playing and play_slow else "Slow"
        if st.button(f"Slow {lbl}", use_container_width=True, key="btn_slow"):
            if playing and play_slow:
                stop_play()
            else:
                st.session_state.playing = True
                st.session_state.play_slow = True
            st.rerun()
    with cback:
        if st.button(f"−{FRAME_STEP} frames", use_container_width=True, disabled=playing, key="btn_back"):
            nudge_frame(-FRAME_STEP, n_frames)
            stop_play()
            st.rerun()
    with cfwd:
        if st.button(f"+{FRAME_STEP} frames", use_container_width=True, disabled=playing, key="btn_fwd"):
            nudge_frame(FRAME_STEP, n_frames)
            stop_play()
            st.rerun()
    with czout:
        if st.button("Zoom −", use_container_width=True, disabled=playing, key="btn_zout"):
            st.session_state.frame_zoom = max(
                ZOOM_MIN, float(st.session_state.frame_zoom) - 0.2
            )
            st.rerun()
    with czin:
        if st.button("Zoom +", use_container_width=True, disabled=playing, key="btn_zin"):
            st.session_state.frame_zoom = min(
                ZOOM_MAX, float(st.session_state.frame_zoom) + 0.2
            )
            st.rerun()
    if playing:
        mode = "slow" if play_slow else "normal"
        st.caption(f"Playing ({mode}) — click Pause to stop")
    else:
        sync_scrub_slider()
    st.slider(
        "Video frame",
        0,
        max(0, n_frames - 1),
        disabled=playing,
        key="scrub_slider",
    )
    if not playing:
        st.session_state.scrub_frame = int(st.session_state.scrub_slider)


def render_video_player(
    video: Path,
    meta: dict,
    fps: float,
    n_frames: int,
    batch_root: Path,
    emits: list[dict],
    current_emit: dict,
) -> None:
    layout = str(st.session_state.view_layout)
    idx = int(st.session_state.scrub_frame)
    playing = bool(st.session_state.playing)
    play_slow = bool(st.session_state.play_slow)
    src_fr = src_frame(meta, idx)
    t_match = (src_fr - meta["start"]) / meta["src_fps"]

    render_events_strip(emits, t_match, current_emit)
    render_transport_bar(n_frames, playing, play_slow)
    idx = int(st.session_state.scrub_frame)
    src_fr = src_frame(meta, idx)
    t_match = (src_fr - meta["start"]) / meta["src_fps"]

    rgb = cached_read_frame(str(video), idx)
    if rgb is None:
        st.error("Could not read frame")
        return
    src_fr = src_frame(meta, idx)
    rgb, best_cam = apply_view_layout(rgb, layout, src_fr, batch_root)
    t_vid = idx / fps if fps > 0 else 0.0
    vp_h = BEST_BALL_VIEWPORT if layout == "Best ball cam + pitch" else VIEWPORT_HEIGHT
    render_scroll_zoom_frame(rgb, float(st.session_state.frame_zoom), viewport_h=vp_h)
    cam_note = f" · cam **{best_cam}**" if best_cam else ""
    st.caption(
        f"frame **{idx}**/{n_frames - 1} · src **{src_fr}** · "
        f"match **{t_match:.2f}s**{cam_note} · scroll on video to zoom"
    )

    if playing:
        play_fps = 0.75 if play_slow else float(meta["out_fps"])
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
        st.session_state.scrub_slider = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "play_slow" not in st.session_state:
        st.session_state.play_slow = False
    if "frame_zoom" not in st.session_state:
        st.session_state.frame_zoom = ZOOM_DEFAULT

    video = Path(st.sidebar.text_input("Video", value=str(DEFAULT_VIDEO)))
    emits_path = Path(st.sidebar.text_input("Emits JSON", value=str(DEFAULT_EMITS)))
    if not video.is_file() or not emits_path.is_file():
        st.error("Video or emits JSON missing.")
        st.stop()

    meta = load_video_meta(video)
    fps, n_frames = video_props(video)
    render_stride_note(meta)
    st.sidebar.selectbox(
        "View layout",
        VIEW_LAYOUTS,
        key="view_layout",
        help="Best ball = enlarged quad tile from mosaic (fast, no live detect).",
    )
    batch_root = Path(
        st.sidebar.text_input("Batch output (best cam)", value=str(DEFAULT_BATCH_OUT))
    )

    emits = mosaic_emits(emits_path)
    batch = batch_hi(DEFAULT_BATCH)
    show_all = st.sidebar.checkbox("Show all emits", value=False)
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
    emit = emits[emit_i - 1]
    emit_t = float(emit.get("t_end", 0))
    emit_src = int(emit.get("frame_id", 0))

    if "boot_scrub" not in st.session_state:
        jump_idx = max(0, vid_idx_for_src(meta, emit_src) - 4)
        st.session_state.scrub_frame = jump_idx
        st.session_state.scrub_slider = jump_idx
        st.session_state.boot_scrub = True

    if st.sidebar.button("Jump to emit frame"):
        stop_play()
        jump_idx = max(0, vid_idx_for_src(meta, emit_src) - 4)
        st.session_state.scrub_frame = jump_idx
        st.session_state.scrub_slider = jump_idx
        st.rerun()

    st.markdown(
        f"### Emit **#{emit_i}** · **{emit['type']}** @ **{emit_t:.1f}s** "
        f"(src frame {emit_src})"
    )
    render_video_player(video, meta, fps, n_frames, batch_root, emits, emit)

    cur = label_for(labels, emit_i)
    note = st.text_input("Note", value=cur.get("note", ""), key=f"note_{emit_i}")
    b1, b2, b3, b4 = st.columns(4)
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
                ef = int(emits[r["i"] - 1].get("frame_id", 0))
                stop_play()
                jump_idx = max(0, vid_idx_for_src(meta, ef) - 4)
                st.session_state.scrub_frame = jump_idx
                st.session_state.scrub_slider = jump_idx
                st.session_state.emit_jump = r["i"]
                break
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


if __name__ == "__main__":
    main()
