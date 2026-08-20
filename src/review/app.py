# Streamlit Review Dashboard — Phase 1
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Pitch 1 / Field 1 (docs/product/PITCH1_DIMENSIONS.json) — not FIFA 105×68
PITCH_LEN = 53.9
PITCH_WID = 34.84
HALF_L = PITCH_LEN / 2.0
HALF_W = PITCH_WID / 2.0


@st.cache_resource
def _load_verify_detector(player_ckpt: str, ball_ckpt: str, thr: float, nms_ver: str = "v4"):
    from src.perception.rfdetr_local import LocalRFDETRDetector

    _ = nms_ver  # bust cache when NMS logic changes
    return LocalRFDETRDetector(
        player_checkpoint=player_ckpt,
        ball_checkpoint=ball_ckpt,
        confidence_threshold=thr,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

def load_events(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])


def load_match_meta(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"match_id": data.get("match_id"), "metadata": data.get("metadata") or {}}


def load_checkpoints(checkpoint_dir: str) -> List[str]:
    if not os.path.exists(checkpoint_dir):
        return []
    return sorted(
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".json")
    )


def list_run_dirs(output_root: str) -> List[str]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    runs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "events.json").is_file():
            runs.append(p.name)
    return runs


def render_event_summary(events: List[Dict]):
    if not events:
        st.info("No events loaded")
        return
    counts = {}
    for event in events:
        t = event.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    cols = st.columns(5)
    metrics = [
        ("Total Events", len(events)),
        ("Passes", counts.get("pass", 0)),
        ("Dribbles", counts.get("dribble", 0)),
        ("Shots", counts.get("shot", 0)),
        ("Recoveries", counts.get("recovery", 0)),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_event_timeline(events: List[Dict]):
    if not events or not PLOTLY_AVAILABLE:
        return
    df = pd.DataFrame(
        [
            {
                "frame": e.get("start_frame", 0),
                "type": e.get("type", "unknown"),
                "confidence": e.get("confidence", 0.0),
            }
            for e in events
        ]
    )
    fig = px.scatter(
        df,
        x="frame",
        y="type",
        color="confidence",
        color_continuous_scale="Viridis",
        title="Event Timeline",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_event_table(events: List[Dict]):
    if not events:
        return
    rows = [
        {
            "ID": e.get("id", ""),
            "Type": e.get("type", ""),
            "Start Frame": e.get("start_frame", 0),
            "End Frame": e.get("end_frame", 0),
            "Confidence": f"{e.get('confidence', 0.0):.2f}",
            "Players": len(e.get("involved_players", [])),
        }
        for e in events
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_pitch_map(events: List[Dict]):
    if not events or not PLOTLY_AVAILABLE:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-HALF_L, HALF_L, HALF_L, -HALF_L, -HALF_L],
            y=[-HALF_W, -HALF_W, HALF_W, HALF_W, -HALF_W],
            mode="lines",
            name="Pitch 1",
            line=dict(color="green", width=2),
        )
    )
    # Cap markers so large runs stay responsive
    show = events[:500]
    for event in show:
        start = event.get("start_location") or {}
        end = event.get("end_location") or {}
        et = event.get("type", "")
        color = {"pass": "blue", "shot": "red", "dribble": "orange", "recovery": "purple"}.get(
            et, "gray"
        )
        fig.add_trace(
            go.Scatter(
                x=[start.get("x", 0), end.get("x", 0)],
                y=[start.get("y", 0), end.get("y", 0)],
                mode="lines+markers",
                name=et,
                line=dict(color=color, width=1),
                marker=dict(size=6),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Events on Pitch 1 (showing {len(show)}/{len(events)})",
        xaxis_title="X meters (+north)",
        yaxis_title="Y meters (+left)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def load_frame_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def render_frame_pitch(frame_df: pd.DataFrame):
    if frame_df.empty or not PLOTLY_AVAILABLE:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-HALF_L, HALF_L, HALF_L, -HALF_L, -HALF_L],
            y=[-HALF_W, -HALF_W, HALF_W, HALF_W, -HALF_W],
            mode="lines",
            name="Pitch 1",
            line=dict(color="green", width=2),
        )
    )
    players = frame_df[frame_df["Player_ID"] != -1]
    balls = frame_df[frame_df["Player_ID"] == -1]
    if not players.empty:
        sample = players.sample(n=min(2000, len(players)), random_state=0)
        fig.add_trace(
            go.Scatter(
                x=sample["Location_X"],
                y=sample["Location_Y"],
                mode="markers",
                name="Players",
                marker=dict(
                    size=5,
                    color=sample["Team_ID"],
                    colorscale="Bluered",
                    opacity=0.45,
                ),
            )
        )
    if not balls.empty:
        fig.add_trace(
            go.Scatter(
                x=balls["Location_X"],
                y=balls["Location_Y"],
                mode="markers",
                name="Ball",
                marker=dict(size=8, color="orange", symbol="circle"),
            )
        )
    fig.update_layout(
        title="Tracked locations on Pitch 1",
        xaxis_title="X meters (+north)",
        yaxis_title="Y meters (+left)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def persist_corrections(run_dir: Path, events: List[Dict], notes: str) -> Path:
    path = run_dir / "corrections.json"
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "events": events,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    events_path = run_dir / "events.json"
    meta = load_match_meta(str(events_path))
    out = {
        "match_id": meta.get("match_id") or run_dir.name,
        "events": events,
        "metadata": {
            **(meta.get("metadata") or {}),
            "reviewed": True,
            "reviewed_at": payload["updated_at"],
        },
    }
    events_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def render_corrections_editor(run_dir: Path, events: List[Dict]):
    st.subheader("Manual corrections")
    st.caption("Edit type / confidence, drop bad rows, then Persist to update events.json.")
    if not events:
        st.info("No events to edit")
        return
    edit_rows = []
    for e in events:
        start = e.get("start_location") or {}
        edit_rows.append(
            {
                "id": e.get("id", ""),
                "type": e.get("type", ""),
                "start_frame": int(e.get("start_frame", 0)),
                "confidence": float(e.get("confidence", 0.0)),
                "x": float(start.get("x", 0.0)),
                "y": float(start.get("y", 0.0)),
                "keep": True,
            }
        )
    edited = st.data_editor(
        pd.DataFrame(edit_rows),
        use_container_width=True,
        num_rows="dynamic",
        key="corrections_editor",
    )
    notes = st.text_input("Review notes", value="")
    if st.button("Persist corrections", type="primary"):
        by_id = {e.get("id"): e for e in events}
        updated = []
        for _, row in edited.iterrows():
            if not bool(row.get("keep", True)):
                continue
            eid = row["id"]
            base = dict(by_id.get(eid) or {})
            base["id"] = eid
            base["type"] = str(row["type"])
            base["start_frame"] = int(row["start_frame"])
            base["confidence"] = float(row["confidence"])
            loc = dict(base.get("start_location") or {})
            loc["x"] = float(row["x"])
            loc["y"] = float(row["y"])
            base["start_location"] = loc
            if "end_location" not in base:
                base["end_location"] = dict(loc)
            updated.append(base)
        path = persist_corrections(run_dir, updated, notes)
        st.success(f"Saved {len(updated)} events → {path.name} and events.json")
        st.rerun()


def render_synced_frame_review(
    run_dir: Path,
    run_name: str,
    frame_df: pd.DataFrame,
    events: List[Dict],
):
    """Video verification: detector boxes vs mapped tracks + pitch."""
    from src.review.frame_sync import (
        ball_zoom_crop,
        draw_det_boxes,
        draw_labels_on_frame,
        draw_legend,
        guess_video_for_run,
        keep_top1_ball,
        load_H_inv,
        read_video_frame,
        rows_for_frame,
    )

    st.header("Verify labels — video + pitch")
    st.info(
        "**Video:** RF-DETR bounding boxes only. "
        "**Pitch 1 panel:** yellow ball + team-colored players. "
        "MAP-BALL X is off by default (optional debug in sidebar)."
    )
    repo = Path(__file__).resolve().parents[2]
    default_video = guess_video_for_run(run_name or run_dir.name, repo)
    video_default = str(default_video) if default_video else ""
    video_path = Path(
        st.sidebar.text_input("Video file", value=video_default, key="review_video")
    )
    if not video_path.is_file():
        st.warning("Set a valid video path in the sidebar to scrub frames.")
        return

    frame_ids = sorted(int(x) for x in frame_df["frame_id"].unique())
    if not frame_ids:
        st.warning("No frame_id values in frame_data.csv")
        return
    ball_frames = sorted(
        int(x) for x in frame_df.loc[frame_df["Player_ID"] == -1, "frame_id"].unique()
    )

    st.sidebar.subheader("Verify overlays")
    show_dets = st.sidebar.checkbox("RF-DETR boxes (truth check)", value=True, key="show_dets")
    show_map_ball = st.sidebar.checkbox(
        "MAP-BALL X on video (debug)", value=False, key="show_map_ball_off"
    )
    show_map_players = st.sidebar.checkbox(
        "Player map dots on video (debug)", value=False, key="show_map_players"
    )
    show_zoom = st.sidebar.checkbox("Ball zoom crop", value=False, key="show_zoom")
    only_ball = st.sidebar.checkbox("Only frames with exported ball", value=True, key="only_ball")
    det_thr = st.sidebar.slider("Detect thr", 0.05, 0.5, 0.15, 0.05, key="det_thr")
    play_fps = st.sidebar.slider(
        "Play speed (UI fps)", 0.5, 8.0, 2.0, 0.5, key="play_fps",
        help="Play runs RF-DETR each frame (boxes). Keep speed low or raise stride.",
    )
    play_stride = st.sidebar.slider(
        "Play stride (frames in list)", 1, 10, 1, 1, key="play_stride",
    )

    nav = ball_frames if (only_ball and ball_frames) else frame_ids
    if "verify_nav_i" not in st.session_state:
        st.session_state.verify_nav_i = len(nav) // 2
    if "verify_playing" not in st.session_state:
        st.session_state.verify_playing = False
    st.session_state.verify_nav_i = int(
        np.clip(st.session_state.verify_nav_i, 0, max(0, len(nav) - 1))
    )

    playing = bool(st.session_state.verify_playing)
    dets_enabled = True if playing else bool(show_dets)
    # Never show MAP-BALL during play; paused only if debug checkbox on
    map_ball_on = (not playing) and bool(show_map_ball)
    map_players_on = (not playing) and bool(show_map_players)

    cplay, cprev, cind, cnext = st.columns([1, 1, 2, 1])
    with cplay:
        label = "⏸ Pause" if playing else "▶ Play"
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.verify_playing = not playing
            st.rerun()
    with cprev:
        if st.button("◀ Prev", use_container_width=True, disabled=playing):
            st.session_state.verify_nav_i = max(0, st.session_state.verify_nav_i - 1)
    with cnext:
        if st.button("Next ▶", use_container_width=True, disabled=playing):
            st.session_state.verify_nav_i = min(len(nav) - 1, st.session_state.verify_nav_i + 1)
    frame_id = int(nav[st.session_state.verify_nav_i])
    with cind:
        if playing:
            st.caption(f"Playing… frame **{frame_id}**")
        else:
            frame_id = st.slider(
                "Frame",
                min_value=int(nav[0]),
                max_value=int(nav[-1]),
                value=frame_id,
                step=1,
                key="review_frame_slider",
            )
            frame_id = min(nav, key=lambda f: abs(f - int(frame_id)))
            st.session_state.verify_nav_i = nav.index(frame_id)

    st.caption(
        f"Nav {st.session_state.verify_nav_i + 1}/{len(nav)} · "
        f"{'ball-export' if only_ball and ball_frames else 'all-export'}"
        + (" · **PLAYING** (boxes on, MAP-BALL off)" if playing else "")
    )

    events_here = [
        e
        for e in events
        if abs(int(e.get("start_frame", -10**9)) - int(frame_id)) <= 2
    ]
    rows = rows_for_frame(frame_df, frame_id)

    try:
        frame, fps, nframes = read_video_frame(video_path, frame_id)
    except Exception as exc:
        st.error(f"Video read failed: {exc}")
        return

    t_sec = float(frame_id) / max(float(fps), 1.0)
    st.caption(f"Frame **{frame_id}** · ~{t_sec:.1f}s · source {fps:.0f} fps · file has {nframes} frames")

    H_inv, calib = load_H_inv(video_path)
    calib_wh = (calib or {}).get("image_wh") or [frame.shape[1], frame.shape[0]]
    vis = frame.copy()

    dets = []
    if dets_enabled:
        player_ckpt = str(repo / "models/people_after_100_epochs.pth")
        ball_ckpt = str(repo / "models/v12_hard_snaps/post_train/checkpoint.pth")
        cache_key = (frame_id, float(det_thr), str(video_path), "nms_v4")
        if st.session_state.get("verify_det_key") != cache_key:
            with st.spinner("Running RF-DETR on this frame for verification…"):
                detector = _load_verify_detector(
                    player_ckpt, ball_ckpt, float(det_thr), nms_ver="v4"
                )
                dets = detector.detect(frame)
                st.session_state.verify_dets = dets
                st.session_state.verify_det_key = cache_key
        else:
            dets = st.session_state.get("verify_dets") or []
        dets = keep_top1_ball(dets)
        vis = draw_det_boxes(vis, dets)

    if map_ball_on or map_players_on:
        if H_inv is None:
            st.warning("No Match 3 calib — cannot draw mapped dots.")
        elif map_players_on:
            vis = draw_labels_on_frame(
                vis, rows, H_inv, calib_wh, ball_only=False, dedupe_players=True
            )
        else:
            vis = draw_labels_on_frame(vis, rows, H_inv, calib_wh, ball_only=True)
    vis = draw_legend(
        vis,
        ball_only_maps=map_ball_on and not map_players_on,
        maps_on=map_ball_on or map_players_on,
    )

    n_p = sum(1 for d in dets if getattr(d, "class_name", "") != "ball" and int(d.class_id) != 1)
    n_b = sum(1 for d in dets if getattr(d, "class_name", "") == "ball" or int(d.class_id) == 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detector players", n_p)
    m2.metric("Detector balls", n_b)
    m3.metric("Export tracks", int(len(rows)))
    m4.metric("Export ball", int((rows["Player_ID"] == -1).sum()) if len(rows) else 0)

    # Full-bleed video stage
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        div[data-testid="stImage"] img {
            width: 100% !important;
            max-height: 88vh !important;
            object-fit: contain !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image(
        cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=f"{video_path.name} · frame {frame_id}",
    )

    from src.review.pitch1_panel import (
        ball_trail_from_frame_df,
        cam_label_from_video,
        draw_pitch1_ball_panel,
    )
    from src.review.multicam_fuse import fuse_frame_for_pitch

    # Sibling cams under same output root → fused Pitch 1 (falls back to this cam only)
    output_root = Path(run_dir).parent
    fused = fuse_frame_for_pitch(output_root, frame_id, primary_rows=rows)
    ball_xy = fused["ball_xy"]
    players = fused["players"]
    trail = ball_trail_from_frame_df(frame_df, frame_id, back=40)
    cam = cam_label_from_video(video_path)
    mode = (
        f"fuse n_cams={fused['n_cams']} ({','.join(fused['cams'])})"
        if fused["n_cams"] > 1
        else "single-cam export"
    )
    pitch_panel = draw_pitch1_ball_panel(
        720,
        960,
        ball_xy,
        cam=cam,
        mode=mode,
        trail=trail,
        players=players,
    )
    st.subheader("Pitch 1 — ball + players")
    if fused["n_cams"] > 1:
        st.caption(
            f"Fused across **{fused['n_cams']}** cams ({', '.join(fused['cams'])}). "
            "Blue = Team 0 · Red = Team 1 · Yellow = ball. "
            "Still not a full 22 — only mapped detections from available exports."
        )
    else:
        st.caption(
            "Single-cam export only (add sibling cam runs under the same output root to fuse). "
            "Blue = Team 0 · Red = Team 1 · Yellow = ball"
        )
    st.image(
        cv2.cvtColor(pitch_panel, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=(
            f"Pitch 1 · {mode} · frame {frame_id} · "
            f"{len(players)} players · ball={'yes' if ball_xy else 'no'}"
        ),
    )

    if show_zoom and H_inv is not None:
        with st.expander("Ball zoom", expanded=False):
            zoom_base = vis if (dets_enabled or map_ball_on or map_players_on) else frame
            zoom = ball_zoom_crop(zoom_base, rows, H_inv, calib_wh)
            st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.expander("Tables (same frame)", expanded=False):
        if len(rows):
            st.dataframe(
                rows[
                    [
                        c
                        for c in [
                            "Player_ID",
                            "Team_ID",
                            "Location_X",
                            "Location_Y",
                            "Event",
                            "confidence",
                        ]
                        if c in rows.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        if dets:
            st.caption("Detector boxes this frame")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": getattr(d, "class_name", d.class_id),
                            "conf": round(float(d.confidence), 3),
                            "bbox": tuple(round(float(v), 1) for v in d.bbox),
                        }
                        for d in dets
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # Auto-advance like video while Play is on
    if st.session_state.verify_playing:
        nxt = st.session_state.verify_nav_i + int(play_stride)
        if nxt >= len(nav):
            st.session_state.verify_playing = False
            st.toast("Playback finished")
            st.rerun()
        else:
            st.session_state.verify_nav_i = nxt
            time.sleep(1.0 / max(float(play_fps), 0.5))
            st.rerun()


def main():
    st.set_page_config(page_title="Soccer Analysis Dashboard", layout="wide")
    st.title("Soccer Analysis — Phase 1 Review")

    st.sidebar.header("Data Selection")
    default_root = "data/output/full_match_2min_partial"
    if not Path(default_root).is_dir():
        default_root = "data/output"
    output_root = st.sidebar.text_input("Output root", value=default_root)
    runs = list_run_dirs(output_root)
    if runs:
        prefer = 0
        if "P10-002" in runs:
            prefer = runs.index("P10-002")
        selected = st.sidebar.selectbox("Match run", options=runs, index=prefer)
        run_dir = Path(output_root) / selected
    else:
        selected = None
        run_dir = Path(output_root)
        st.sidebar.warning("No run folders with events.json yet")

    json_path = run_dir / "events.json"
    try:
        events = load_events(str(json_path))
        meta = load_match_meta(str(json_path))
    except Exception as exc:
        st.error(f"Failed to load events: {exc}")
        events, meta = [], {}

    checkpoints = load_checkpoints(str(run_dir / "checkpoints"))
    if checkpoints:
        st.sidebar.subheader("Checkpoints")
        pick = st.sidebar.selectbox("Checkpoint", options=["(final)"] + checkpoints, index=0)
        if pick != "(final)":
            events = load_events(str(run_dir / "checkpoints" / pick))

    if meta.get("match_id"):
        st.caption(f"Match: `{meta['match_id']}` · path `{json_path}`")

    try:
        frame_df = load_frame_data(run_dir / "frame_data.csv")
    except Exception as exc:
        st.error(f"Failed to load frame_data: {exc}")
        frame_df = pd.DataFrame()

    if not frame_df.empty:
        try:
            render_synced_frame_review(run_dir, selected or run_dir.name, frame_df, events)
        except Exception as exc:
            st.error(f"Frame review failed: {exc}")

        with st.expander("All-frames pitch density", expanded=False):
            try:
                render_frame_pitch(frame_df)
            except Exception as exc:
                st.warning(f"Pitch map skipped: {exc}")
            balls = int((frame_df["Player_ID"] == -1).sum()) if "Player_ID" in frame_df.columns else 0
            st.caption(f"{len(frame_df)} rows · {balls} ball rows")

    if events:
        st.header("Event Summary")
        render_event_summary(events)
        try:
            with st.expander("Event timeline + map", expanded=False):
                render_event_timeline(events)
                render_pitch_map(events)
        except Exception as exc:
            st.warning(f"Event plots skipped: {exc}")
        st.header("Event Details")
        render_event_table(events)
        if len(events) > 400:
            st.info(f"{len(events)} events — showing first 400 in editor.")
            render_corrections_editor(run_dir, events[:400])
        else:
            render_corrections_editor(run_dir, events)
    else:
        st.warning(f"No heuristic events in {json_path}")
        if frame_df.empty:
            st.info("Point Output root at a run folder with events.json + frame_data.csv")


if __name__ == "__main__":
    main()
