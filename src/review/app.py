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


def _inject_scroll_fix() -> None:
    """CSS-only layout + optional sidebar hide (no components.html — avoids temp I/O)."""
    hide = bool(st.session_state.get("hide_sidebar", False))
    zoom = float(st.session_state.get("coach_view_zoom", 1.2))
    zoom = max(0.5, min(2.0, zoom))
    max_h_vh = min(96.0, 78.0 * zoom)
    hide_css = ""
    if hide:
        hide_css = """
        section[data-testid="stSidebar"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
        }
        """

    st.markdown(
        f"""
        <style>
        [data-testid="stMain"] {{
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
        }}
        [data-testid="stSidebarContent"] {{
            overflow-y: auto !important;
        }}
        div[data-testid="stAppViewContainer"] > .main .block-container,
        [data-testid="stMainBlockContainer"] {{
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        div[data-testid="stImage"] img {{
            width: 100% !important;
            max-height: {max_h_vh}vh !important;
            object-fit: contain !important;
        }}
        {hide_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _log_review_exc(where: str, exc: BaseException) -> Path:
    """Write full traceback to local SSD (not LaCie) for EIO debugging."""
    import traceback

    log = Path("/tmp/scv_frame_review_errors.log")
    try:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {where} ===\n")
            f.write(f"{type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
    except OSError:
        pass
    return log


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
    from src.review.events_bar import EVENT_TYPES, event_counts

    counts = event_counts(events)
    cols = st.columns(6)
    metrics = [("Total", len(events))] + [
        (t.capitalize(), counts.get(t, 0)) for t in EVENT_TYPES
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
    from src.review.events_bar import EVENT_COLORS_PLOTLY

    show = events[:500]
    for event in show:
        start = event.get("start_location") or {}
        end = event.get("end_location") or {}
        et = event.get("type", "")
        color = EVENT_COLORS_PLOTLY.get(et, "gray")
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


def _jump_nav_to_frame(nav: List[int], frame_id: int) -> None:
    """Set verify_nav_i to closest frame in nav list."""
    if not nav:
        return
    fid = int(frame_id)
    if fid in nav:
        st.session_state.verify_nav_i = nav.index(fid)
    else:
        st.session_state.verify_nav_i = min(range(len(nav)), key=lambda i: abs(nav[i] - fid))


def render_coach_guide(simple: bool) -> None:
    if not simple:
        return
    from src.review.coach_ux import GUIDE_STEPS

    with st.expander("How to use this screen (3 steps)", expanded=True):
        for i, step in enumerate(GUIDE_STEPS, 1):
            st.markdown(f"{i}. {step}")


def render_growth_label_panel(
    run_dir: Path,
    frame_id: int,
    events: List[Dict],
    nav: List[int],
    simple: bool,
) -> None:
    """Per-frame human labels → labels.json (growth engineering)."""
    from src.review.coach_ux import (
        BALL_VISIBLE_LABELS,
        QA_LABELS,
        ball_visible_options,
        format_ball_visible,
        format_qa,
        qa_index,
        qa_options,
    )
    from src.review.frame_labels import (
        flagged_frames,
        get_frame_label,
        label_stats,
        load_labels,
        low_conf_event_frames,
        save_labels,
        set_frame_label,
    )

    title = "Quick check — this moment" if simple else "Frame labels"
    st.subheader(title)
    if simple:
        st.caption("Your answers are saved when you click **Save this frame**.")
    else:
        st.caption("Growth engineering — persists to `labels.json`.")

    from src.review.events_bar import events_at_frame

    data = load_labels(run_dir)
    cur = get_frame_label(data, frame_id)
    stats = label_stats(data)
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Frames you reviewed", stats["reviewed"])
    lc2.metric("Flagged", stats["flagged"])
    lc3.metric("Ball issues", stats["bad_ball"])

    events_here = events_at_frame(events, frame_id)
    low_here = [e for e in events_here if float(e.get("confidence", 0)) < 0.80]

    if events_here:
        chips = " · ".join(
            f"**{e.get('type', '?')}** ({float(e.get('confidence', 0)):.2f})"
            for e in events_here
        )
        st.markdown(f"Events at this moment: {chips}")
    vis_opts = ball_visible_options()
    qa_opts = qa_options()

    c1, c2, c3 = st.columns(3)
    with c1:
        if simple:
            ball_visible = st.radio(
                "Can you see the ball?",
                vis_opts,
                index=vis_opts.index(cur["ball_visible"])
                if cur["ball_visible"] in vis_opts
                else 2,
                format_func=format_ball_visible,
                key=f"lbl_ball_vis_{frame_id}",
            )
            ball_box_ok = st.radio(
                "Is the orange box on the ball?",
                qa_opts,
                index=qa_index(cur["ball_box_ok"]),
                format_func=format_qa,
                key=f"lbl_ball_box_{frame_id}",
            )
        else:
            ball_visible = st.selectbox(
                "Ball visible?",
                vis_opts,
                index=vis_opts.index(cur["ball_visible"])
                if cur["ball_visible"] in vis_opts
                else 2,
                key=f"lbl_ball_vis_{frame_id}",
            )
            ball_box_ok = st.selectbox(
                "Ball box OK?",
                qa_opts,
                index=qa_index(cur["ball_box_ok"]),
                key=f"lbl_ball_box_{frame_id}",
            )
    with c2:
        if simple:
            pitch_ball_ok = st.radio(
                "Is the yellow dot on the map in the right place?",
                qa_opts,
                index=qa_index(cur["pitch_ball_ok"]),
                format_func=format_qa,
                key=f"lbl_pitch_ball_{frame_id}",
            )
            team_ok = st.radio(
                "Do team colours on the map look right?",
                qa_opts,
                index=qa_index(cur["team_ok"]),
                format_func=format_qa,
                key=f"lbl_team_{frame_id}",
            )
        else:
            pitch_ball_ok = st.selectbox(
                "Pitch ball dot OK?",
                qa_opts,
                index=qa_index(cur["pitch_ball_ok"]),
                key=f"lbl_pitch_ball_{frame_id}",
            )
            team_ok = st.selectbox(
                "Team colors OK?",
                qa_opts,
                index=qa_index(cur["team_ok"]),
                key=f"lbl_team_{frame_id}",
            )
    with c3:
        event_default = cur["event_ok"]
        if event_default == "unset" and low_here:
            event_default = "bad"
        elif event_default == "unset" and events_here:
            event_default = "good"
        if simple:
            event_ok = st.radio(
                "Do the listed events make sense here?",
                qa_opts,
                index=qa_index(event_default),
                format_func=format_qa,
                key=f"lbl_event_{frame_id}",
            )
            flagged = st.checkbox(
                "Flag this moment for follow-up",
                value=bool(cur["flag"]),
                key=f"lbl_flag_{frame_id}",
            )
        else:
            event_ok = st.selectbox(
                "Events at frame OK?",
                qa_opts,
                index=qa_index(event_default),
                key=f"lbl_event_{frame_id}",
            )
            flagged = st.checkbox("Flag for follow-up", value=bool(cur["flag"]), key=f"lbl_flag_{frame_id}")
    note_lbl = "Notes (optional)" if simple else "Frame note"
    note = st.text_input(note_lbl, value=cur["note"], key=f"lbl_note_{frame_id}")

    bsave, bflag, blow = st.columns([2, 1, 1])
    save_lbl = "Save this frame" if simple else "Save frame label"
    with bsave:
        if st.button(save_lbl, type="primary", key=f"lbl_save_{frame_id}"):
            set_frame_label(
                data,
                frame_id,
                {
                    "ball_visible": ball_visible,
                    "ball_box_ok": ball_box_ok,
                    "pitch_ball_ok": pitch_ball_ok,
                    "team_ok": team_ok,
                    "event_ok": event_ok,
                    "flag": flagged,
                    "note": note,
                },
            )
            path = save_labels(run_dir, data)
            st.success(f"Saved — frame {frame_id}")
            if not simple:
                st.caption(str(path.name))
            st.rerun()
    with bflag:
        flagged_list = flagged_frames(data)
        flag_lbl = "Next flagged" if simple else "Next flagged ▶"
        nxt = next((f for f in flagged_list if f > frame_id), None)
        if st.button(flag_lbl, disabled=not flagged_list, key="lbl_next_flag"):
            _jump_nav_to_frame(nav, nxt if nxt is not None else flagged_list[0])
            st.rerun()
    with blow:
        low_frames = low_conf_event_frames(events)
        lc_lbl = "Next unsure event" if simple else "Next low-conf ▶"
        nxt_lc = next((f for f in low_frames if f > frame_id), None)
        if st.button(lc_lbl, disabled=not low_frames, key="lbl_next_lc"):
            _jump_nav_to_frame(nav, nxt_lc if nxt_lc is not None else low_frames[0])
            st.rerun()

    if low_here and simple:
        st.info(
            f"This moment has {len(low_here)} automatic event(s) the system wasn't sure about. "
            "You can fix them on the **Fix events** tab."
        )
    elif low_here:
        st.warning(f"{len(low_here)} event(s) here with conf < 0.80 — drop or fix in Events tab.")


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


def render_corrections_editor(run_dir: Path, events: List[Dict], simple: bool = True):
    title = "Fix automatic events" if simple else "Manual corrections"
    st.subheader(title)
    if simple:
        st.caption(
            "Uncheck **keep** to remove a wrong event. Edit the type or location, then **Save changes**."
        )
    else:
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
        column_config={
            "type": st.column_config.SelectboxColumn(
                "type",
                options=["pass", "dribble", "movement", "recovery", "shot"],
                required=True,
            ),
        },
    )
    notes_lbl = "Notes about your changes" if simple else "Review notes"
    notes = st.text_input(notes_lbl, value="")
    btn_lbl = "Save changes" if simple else "Persist corrections"
    if st.button(btn_lbl, type="primary"):
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
        msg = f"Saved {len(updated)} events" if simple else f"Saved {len(updated)} events → {path.name} and events.json"
        st.success(msg)
        st.rerun()


def render_synced_frame_review(
    run_dir: Path,
    run_name: str,
    frame_df: pd.DataFrame,
    events: List[Dict],
    simple: bool,
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

    header = "Watch the match" if simple else "Verify labels — video + pitch"
    st.header(header)
    render_coach_guide(simple)
    if simple:
        st.info(
            "**Left:** four camera views with player and ball boxes. "
            "**Right:** mini pitch map — yellow dot = ball, blue/red = teams. "
            "**Events bar** under the video: Pass · Dribble · Movement · Recovery · Shot."
        )
    else:
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
        st.warning("Pick a match video in the sidebar (or ask your engineer to set the default path).")
        return

    frame_ids = sorted(int(x) for x in frame_df["frame_id"].unique())
    if not frame_ids:
        st.warning("No frame_id values in frame_data.csv")
        return
    ball_frames = sorted(
        int(x) for x in frame_df.loc[frame_df["Player_ID"] == -1, "frame_id"].unique()
    )

    from src.review.cam_mosaic import VIEW_OPTIONS, build_cam_view

    whole_pitch = VIEW_OPTIONS[0]
    if simple:
        cam_view = whole_pitch
        apply_defish = True
        st.sidebar.caption("Camera view: whole pitch (locked in coach mode)")
        st.sidebar.slider(
            "Video zoom",
            min_value=0.7,
            max_value=1.5,
            value=float(st.session_state.coach_view_zoom),
            step=0.05,
            key="coach_view_zoom",
            help="Larger mosaic + pitch stack (default 1.2 = 20% bigger).",
        )
    else:
        st.sidebar.subheader("Camera stitch / filter")
        cam_view = st.sidebar.selectbox(
            "Camera view",
            options=VIEW_OPTIONS,
            index=0,
            key="cam_stitch_view_v3",
            help="cw90: Left top · South left · P10|P9 / P7|P8. See match3_camera_layout rule.",
        )
        apply_defish = st.sidebar.checkbox(
            "Defish P7–P10 in camera view",
            value=True,
            key="cam_view_defish_on",
            help="ON = product default (straighter pitch). Boxes run after defish. Off only for raw A/B.",
        )

    st.sidebar.subheader("Your progress")
    from src.review.frame_labels import label_stats, load_labels

    _lbl = label_stats(load_labels(run_dir))
    st.sidebar.metric("Frames reviewed", _lbl["reviewed"])
    if _lbl["flagged"]:
        st.sidebar.caption(f"⚑ {_lbl['flagged']} flagged for follow-up")

    if simple:
        show_dets = True
        show_map_ball = False
        show_map_players = False
        show_zoom = False
        only_ball = False
        det_thr = 0.15
        play_fps = 2.0
        play_stride = 2
        with st.sidebar.expander("Advanced settings", expanded=False):
            st.caption("Only change these if you know what you're doing.")
            show_dets = st.checkbox("Show detection boxes", value=True, key="show_dets_adv")
            only_ball = st.checkbox("Only ball moments", value=False, key="only_ball_adv")
            play_fps = st.slider("Play speed", 0.5, 4.0, 2.0, 0.5, key="play_fps_adv")
            play_stride = st.slider("Skip frames when playing", 1, 10, 2, 1, key="play_stride_adv")
    else:
        st.sidebar.subheader("Verify overlays")
        show_dets = st.sidebar.checkbox(
            "RF-DETR boxes (players + ball)",
            value=True,
            key="show_dets_ball_on",
            help="ON = product default. First frame can take 20–40s (4 cams). Uncheck only if USB EIO / hung.",
        )
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
            "Play speed (UI fps)", 0.5, 8.0, 4.0, 0.5, key="play_fps",
            help="While playing: export-only (no live RF-DETR). Pause for detection boxes.",
        )
        play_stride = st.sidebar.slider(
            "Play stride (frames in list)", 1, 10, 1, 1, key="play_stride",
        )

    nav = ball_frames if (only_ball and ball_frames) else frame_ids
    if "verify_nav_i" not in st.session_state:
        st.session_state.verify_nav_i = len(nav) // 2
    if "verify_playing" not in st.session_state:
        st.session_state.verify_playing = False
    if "verify_play_slow" not in st.session_state:
        st.session_state.verify_play_slow = False
    st.session_state.verify_nav_i = int(
        np.clip(st.session_state.verify_nav_i, 0, max(0, len(nav) - 1))
    )

    playing = bool(st.session_state.verify_playing)
    play_slow = bool(st.session_state.verify_play_slow)
    # Live RF-DETR on 4 cams takes seconds/frame — smooth play uses batch export only.
    dets_enabled = bool(show_dets) and not playing
    # Never show MAP-BALL during play; paused only if debug checkbox on
    map_ball_on = (not playing) and bool(show_map_ball)
    map_players_on = (not playing) and bool(show_map_players)

    if "hide_sidebar" not in st.session_state:
        st.session_state.hide_sidebar = False

    def _stop_play():
        st.session_state.verify_playing = False
        st.session_state.verify_play_slow = False

    cplay, cslow, cprev, cnext, cind = st.columns([1, 1, 1, 1, 2])
    with cplay:
        play_lbl = "Pause" if playing and not play_slow else "Play"
        if not simple:
            play_lbl = "⏸ Pause" if playing and not play_slow else "▶ Play"
        if st.button(play_lbl, use_container_width=True, type="primary"):
            if playing and not play_slow:
                _stop_play()
            elif playing and play_slow:
                _stop_play()
            else:
                st.session_state.verify_playing = True
                st.session_state.verify_play_slow = False
            st.rerun()
    with cslow:
        slow_lbl = "Pause" if playing and play_slow else "Slow"
        if not simple:
            slow_lbl = "⏸ Pause" if playing and play_slow else "🐢 Slow"
        if st.button(slow_lbl, use_container_width=True):
            if playing and play_slow:
                _stop_play()
            else:
                st.session_state.verify_playing = True
                st.session_state.verify_play_slow = True
            st.rerun()
    with cprev:
        prev_lbl = "Previous" if simple else "◀ Prev"
        if st.button(prev_lbl, use_container_width=True, disabled=playing):
            st.session_state.verify_nav_i = max(0, st.session_state.verify_nav_i - 1)
    with cnext:
        next_lbl = "Next" if simple else "Next ▶"
        if st.button(next_lbl, use_container_width=True, disabled=playing):
            st.session_state.verify_nav_i = min(len(nav) - 1, st.session_state.verify_nav_i + 1)
    frame_id = int(nav[st.session_state.verify_nav_i])
    with cind:
        if playing:
            mode = "slow" if play_slow else "smooth"
            st.caption(f"Playing ({mode})… frame **{frame_id}**")
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

    if playing:
        if play_slow:
            play_fps_eff, play_stride_eff = 0.75, 1
        elif simple:
            play_fps_eff, play_stride_eff = 4.0, 1
        else:
            play_fps_eff = float(play_fps)
            play_stride_eff = int(play_stride)
    else:
        play_fps_eff, play_stride_eff = float(play_fps), int(play_stride)

    st.caption(
        f"Nav {st.session_state.verify_nav_i + 1}/{len(nav)} · "
        f"{'ball-export' if only_ball and ball_frames else 'all-export'}"
        + (" · **PLAYING**" if playing else "")
        + (" · slow" if play_slow else "")
        + (" · export playback" if playing else "")
        + (" · boxes on" if dets_enabled else "")
    )
    if playing:
        st.caption(
            "Smooth play shows video + batch pitch map (no live AI). "
            "**Pause** to load detection boxes — first frame can take 20–40s."
        )

    from src.review.events_bar import draw_events_bar, events_at_frame, events_up_to_frame

    events_here = events_at_frame(events, frame_id)

    rows = rows_for_frame(frame_df, frame_id)

    from src.review.io_retry import is_transient_io

    try:
        frame, fps, nframes = read_video_frame(video_path, frame_id)
    except Exception as exc:
        _log_review_exc("read_video_frame", exc)
        if is_transient_io(exc):
            _stop_play()
            st.error(f"Video I/O blip ({exc}) — paused. Retry ▶ / Next.")
        else:
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
        try:
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
        except Exception as exc:
            _log_review_exc("rfdetr_detect", exc)
            st.warning(f"RF-DETR skipped ({exc})")
            dets = []
            if is_transient_io(exc):
                st.session_state.verify_playing = False
                st.session_state.verify_play_slow = False

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

    # Full-bleed stage (scroll CSS injected once in main)

    from src.review.pitch1_panel import cam_label_from_video
    # Always reload review modules — fileWatcher was none; stale imports hide box/team fixes.
    import importlib

    from src.review import cam_mosaic as _cam_mosaic
    from src.review import multicam_fuse as _mc_fuse
    from src.review import team_live as _team_live

    importlib.reload(_team_live)
    importlib.reload(_mc_fuse)
    importlib.reload(_cam_mosaic)
    build_cam_view = _cam_mosaic.build_cam_view
    fill_quad_dets_for_pitch = _cam_mosaic.fill_quad_dets_for_pitch
    match3_videos = _cam_mosaic.match3_videos

    primary_cam = cam_label_from_video(video_path)
    output_root_view = Path(run_dir).parent
    # Mutable bag: mosaic tiles fill dets + `{cam}__wh` / `__bgr` for Pitch 1 live map
    dets_by_cam: dict = {}
    if dets_enabled and dets and not apply_defish:
        dets_by_cam[primary_cam] = dets
        dets_by_cam[f"{primary_cam}__wh"] = (int(frame.shape[1]), int(frame.shape[0]))

    def _detect_other_cam(cam: str, frame_bgr):
        """RF-DETR on the exact pixels shown in the tile (raw or already-defished)."""
        cache = st.session_state.setdefault("verify_cam_dets", {})
        h, w = frame_bgr.shape[:2]
        key = (cam, int(frame_id), float(det_thr), "nms_v4", bool(apply_defish), w, h)
        if key in cache:
            return cache[key]
        player_ckpt = str(repo / "models/people_after_100_epochs.pth")
        ball_ckpt = str(repo / "models/v12_hard_snaps/post_train/checkpoint.pth")
        detector = _load_verify_detector(
            player_ckpt, ball_ckpt, float(det_thr), nms_ver="v4"
        )
        out = keep_top1_ball(detector.detect(frame_bgr))
        cache[key] = out
        if len(cache) > 64:
            for old in list(cache.keys())[:16]:
                cache.pop(old, None)
        return out

    detect_fn = _detect_other_cam if dets_enabled else None
    det_status = st.empty() if dets_enabled else None
    if dets_enabled and det_status is not None:
        det_status.info("Detecting cameras for boxes… first load can take 20–40s. Uncheck boxes if stuck.")
    wants_quad = (
        cam_view.startswith("Whole pitch")
        or cam_view.startswith("4 quads")
        or cam_view.startswith("Best camera")
    )
    view_zoom = float(st.session_state.get("coach_view_zoom", 1.2)) if simple else 1.0
    tile_w = max(480, int(round(640 * view_zoom)))
    tile_h = max(270, int(round(360 * view_zoom)))
    try:
        mosaic, used_cams = build_cam_view(
            repo,
            cam_view,
            frame_id,
            output_root_view,
            primary_cam=primary_cam,
            tile_w=tile_w,
            tile_h=tile_h,
            dets_by_cam=dets_by_cam if dets_enabled else None,
            detect_fn=detect_fn,
            apply_defish=bool(apply_defish),
        )
    except Exception as exc:
        _log_review_exc("build_cam_view", exc)
        if is_transient_io(exc):
            _stop_play()
        if wants_quad:
            # Never collapse Whole pitch / Best-ball to a single primary cam.
            st.warning(f"Boxes failed on mosaic ({exc}) — showing 4 cams without boxes.")
            try:
                mosaic, used_cams = build_cam_view(
                    repo,
                    cam_view,
                    frame_id,
                    output_root_view,
                    primary_cam=primary_cam,
                    dets_by_cam=None,
                    detect_fn=None,
                    apply_defish=bool(apply_defish),
                )
            except Exception as exc2:
                _log_review_exc("build_cam_view_no_dets", exc2)
                st.warning(f"Camera mosaic skipped ({exc2}) — showing primary frame.")
                mosaic, used_cams = vis, [primary_cam]
        else:
            st.warning(f"Camera mosaic skipped ({exc}) — showing primary frame.")
            mosaic, used_cams = vis, [primary_cam]
    if det_status is not None:
        det_status.empty()

    # Best-ball uses whole-pitch mosaic (ball pick only). Only-* may still be one cam.
    if dets_enabled and detect_fn is not None and cam_view.startswith("Only "):
        try:
            fill_quad_dets_for_pitch(
                match3_videos(repo),
                frame_id,
                dets_by_cam,
                detect_fn,
                apply_defish=bool(apply_defish),
                single_ball=True,
            )
        except Exception as exc:
            _log_review_exc("fill_quad_dets_for_pitch", exc)

    if (
        cam_view.startswith("Only ")
        and used_cams
        and used_cams[0] == primary_cam
        and dets_enabled
    ):
        stage = vis
        stage_caption = f"{video_path.name} · frame {frame_id} · boxes on"
    else:
        stage = mosaic
        box_note = " · boxes on" if dets_enabled else ""
        if cam_view.startswith("Best camera"):
            stage_caption = (
                f"Best ball · whole pitch · frame {frame_id} · "
                f"{','.join(used_cams)}{box_note}"
            )
        else:
            stage_caption = f"{cam_view} · frame {frame_id} · {','.join(used_cams)}{box_note}"

    from src.review.pitch1_panel import (
        ball_trail_from_frame_df,
        cam_label_from_video,
        draw_pitch1_ball_panel,
        ball_xy_from_rows,
        players_from_rows,
    )
    from src.review.cam_mosaic import (
        MOSAIC_SIDE_W,
        compose_coach_stack,
        pitch_stack_metrics,
    )
    from src.review.multicam_fuse import fuse_frame_for_pitch, fuse_live_dets_for_pitch
    from src.perception.team_strategy import session_from_config  # noqa: E402

    cfg = {}
    cfg_path = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    if cfg_path.is_file():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    # Prefer live mosaic boxes → Pitch 1 (matches what coach sees). Export CSV is fallback.
    output_root = Path(run_dir).parent
    # Session-locked kits so blue/red do not swap while scrubbing
    sess_key = f"team_session::{Path(run_dir).name}"
    if st.session_state.get("_team_session_key") != sess_key:
        st.session_state._team_session_key = sess_key
        st.session_state.team_session = session_from_config(cfg)
    team_session = st.session_state.team_session
    fused = None
    if dets_enabled and dets_by_cam:
        try:
            live = fuse_live_dets_for_pitch(
                dets_by_cam,
                apply_undistort=not bool(apply_defish),
                team_session=team_session,
            )
            if live["n_cams"] > 0 and (live["players"] or live["ball_xy"]):
                fused = live
        except Exception as exc:
            _log_review_exc("fuse_live_dets_for_pitch", exc)
    if fused is None:
        try:
            fused = fuse_frame_for_pitch(output_root, frame_id, primary_rows=rows)
        except Exception as exc:
            _log_review_exc("fuse_frame_for_pitch", exc)
            st.warning(f"Multi-cam fuse skipped ({exc})")
            fused = {
                "players": players_from_rows(rows),
                "ball_xy": ball_xy_from_rows(rows),
                "n_cams": 1,
                "cams": [primary_cam],
                "source": "export",
            }
    ball_xy = fused["ball_xy"]
    players = fused["players"]
    trail = ball_trail_from_frame_df(frame_df, frame_id, back=40)
    cam = cam_label_from_video(video_path)
    src = fused.get("source", "export")
    if src == "live":
        mode = f"live boxes → pitch ({','.join(fused['cams'])})"
    elif fused["n_cams"] > 1:
        mode = f"fuse n_cams={fused['n_cams']} ({','.join(fused['cams'])})"
    else:
        mode = "single-cam export"
    # Landscape pitch below mosaic — same width + aligned grid so map connects to cams
    grid_w = max(64, stage.shape[1] - 2 * MOSAIC_SIDE_W)
    grid_h = max(64, int(round(grid_w * 270 / 480)))
    pitch_scale = 0.46 * view_zoom if simple else 0.46
    stack = pitch_stack_metrics(grid_w, grid_h, drop_top=True, scale=pitch_scale)
    pitch_panel = draw_pitch1_ball_panel(
        stack["panel_w"],
        stack["panel_h"],
        ball_xy,
        cam=cam,
        mode=mode,
        trail=trail,
        players=players,
        player_cams=fused.get("player_cams", ()),
        raw_player_maps=fused.get("player_maps_all", ()),
        tight=True,
        orient_hints=True,
        field_w=stack["field_w"],
        field_h=stack["field_h"],
        band_w=stack["band_w"],
        drop_top=True,
        map_orient=stack["map_orient"],
    )
    coach_stack = compose_coach_stack(stage, pitch_panel, connect=True)

    recent_events = events_up_to_frame(events, frame_id)
    flash = events_here[-1].get("type") if events_here else None
    bar_w = max(int(stage.shape[1]), 640)
    events_bar = draw_events_bar(bar_w, t_sec, recent_events, flash)

    st.caption(
        "Left top · South left · North right · **P10|P9 / P7|P8** · pitch map below (cw90). "
        "Hide sidebar for a bigger video."
    )
    st.image(
        cv2.cvtColor(coach_stack, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=stage_caption,
    )
    pitch_cap = (
        f"Pitch 1 below cameras · {len(players)} players · ball "
        f"{'shown' if ball_xy else 'not shown'}"
        if simple
        else f"Pitch 1 · {len(players)}p · ball={'yes' if ball_xy else 'no'}"
    )
    st.caption(pitch_cap)
    if simple:
        st.caption("Blue & red = teams · grey = unsure · yellow = ball")
    elif fused.get("source") == "live":
        st.caption(
            f"Live map {fused['n_cams']} cams · blue/red=team · gray=unsure · yellow=ball"
        )
    elif fused["n_cams"] > 1:
        st.caption(f"Fused {fused['n_cams']} cams · blue T0 · red T1")
    else:
        st.caption("Single-cam · blue T0 · red T1")
    st.image(
        cv2.cvtColor(events_bar, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption="Events — Pass · Dribble · Movement · Recovery · Shot",
    )

    if show_zoom and H_inv is not None:
        with st.expander("Ball zoom", expanded=False):
            zoom_base = vis if (dets_enabled or map_ball_on or map_players_on) else frame
            zoom = ball_zoom_crop(zoom_base, rows, H_inv, calib_wh)
            st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB), use_container_width=True)

    render_growth_label_panel(run_dir, frame_id, events, nav, simple)

    if not simple:
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
        nxt = st.session_state.verify_nav_i + int(play_stride_eff)
        if nxt >= len(nav):
            _stop_play()
            st.toast("Playback finished")
            st.rerun()
        else:
            st.session_state.verify_nav_i = nxt
            time.sleep(1.0 / max(float(play_fps_eff), 0.25))
            st.rerun()


def main():
    from src.review.coach_ux import SIMPLE_MODE_KEY, is_simple_mode

    st.set_page_config(page_title="Match Review", layout="wide", page_icon="⚽")
    if SIMPLE_MODE_KEY not in st.session_state:
        st.session_state[SIMPLE_MODE_KEY] = True
    if "hide_sidebar" not in st.session_state:
        st.session_state.hide_sidebar = False
    if "coach_view_zoom" not in st.session_state:
        st.session_state.coach_view_zoom = 1.2
    simple = is_simple_mode(st.session_state)
    _inject_scroll_fix()
    head_l, head_m, head_r = st.columns([4, 1, 1])
    with head_l:
        st.title("Match Review" if simple else "Soccer Analysis — Phase 1 Review")
        if simple:
            st.caption("Watch the match, rate what you see, save your feedback.")
    with head_m:
        coach_lbl = "Coach mode" if simple else "Expert mode"
        if st.button(coach_lbl, key="toggle_coach_mode"):
            st.session_state[SIMPLE_MODE_KEY] = not simple
            st.rerun()
    with head_r:
        side_lbl = "Show menu" if st.session_state.hide_sidebar else "Bigger view"
        if st.button(side_lbl, key="toggle_sidebar_top"):
            st.session_state.hide_sidebar = not st.session_state.hide_sidebar
            st.rerun()

    st.sidebar.header("Choose match" if simple else "Data Selection")
    default_root = os.environ.get("SOCCER_OUTPUT_ROOT", "data/output/full_match_2min")
    if not Path(default_root).is_dir():
        default_root = "data/output/full_match_2min"
    if not Path(default_root).is_dir():
        default_root = "data/output"
    if simple:
        output_root = default_root
        runs = list_run_dirs(output_root)
        if runs:
            prefer = runs.index("P10-002") if "P10-002" in runs else 0
            selected = st.sidebar.selectbox("Match", options=runs, index=prefer)
            run_dir = Path(output_root) / selected
        else:
            selected = None
            run_dir = Path(output_root)
            st.sidebar.warning("No processed matches yet — run the pipeline first.")
    else:
        output_root = st.sidebar.text_input("Output root", value=default_root)
        runs = list_run_dirs(output_root)
        if runs:
            prefer = runs.index("P10-002") if "P10-002" in runs else 0
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
    if checkpoints and not simple:
        st.sidebar.subheader("Checkpoints")
        pick = st.sidebar.selectbox("Checkpoint", options=["(final)"] + checkpoints, index=0)
        if pick != "(final)":
            events = load_events(str(run_dir / "checkpoints" / pick))

    if meta.get("match_id") and not simple:
        st.caption(f"Match: `{meta['match_id']}` · path `{json_path}`")

    try:
        frame_df = load_frame_data(run_dir / "frame_data.csv")
    except Exception as exc:
        st.error(f"Failed to load frame_data: {exc}")
        frame_df = pd.DataFrame()

    tab_watch, tab_events = st.tabs(
        ["Watch & rate", "Fix events"] if simple else ["Label & verify", "Events & corrections"]
    )

    with tab_watch:
        if not frame_df.empty:
            try:
                render_synced_frame_review(
                    run_dir, selected or run_dir.name, frame_df, events, simple
                )
            except Exception as exc:
                from src.review.io_retry import is_transient_io

                log = _log_review_exc("render_synced_frame_review", exc)
                if is_transient_io(exc):
                    st.session_state.verify_playing = False
                    st.session_state.verify_play_slow = False
                    st.error(
                        f"Video read hiccup ({exc}). Playback paused — try **Next** or refresh."
                        if simple
                        else f"Frame review hit a USB/disk I/O blip ({exc}). "
                        f"Playback paused — click ▶ or refresh. Log: `{log}`"
                    )
                else:
                    st.error(f"Something went wrong: {exc}" + ("" if simple else f"  · log `{log}`"))

            if not simple:
                with st.expander("All-frames pitch density", expanded=False):
                    try:
                        render_frame_pitch(frame_df)
                    except Exception as exc:
                        st.warning(f"Pitch map skipped: {exc}")
                    balls = (
                        int((frame_df["Player_ID"] == -1).sum())
                        if "Player_ID" in frame_df.columns
                        else 0
                    )
                    st.caption(f"{len(frame_df)} rows · {balls} ball rows")
        else:
            msg = (
                "No match data loaded yet. Ask your engineer to run the batch pipeline, "
                "or switch to **Expert mode** to pick a folder."
                if simple
                else "No frame_data.csv — run batch pipeline or pick another output root."
            )
            st.warning(msg)

    with tab_events:
        if events:
            if simple:
                st.header("Automatic events")
                st.caption(
                    "Pass · Dribble · Movement · Recovery · Shot — from batch pipeline. "
                    "Uncheck **keep** to remove wrong rows, then **Save changes**."
                )
                render_event_summary(events)
            else:
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
                render_corrections_editor(run_dir, events[:400], simple)
            else:
                render_corrections_editor(run_dir, events, simple)
        else:
            st.warning(
                "No events found for this match."
                if simple
                else f"No heuristic events in {json_path}"
            )


if __name__ == "__main__":
    main()
