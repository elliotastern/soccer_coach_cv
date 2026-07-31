"""Streamlit viewer: scrub video frames and draw local RF-DETR boxes."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.state.types import Detection
from src.perception.rfdetr_local import LocalRFDETRDetector


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_PLAYER_CKPT = REPO_ROOT / "models" / "people_after_100_epochs.pth"
DEFAULT_BALL_CKPT = REPO_ROOT / "models" / "ball_89.pth"


def list_videos(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    videos = [
        p for p in sorted(root.rglob("*.mp4"))
        if not p.name.startswith("._")
    ]
    return videos


def open_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def video_meta(cap) -> Tuple[int, float, int, int]:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, fps, width, height


def read_frame(cap, frame_id: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_id}")
    return frame


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x, y, w, h = [int(v) for v in det.bbox]
        color = (0, 255, 0) if det.class_name == "player" else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            vis, label, (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return vis


def detections_table(detections: List[Detection]) -> pd.DataFrame:
    rows = []
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        rows.append({
            "#": i + 1,
            "class": det.class_name,
            "confidence": round(det.confidence, 3),
            "x": round(x, 1),
            "y": round(y, 1),
            "w": round(w, 1),
            "h": round(h, 1),
        })
    return pd.DataFrame(rows)


@st.cache_resource
def load_detector(player_ckpt: str, ball_ckpt: str, threshold: float):
    return LocalRFDETRDetector(
        player_checkpoint=player_ckpt,
        ball_checkpoint=ball_ckpt,
        confidence_threshold=threshold,
    )


def sidebar_paths() -> Tuple[Optional[Path], float, str, str]:
    st.sidebar.header("Source")
    video_root = Path(st.sidebar.text_input(
        "Video root", value=str(DEFAULT_VIDEO_ROOT)
    ))
    videos = list_videos(video_root)
    if not videos:
        st.sidebar.error(f"No .mp4 files under {video_root}")
        return None, 0.5, str(DEFAULT_PLAYER_CKPT), str(DEFAULT_BALL_CKPT)

    labels = [str(p.relative_to(video_root)) for p in videos]
    chosen = st.sidebar.selectbox("Video", options=labels, index=0)
    video_path = video_root / chosen

    st.sidebar.header("Models")
    player_ckpt = st.sidebar.text_input("Player checkpoint", value=str(DEFAULT_PLAYER_CKPT))
    ball_ckpt = st.sidebar.text_input("Ball checkpoint", value=str(DEFAULT_BALL_CKPT))
    threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
    return video_path, threshold, player_ckpt, ball_ckpt


def main():
    st.set_page_config(page_title="Detection Preview", layout="wide")
    st.title("RF-DETR Detection Preview")
    st.caption("Local people + ball checkpoints · scrub frames · inspect boxes")

    video_path, threshold, player_ckpt, ball_ckpt = sidebar_paths()
    if video_path is None:
        st.stop()

    try:
        detector = load_detector(player_ckpt, ball_ckpt, threshold)
    except Exception as exc:
        st.error(f"Failed to load detector: {exc}")
        st.stop()

    cap = open_capture(str(video_path))
    frame_count, fps, width, height = video_meta(cap)
    st.sidebar.write(f"{width}×{height} · {frame_count} frames · {fps:.1f} fps")

    max_frame = max(frame_count - 1, 0)
    frame_id = st.slider("Frame", 0, max_frame, 0)
    auto_run = st.sidebar.checkbox("Auto-detect on frame change", value=True)
    run_clicked = st.button("Detect this frame", type="primary")

    try:
        frame = read_frame(cap, frame_id)
    finally:
        cap.release()

    cached_id = st.session_state.get("det_frame_id")
    threshold_changed = st.session_state.get("det_threshold") != threshold
    needs_detect = (
        run_clicked
        or cached_id is None
        or threshold_changed
        or (auto_run and cached_id != frame_id)
    )
    if needs_detect:
        with st.spinner("Running RF-DETR…"):
            st.session_state.det_list = detector.detect(frame)
            st.session_state.det_frame_id = frame_id
            st.session_state.det_frame = frame
            st.session_state.det_threshold = threshold
    elif cached_id != frame_id:
        st.session_state.det_list = []
        st.info("Frame changed — enable auto-detect or click **Detect this frame**.")

    detections = st.session_state.get("det_list", [])
    if st.session_state.get("det_frame_id") == frame_id:
        frame = st.session_state.get("det_frame", frame)

    players = [d for d in detections if d.class_name == "player"]
    balls = [d for d in detections if d.class_name == "ball"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(players))
    c2.metric("Balls", len(balls))
    c3.metric("Frame", frame_id)

    vis = draw_detections(frame, detections) if detections else frame
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    if detections:
        st.subheader("Detections")
        st.dataframe(detections_table(detections), use_container_width=True, hide_index=True)

        out_dir = REPO_ROOT / "reports" / "match1_detect_preview"
        if st.button("Save annotated JPEG"):
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_path.stem}_frame{frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            st.success(f"Saved {out_path}")


if __name__ == "__main__":
    main()
