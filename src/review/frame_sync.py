"""Frame-sync review helpers: video + pitch overlays from export rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src.mapping.match3_xy import load_calib, load_calib_for_video

HALF_L = 53.9 / 2.0
HALF_W = 34.84 / 2.0


def guess_video_for_run(run_name: str, repo_root: Path) -> Optional[Path]:
    """Map run folder name → raw Match video when possible."""
    candidates = [
        repo_root / "data/raw/Match 3" / f"{run_name}.mp4",
        repo_root / "data/raw/Match 3" / f"{run_name.replace('_', '-')}.mp4",
    ]
    # common stems
    if run_name.startswith("P10"):
        candidates.insert(0, repo_root / "data/raw/Match 3/P10-002.mp4")
    if run_name.startswith("P1") and "P10" not in run_name:
        candidates.insert(0, repo_root / "data/raw/Match 3/P1-006.mp4")
    for p in candidates:
        if p.is_file():
            return p
    # search Match 2/3 by stem
    for folder in (repo_root / "data/raw/Match 3", repo_root / "data/raw/Match 2"):
        if not folder.is_dir():
            continue
        for p in folder.glob("*.mp4"):
            if p.name.startswith("._"):
                continue
            if run_name in p.stem or p.stem in run_name:
                return p
    return None


def load_H_inv(video_path: Path):
    calib = load_calib_for_video(str(video_path))
    if calib is None:
        # try cam from stem
        for cam in ("P10", "P1", "P6", "P7", "P8", "P9"):
            if cam.lower() in video_path.stem.lower().replace("goal", "x"):
                calib = load_calib(cam)
                break
    if calib is None:
        return None, None
    H = np.asarray(calib["H"], dtype=float)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, calib
    return H_inv, calib


def pitch_to_pixel(H_inv, x: float, y: float, frame_wh, calib_wh) -> tuple[int, int] | None:
    v = H_inv @ np.array([float(x), float(y), 1.0], dtype=float)
    if abs(v[2]) < 1e-8:
        return None
    px, py = float(v[0] / v[2]), float(v[1] / v[2])
    cw, ch = float(calib_wh[0]), float(calib_wh[1])
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    if cw > 1 and ch > 1:
        px *= fw / cw
        py *= fh / ch
    return int(round(px)), int(round(py))


def read_video_frame(video_path: Path, frame_id: int):
    """Read one frame; retry on LaCie/USB EIO."""
    from src.review.io_retry import call_with_io_retry

    def _once():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        if not ok or frame is None:
            raise ValueError(f"cannot read frame {frame_id}")
        return frame, fps, n

    return call_with_io_retry(_once, tries=5, label=f"read:{video_path.name}")


def rows_for_frame(frame_df: pd.DataFrame, frame_id: int) -> pd.DataFrame:
    return frame_df[frame_df["frame_id"] == int(frame_id)]


def draw_legend(vis: np.ndarray, ball_only_maps: bool = True, maps_on: bool = True) -> np.ndarray:
    """Burn a verification legend into the top-left of the frame."""
    out = vis.copy()
    if not maps_on:
        lines = [
            ("GREEN box = RF-DETR player", (0, 220, 0)),
            ("ORANGE box = RF-DETR ball (top-1)", (0, 165, 255)),
            ("Play mode: bounding boxes only", (200, 200, 200)),
        ]
    elif ball_only_maps:
        lines = [
            ("GREEN box = RF-DETR player (detector)", (0, 220, 0)),
            ("ORANGE box = RF-DETR ball (detector, top-1)", (0, 165, 255)),
            ("YELLOW X = MAP-BALL (exported pitch→video)", (0, 255, 255)),
            ("Players by team live on Pitch 1 panel below", (180, 180, 180)),
        ]
    else:
        lines = [
            ("GREEN box = RF-DETR player (detector)", (0, 220, 0)),
            ("ORANGE box = RF-DETR ball (detector)", (0, 165, 255)),
            ("DOT + ID = exported player map track", (255, 255, 255)),
            ("YELLOW X = MAP-BALL (exported)", (0, 255, 255)),
        ]
    y0 = 28
    cv2.rectangle(out, (8, 8), (760, 8 + 28 * len(lines) + 12), (0, 0, 0), -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            out, text, (18, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
    return out


def keep_top1_ball(detections: list) -> list:
    """Coach/gallery style: at most one ball box on video."""
    players = [
        d for d in detections
        if getattr(d, "class_name", "") != "ball" and int(getattr(d, "class_id", -1)) != 1
    ]
    balls = [
        d for d in detections
        if getattr(d, "class_name", "") == "ball" or int(getattr(d, "class_id", -1)) == 1
    ]
    if not balls:
        return players
    best = max(balls, key=lambda d: float(d.confidence))
    return players + [best]


def draw_det_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw RF-DETR boxes for verification (thick, high contrast)."""
    vis = frame.copy()
    for det in detections:
        x, y, w, h = [int(v) for v in det.bbox]
        is_ball = getattr(det, "class_name", "") == "ball" or int(det.class_id) == 1
        color = (0, 165, 255) if is_ball else (0, 220, 0)
        thick = 4 if is_ball else 3
        cv2.rectangle(vis, (x, y), (x + max(1, w), y + max(1, h)), color, thick)
        label = f"{'BALL' if is_ball else 'P'} {float(det.confidence):.2f}"
        ty = max(28, y - 10)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis, (x, ty - th - 6), (x + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def _dedupe_player_map_rows(rows: pd.DataFrame, H_inv, frame_wh, calib_wh, min_px: float = 48.0):
    """One map dot per on-pitch person when tracker spawned duplicate IDs nearby."""
    if rows is None or len(rows) == 0:
        return rows
    balls = rows[rows["Player_ID"] == -1]
    players = rows[rows["Player_ID"] != -1]
    if len(players) == 0:
        return rows
    kept_idx = []
    centers = []
    for idx, r in players.iterrows():
        pt = pitch_to_pixel(
            H_inv, float(r.Location_X), float(r.Location_Y), frame_wh, calib_wh
        )
        if pt is None:
            continue
        px, py = pt
        if any((px - cx) ** 2 + (py - cy) ** 2 < min_px ** 2 for cx, cy in centers):
            continue
        centers.append((px, py))
        kept_idx.append(idx)
    return pd.concat([players.loc[kept_idx], balls], axis=0) if kept_idx else balls


def draw_labels_on_frame(
    frame: np.ndarray,
    rows: pd.DataFrame,
    H_inv,
    calib_wh,
    dedupe_players: bool = True,
    ball_only: bool = False,
) -> np.ndarray:
    """Draw exported pitch tracks reprojected to pixels.

    ball_only=True → gallery style: only MAP-BALL (no player dots on video).
    """
    vis = frame.copy()
    fh, fw = vis.shape[:2]
    if rows is None or len(rows) == 0:
        return vis
    if ball_only:
        draw_rows = rows[rows["Player_ID"] == -1]
    elif dedupe_players:
        draw_rows = _dedupe_player_map_rows(rows, H_inv, (fw, fh), calib_wh)
    else:
        draw_rows = rows
    for _, r in draw_rows.iterrows():
        pt = pitch_to_pixel(
            H_inv, float(r.Location_X), float(r.Location_Y), (fw, fh), calib_wh
        )
        if pt is None:
            continue
        px, py = pt
        if not (0 <= px < fw and 0 <= py < fh):
            continue
        is_ball = int(r.Player_ID) == -1
        if is_ball:
            cv2.drawMarker(vis, (px, py), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 36, 3)
            cv2.circle(vis, (px, py), 18, (0, 165, 255), 3)
            label = "MAP-BALL"
            color = (0, 255, 255)
        else:
            team = int(r.Team_ID)
            color = (255, 80, 80) if team == 0 else (80, 80, 255)
            cv2.circle(vis, (px, py), 11, color, -1)
            cv2.circle(vis, (px, py), 14, (255, 255, 255), 2)
            label = f"MAP T{team}#{int(r.Player_ID)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (px + 10, py - th - 14), (px + 16 + tw, py - 4), (0, 0, 0), -1)
        cv2.putText(
            vis, label, (px + 12, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
    return vis


def ball_zoom_crop(frame: np.ndarray, rows: pd.DataFrame, H_inv, calib_wh, pad: int = 220):
    """Crop around mapped ball (or frame center) for easy visual check."""
    fh, fw = frame.shape[:2]
    balls = rows[rows["Player_ID"] == -1] if len(rows) else rows
    cx, cy = fw // 2, fh // 2
    if H_inv is not None and len(balls):
        r = balls.iloc[0]
        pt = pitch_to_pixel(
            H_inv, float(r.Location_X), float(r.Location_Y), (fw, fh), calib_wh
        )
        if pt is not None:
            cx, cy = pt
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(fw, cx + pad), min(fh, cy + pad)
    crop = frame[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (2, 2), (crop.shape[1] - 3, crop.shape[0] - 3), (0, 255, 255), 2)
    cv2.putText(
        crop, "BALL ZOOM", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    return crop


def pitch_figure_for_frame(rows: pd.DataFrame, events_here: list, go_module):
    """Build plotly pitch figure for one frame."""
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=[-HALF_L, HALF_L, HALF_L, -HALF_L, -HALF_L],
            y=[-HALF_W, -HALF_W, HALF_W, HALF_W, -HALF_W],
            mode="lines",
            name="Pitch 1",
            line=dict(color="green", width=2),
        )
    )
    players = rows[rows["Player_ID"] != -1] if len(rows) else rows
    balls = rows[rows["Player_ID"] == -1] if len(rows) else rows
    if len(players):
        fig.add_trace(
            go_module.Scatter(
                x=players["Location_X"],
                y=players["Location_Y"],
                mode="markers+text",
                text=[f"#{int(i)}" for i in players["Player_ID"]],
                textposition="top center",
                name="Players",
                marker=dict(
                    size=12,
                    color=players["Team_ID"],
                    colorscale="Bluered",
                    line=dict(width=1, color="white"),
                ),
            )
        )
    if len(balls):
        fig.add_trace(
            go_module.Scatter(
                x=balls["Location_X"],
                y=balls["Location_Y"],
                mode="markers",
                name="Ball",
                marker=dict(size=16, color="orange", symbol="circle"),
            )
        )
    for e in events_here[:40]:
        s = e.get("start_location") or {}
        t = e.get("end_location") or {}
        fig.add_trace(
            go_module.Scatter(
                x=[s.get("x", 0), t.get("x", 0)],
                y=[s.get("y", 0), t.get("y", 0)],
                mode="lines",
                line=dict(color="cyan", width=2),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Pitch 1 — this frame",
        xaxis_title="X (+north)",
        yaxis_title="Y (+left)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
