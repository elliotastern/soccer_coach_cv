"""Pitch 1 ball panel — same look as Match 3 pitchmap gallery (north up)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_GS = ROOT / "scripts" / "gold_set"
if str(_GS) not in sys.path:
    sys.path.insert(0, str(_GS))

from pitch1 import load_pitch1, pitch1_landmarks  # noqa: E402

_PITCH1 = load_pitch1()
PITCH_LEN_M = float(_PITCH1["length_m"])
PITCH_WID_M = float(_PITCH1["width_m"])
_PITCH_LMS = pitch1_landmarks(_PITCH1)
_CIRCLE_R = float(_PITCH1["marks"]["center_circle_radius_m"])


def draw_pitch1_ball_panel(
    panel_w: int,
    panel_h: int,
    ball_xy: Optional[tuple[float, float]],
    cam: str = "",
    mode: str = "export",
    trail: Sequence[tuple[float, float]] = (),
    players: Sequence[tuple[float, float, int, int]] = (),
    tight: bool = False,
) -> np.ndarray:
    """North up, +y left — gallery pitch with players (by team) + yellow ball.

    tight=True: crop to sidelines only (no outside margin) for compact coach map.
    """
    vis = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    vis[:] = (32, 48, 36)
    margin = 4 if tight else 28
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
    for i in range(10):
        ya = y0 + int(i * ph / 10)
        yb = y0 + int((i + 1) * ph / 10)
        color = (55, 130, 55) if i % 2 == 0 else (48, 118, 48)
        cv2.rectangle(vis, (x0, ya), (x0 + pw, yb), color, -1)

    def m_to_px(xm, ym):
        px = x0 + int((PITCH_WID_M / 2.0 - ym) / PITCH_WID_M * pw)
        py = y0 + int((PITCH_LEN_M / 2.0 - xm) / PITCH_LEN_M * ph)
        return px, py

    def line_marks(*names, color=(240, 240, 240), thick=2):
        pts = [m_to_px(*_PITCH_LMS[n]["xy"]) for n in names]
        cv2.polylines(vis, [np.array(pts, np.int32)], False, color, thick)

    def box_poly(*names, color=(240, 240, 240)):
        pts = [m_to_px(*_PITCH_LMS[n]["xy"]) for n in names]
        cv2.polylines(vis, [np.array(pts, np.int32)], True, color, 2)

    box_poly(
        "left_far_corner",
        "left_near_corner",
        "right_near_corner",
        "right_far_corner",
    )
    line_marks("halfway_far_touch", "halfway_near_touch")
    m_per_px = PITCH_LEN_M / float(ph)
    r = max(2, int(round(_CIRCLE_R / m_per_px)))
    cv2.circle(vis, m_to_px(0.0, 0.0), r, (240, 240, 240), 2)
    cv2.circle(vis, m_to_px(0.0, 0.0), 3, (240, 240, 240), -1)
    box_poly(
        "left_box_goal_near",
        "left_box_18_near",
        "left_box_18_far",
        "left_box_goal_far",
    )
    box_poly(
        "right_box_goal_near",
        "right_box_18_near",
        "right_box_18_far",
        "right_box_goal_far",
    )
    line_marks("left_post_near", "left_post_far", color=(180, 255, 180), thick=3)
    line_marks("right_post_near", "right_post_far", color=(180, 255, 180), thick=3)

    team_color = {0: (255, 120, 60), 1: (60, 60, 255)}
    n0 = n1 = 0
    for xm, ym, team, pid in players:
        team_i = int(team)
        color = team_color.get(team_i, (200, 200, 200))
        if team_i == 0:
            n0 += 1
        elif team_i == 1:
            n1 += 1
        p = m_to_px(xm, ym)
        cv2.circle(vis, p, 8 if tight else 9, color, -1)
        cv2.circle(vis, p, 10 if tight else 11, (255, 255, 255), 2)
        if not tight:
            label = f"T{team_i}#{int(pid)}"
            cv2.putText(
                vis, label, (p[0] + 12, p[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

    for xm, ym in list(trail)[-40:]:
        cv2.circle(vis, m_to_px(xm, ym), 3, (80, 180, 255), -1)
    if ball_xy is not None:
        p = m_to_px(*ball_xy)
        cv2.circle(vis, p, 9 if tight else 10, (0, 255, 255), -1)
        cv2.circle(vis, p, 11 if tight else 12, (0, 0, 0), 2)

    if tight:
        # Sidelines only — drop outside margin + verbose chrome
        crop = vis[y0 : y0 + ph, x0 : x0 + pw].copy()
        if n0 or n1:
            tag = f"N↑  blue={n0} red={n1}"
        else:
            tag = f"N↑  players={len(players)}"
        cv2.putText(
            crop, tag, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return crop

    if ball_xy is not None:
        tag = f"pitch  x={ball_xy[0]:+.1f}m  y={ball_xy[1]:+.1f}m"
    else:
        tag = "pitch  no ball"
    cv2.rectangle(vis, (8, 8), (min(panel_w - 8, 720), 102), (0, 0, 0), -1)
    cv2.putText(vis, tag, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        vis,
        f"{cam or 'none'}  map={mode}  N↑",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        vis,
        f"T0={n0} (blue)  T1={n1} (red)  players={len(players)}",
        (16, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


def ball_xy_from_rows(rows) -> Optional[tuple[float, float]]:
    if rows is None or len(rows) == 0:
        return None
    balls = rows[rows["Player_ID"] == -1] if "Player_ID" in rows.columns else rows
    if len(balls) == 0:
        return None
    r = balls.iloc[0]
    return float(r.Location_X), float(r.Location_Y)


def players_from_rows(rows) -> list[tuple[float, float, int, int]]:
    """(x, y, team_id, player_id) for mapped players this frame."""
    if rows is None or len(rows) == 0:
        return []
    players = rows[rows["Player_ID"] != -1] if "Player_ID" in rows.columns else rows
    out = []
    for _, r in players.iterrows():
        out.append(
            (
                float(r.Location_X),
                float(r.Location_Y),
                int(r.Team_ID) if "Team_ID" in players.columns else -1,
                int(r.Player_ID),
            )
        )
    return out


def ball_trail_from_frame_df(frame_df, frame_id: int, back: int = 40):
    if frame_df is None or len(frame_df) == 0:
        return []
    balls = frame_df[frame_df["Player_ID"] == -1]
    if len(balls) == 0:
        return []
    lo = int(frame_id) - int(back)
    window = balls[(balls["frame_id"] >= lo) & (balls["frame_id"] <= int(frame_id))]
    window = window.sort_values("frame_id")
    return [
        (float(r.Location_X), float(r.Location_Y))
        for _, r in window.iterrows()
    ]


def cam_label_from_video(video_path: Path) -> str:
    try:
        from raw_cam_id import cam_id_from_raw_name

        return cam_id_from_raw_name(video_path.name)
    except Exception:
        return video_path.stem
