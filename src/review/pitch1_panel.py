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
_CAM_COL = {
    "P7": (255, 200, 80),
    "P8": (255, 80, 255),
    "P9": (80, 255, 255),
    "P10": (80, 180, 255),
}
MOSAIC_CHROME_BG = (28, 42, 32)
_CHROME_BG = MOSAIC_CHROME_BG
_TIGHT_STAT_H = 22
_TIGHT_NORTH_H = 16
_TIGHT_SOUTH_H = 16
_TIGHT_SIDE_W = 34


def _compass_labels(map_orient: str) -> tuple[str, str, str, str]:
    """Return top, bottom, west, east chrome labels."""
    if map_orient == "cw90":
        return "Left", "Right", "South", "North"
    return "North", "South", "Left", "Right"


def _wrap_tight_pitch_chrome(
    field: np.ndarray,
    n0: int,
    n1: int,
    n_players: int,
    orient_hints: bool,
    drop_top: bool = False,
    band_w: int | None = None,
    map_orient: str = "north_up",
) -> np.ndarray:
    """Stats above field; compass labels outside touchlines — nothing on the grass."""
    ph, pw = field.shape[:2]
    stat_h = _TIGHT_STAT_H
    top_h = 0 if drop_top else (_TIGHT_NORTH_H if orient_hints else 0)
    bottom_h = _TIGHT_SOUTH_H if orient_hints else 0
    side_w = _TIGHT_SIDE_W if orient_hints else 0
    grass_w = max(pw, int(band_w or 0))
    total_h = stat_h + top_h + ph + bottom_h
    total_w = side_w + grass_w + side_w
    out = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    out[:] = _CHROME_BG
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_lbl, bottom_lbl, west_lbl, east_lbl = _compass_labels(map_orient)
    if n0 or n1:
        stat = f"blue={n0}  red={n1}"
    else:
        stat = f"players={n_players}"
    cv2.putText(
        out, stat, (8, stat_h - 6), font, 0.48, (235, 235, 235), 1, cv2.LINE_AA,
    )
    fy = stat_h + top_h
    fx = side_w
    out[fy : fy + ph, fx : fx + pw] = field
    if orient_hints:
        (tw, _), _ = cv2.getTextSize(top_lbl, font, 0.46, 1)
        cv2.putText(
            out, top_lbl, (fx + (grass_w - tw) // 2, stat_h + top_h - 4),
            font, 0.46, (210, 210, 210), 1, cv2.LINE_AA,
        )
        (bw, _), _ = cv2.getTextSize(bottom_lbl, font, 0.46, 1)
        cv2.putText(
            out, bottom_lbl, (fx + (grass_w - bw) // 2, fy + ph + bottom_h - 4),
            font, 0.46, (210, 210, 210), 1, cv2.LINE_AA,
        )
        (lw, lh), _ = cv2.getTextSize(west_lbl, font, 0.44, 1)
        cv2.putText(
            out, west_lbl, (max(2, (side_w - lw) // 2), fy + (ph + lh) // 2),
            font, 0.44, (210, 210, 210), 1, cv2.LINE_AA,
        )
        (rw, rh), _ = cv2.getTextSize(east_lbl, font, 0.44, 1)
        cv2.putText(
            out, east_lbl, (fx + grass_w + max(2, (side_w - rw) // 2), fy + (ph + rh) // 2),
            font, 0.44, (210, 210, 210), 1, cv2.LINE_AA,
        )
    return out


def draw_pitch1_ball_panel(
    panel_w: int,
    panel_h: int,
    ball_xy: Optional[tuple[float, float]],
    cam: str = "",
    mode: str = "export",
    trail: Sequence[tuple[float, float]] = (),
    players: Sequence[tuple[float, float, int, int]] = (),
    player_cams: Sequence[str] = (),
    raw_player_maps: Sequence[tuple[float, float, str]] = (),
    tight: bool = False,
    orient_hints: bool = True,
    landscape: bool = False,
    field_w: int | None = None,
    field_h: int | None = None,
    band_w: int | None = None,
    drop_top: bool = False,
    drop_north: bool = False,
    map_orient: str = "north_up",
) -> np.ndarray:
    """Pitch 1 map with players (by team) + yellow ball.

    map_orient=north_up: goals on short N/S edges (landmark diagram).
    map_orient=cw90: 90° CW — Left top, Right bottom, South left, North right.
    drop_top / drop_north: skip top chrome row (Left or North) when mosaic shows it.
    """
    if drop_top or drop_north:
        drop_top = True
    vis = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    vis[:] = _CHROME_BG
    margin = 4 if tight else 28
    avail_w = panel_w - 2 * margin
    avail_h = panel_h - 2 * margin
    if field_w is not None and field_h is not None and tight:
        pw, ph = int(field_w), int(field_h)
        x0 = _TIGHT_SIDE_W
        y0 = margin + max(0, (avail_h - ph) // 2)
    elif field_w is not None and tight:
        pw = int(field_w)
        if map_orient == "cw90":
            ph = int(round(pw * PITCH_WID_M / PITCH_LEN_M))
        else:
            ph = int(round(pw * (PITCH_WID_M / PITCH_LEN_M if landscape else PITCH_LEN_M / PITCH_WID_M)))
        x0 = _TIGHT_SIDE_W
        y0 = margin + max(0, (avail_h - ph) // 2)
    else:
        if map_orient == "cw90":
            field_aspect = PITCH_LEN_M / PITCH_WID_M
        else:
            field_aspect = (
                (PITCH_LEN_M / PITCH_WID_M) if landscape else (PITCH_WID_M / PITCH_LEN_M)
            )
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
        if map_orient == "cw90":
            px = x0 + int((xm + PITCH_LEN_M / 2.0) / PITCH_LEN_M * pw)
            py = y0 + int((PITCH_WID_M / 2.0 - ym) / PITCH_WID_M * ph)
            return px, py
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
    m_per_px = PITCH_LEN_M / float(pw if map_orient == "cw90" else ph)
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

    for xm, ym, cam in raw_player_maps:
        col = _CAM_COL.get(str(cam), (140, 140, 140))
        p = m_to_px(xm, ym)
        cv2.circle(vis, p, 3 if tight else 4, col, -1)
        cv2.putText(
            vis, str(cam), (p[0] + 5, p[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA,
        )

    team_color = {0: (255, 120, 60), 1: (60, 60, 255)}
    n0 = n1 = 0
    for idx, player in enumerate(players):
        xm, ym, team, pid = player[:4]
        team_i = int(team)
        color = team_color.get(team_i, (200, 200, 200))
        if team_i == 0:
            n0 += 1
        elif team_i == 1:
            n1 += 1
        p = m_to_px(xm, ym)
        cv2.circle(vis, p, 8 if tight else 9, color, -1)
        cv2.circle(vis, p, 10 if tight else 11, (255, 255, 255), 2)
        if player_cams and idx < len(player_cams) and player_cams[idx]:
            label = str(player_cams[idx])
            lcol = _CAM_COL.get(label, (255, 255, 255))
            cv2.putText(
                vis, label, (p[0] + 10, p[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 if tight else 0.45, lcol, 1, cv2.LINE_AA,
            )
        elif not tight:
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
        crop = vis[y0 : y0 + ph, x0 : x0 + pw].copy()
        return _wrap_tight_pitch_chrome(
            crop, n0, n1, len(players), orient_hints,
            drop_top=drop_top, band_w=band_w, map_orient=map_orient,
        )

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
