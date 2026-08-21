"""Multi-cam Pitch 1 fuse for review: merge sibling cam exports by frame."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.mapping.match3_xy import fuse_balls

# Pitch-space merge radius for same person seen by two cams (meters)
PLAYER_MERGE_M = 1.8
# End-line H error is larger — wider merge inside Pitch 1 goal boxes
PLAYER_MERGE_M_BOX = 3.2
# Coach / live map: precision-first player gates (bodies look ok; map was noisy)
PLAYER_MIN_CONF = 0.50
PLAYER_MIN_H = 40.0
PLAYER_MIN_AREA = 800.0
PLAYER_SOLO_CONF = 0.50
PLAYER_GHOST_CONF = 0.55
MATCH3_CAMS = ("P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2")


def _is_ball_det(d) -> bool:
    name = str(getattr(d, "class_name", "") or "").lower()
    return name == "ball" or int(getattr(d, "class_id", -1)) == 1


def player_det_ok(d, *, min_conf: float = PLAYER_MIN_CONF) -> bool:
    """Keep person-like boxes for mosaic draw + pitch map."""
    if _is_ball_det(d):
        return False
    if float(getattr(d, "confidence", 0.0)) < float(min_conf):
        return False
    _x, _y, w, h = [float(v) for v in d.bbox]
    if h < PLAYER_MIN_H or (w * h) < PLAYER_MIN_AREA:
        return False
    # Extremely wide flat boxes are usually ads / boards
    if w > 2.8 * h:
        return False
    return True


def _merge_radius_m(xy_a, xy_b, base_m: float = PLAYER_MERGE_M) -> float:
    """Wider merge if either foot sits in a Pitch 1 goal box."""
    from src.review.team_live import which_goal_box

    if which_goal_box(xy_a) is not None or which_goal_box(xy_b) is not None:
        return max(float(base_m), float(PLAYER_MERGE_M_BOX))
    return float(base_m)


def _cluster_players(
    player_pts: list[dict],
    merge_m: float = PLAYER_MERGE_M,
) -> list[list[dict]]:
    pts = sorted(player_pts, key=lambda p: p["conf"], reverse=True)
    clusters: list[list[dict]] = []
    for p in pts:
        placed = False
        for cl in clusters:
            r = _merge_radius_m(p["xy"], cl[0]["xy"], merge_m)
            if _dist(p["xy"], cl[0]["xy"]) <= r:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


def _fuse_player_clusters(
    clusters: list[list[dict]],
    *,
    merge_m: float = PLAYER_MERGE_M,
    solo_conf: float = PLAYER_SOLO_CONF,
    ghost_conf: float = PLAYER_GHOST_CONF,
) -> list[tuple[float, float, int, int]]:
    """Max-conf xy per cluster; drop weak solos / ghosts far from strong anchors."""
    eligible: list[list[dict]] = []
    for cl in clusters:
        best = max(cl, key=lambda c: c["conf"])
        if len(cl) >= 2:
            eligible.append(cl)
        elif float(best["conf"]) >= solo_conf:
            eligible.append(cl)
    if not eligible:
        return []
    strong_xy = [
        max(cl, key=lambda c: c["conf"])["xy"]
        for cl in eligible
        if float(max(cl, key=lambda c: c["conf"])["conf"]) >= ghost_conf
    ]
    fused = []
    for i, cl in enumerate(eligible):
        best = max(cl, key=lambda c: c["conf"])
        if len(cl) == 1 and float(best["conf"]) < ghost_conf and strong_xy:
            lim = min(
                _merge_radius_m(best["xy"], s, merge_m) * 1.5 for s in strong_xy
            )
            if min(_dist(best["xy"], s) for s in strong_xy) > lim:
                continue
        # Team vote: only count confident team labels; conflict → gray
        team_votes = [int(c.get("team", -1)) for c in cl if int(c.get("team", -1)) >= 0]
        if not team_votes:
            team = -1
        elif len(set(team_votes)) == 1:
            team = team_votes[0]
        else:
            # majority; tie → gray
            c0 = team_votes.count(0)
            c1 = team_votes.count(1)
            team = 0 if c0 > c1 else (1 if c1 > c0 else -1)
        pid = int(best["pid"]) if len(cl) == 1 and int(best.get("pid", -1)) >= 0 else 20_000 + i
        fused.append(
            (float(best["xy"][0]), float(best["xy"][1]), int(team), pid)
        )
    return fused


def _cam_from_run_name(name: str) -> str:
    up = name.upper().replace("-", "_")
    for cam in MATCH3_CAMS:
        key = cam.upper().replace("-", "_")
        if up.startswith(key) or f"_{key}_" in f"_{up}_" or up.startswith(key.replace("_", "")):
            if cam == "P1" and "P10" in up:
                continue
            return cam
    return name.split("-")[0].split("_")[0]


def discover_cam_frame_csvs(output_root: Path) -> dict[str, Path]:
    """Map cam_id → frame_data.csv under an output root (sibling runs)."""
    root = Path(output_root)
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for p in sorted(root.rglob("frame_data.csv")):
        if p.name.startswith("._"):
            continue
        # skip cumulative if primary also exists
        run = p.parent.name
        if run.endswith("_cumulative"):
            continue
        cam = _cam_from_run_name(run)
        # prefer non-partial / longer path later overwritten by first sorted — keep largest file
        prev = found.get(cam)
        if prev is None or p.stat().st_size >= prev.stat().st_size:
            found[cam] = p
    return found


def load_cam_tables(cam_csvs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    from src.review.io_retry import call_with_io_retry, is_transient_io

    out = {}
    for cam, path in cam_csvs.items():
        try:
            df = call_with_io_retry(lambda p=path: pd.read_csv(p), tries=4, label=f"csv:{cam}")
        except Exception as exc:  # noqa: BLE001
            if is_transient_io(exc):
                continue
            raise
        if "frame_id" not in df.columns:
            continue
        df = df.copy()
        df["cam"] = cam
        out[cam] = df
    return out


def _rows_near_frame(df: pd.DataFrame, frame_id: int, tol: int = 2) -> pd.DataFrame:
    return df[(df["frame_id"] >= frame_id - tol) & (df["frame_id"] <= frame_id + tol)]


def _dist(a, b) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def fuse_players_at_frame(
    cam_tables: dict[str, pd.DataFrame],
    frame_id: int,
    merge_m: float = PLAYER_MERGE_M,
    tol: int = 2,
) -> list[tuple[float, float, int, int]]:
    """Return fused (x, y, team_id, fused_id) for Pitch 1 panel."""
    pts = []
    for cam, df in cam_tables.items():
        sub = _rows_near_frame(df, frame_id, tol)
        players = sub[sub["Player_ID"] != -1] if "Player_ID" in sub.columns else sub
        exact = players[players["frame_id"] == int(frame_id)]
        use = exact if len(exact) else players
        for _, r in use.iterrows():
            conf = float(r["confidence"]) if "confidence" in use.columns else 0.5
            if conf < PLAYER_MIN_CONF:
                continue
            pts.append(
                {
                    "xy": (float(r.Location_X), float(r.Location_Y)),
                    "team": int(r.Team_ID) if "Team_ID" in use.columns else -1,
                    "pid": int(r.Player_ID),
                    "conf": conf,
                    "cam": cam,
                }
            )
    if not pts:
        return []
    return _fuse_player_clusters(
        _cluster_players(pts, merge_m=merge_m),
        merge_m=merge_m,
    )


def fuse_ball_at_frame(
    cam_tables: dict[str, pd.DataFrame],
    frame_id: int,
    tol: int = 2,
) -> Optional[tuple[float, float]]:
    """Fuse ball maps across cams → one (x, y) or None."""
    rows = []
    for cam, df in cam_tables.items():
        sub = _rows_near_frame(df, frame_id, tol)
        balls = sub[sub["Player_ID"] == -1] if "Player_ID" in sub.columns else sub.iloc[0:0]
        exact = balls[balls["frame_id"] == int(frame_id)]
        use = exact if len(exact) else balls
        if len(use) == 0:
            continue
        r = use.iloc[0]
        conf = float(r["confidence"]) if "confidence" in use.columns else 0.5
        rows.append(
            {
                "xy": (float(r.Location_X), float(r.Location_Y)),
                "conf": conf,
                "weight": conf,
                "support": conf,
                "cam": cam,
            }
        )
    if not rows:
        return None
    fused = fuse_balls(rows)
    if fused is None:
        # fallback: max-conf solo even below emit gate (review coverage)
        best = max(rows, key=lambda r: r["conf"])
        return best["xy"]
    return tuple(fused["xy"])


def fuse_frame_for_pitch(
    output_root: Path,
    frame_id: int,
    primary_rows: Optional[pd.DataFrame] = None,
) -> dict:
    """Build fused pitch payload; falls back to primary_rows if only one cam."""
    cam_csvs = discover_cam_frame_csvs(output_root)
    tables = load_cam_tables(cam_csvs)
    n_cams = len(tables)
    players = fuse_players_at_frame(tables, frame_id) if n_cams else []
    ball_xy = fuse_ball_at_frame(tables, frame_id) if n_cams else None
    if n_cams <= 1 and primary_rows is not None and len(primary_rows):
        from src.review.pitch1_panel import ball_xy_from_rows, players_from_rows

        if not players:
            players = players_from_rows(primary_rows)
        if ball_xy is None:
            ball_xy = ball_xy_from_rows(primary_rows)
    return {
        "players": players,
        "ball_xy": ball_xy,
        "n_cams": n_cams,
        "cams": sorted(tables.keys()),
        "source": "export",
    }


def fuse_live_dets_for_pitch(
    dets_by_cam: dict,
    *,
    apply_undistort: bool = True,
    merge_m: float = PLAYER_MERGE_M,
    team_session=None,
) -> dict:
    """Map live RF-DETR boxes (same as mosaic) onto Pitch 1 and merge cams.

    ``dets_by_cam`` may include ``{cam}__wh`` = (w, h) of the detect frame and
    optional ``{cam}__bgr`` = detect-space frame for jersey team labeling.
    Pass ``team_session`` (TeamSession) to lock kit identity across frames.
    Use apply_undistort=True for raw mosaic pixels; False when dets are already defished.
    """
    from src.mapping.match3_xy import fuse_balls, load_calib, map_ball_box
    from src.review.team_live import label_player_pts

    if not dets_by_cam:
        return {"players": [], "ball_xy": None, "n_cams": 0, "cams": [], "source": "live"}

    cam_ids = [
        k
        for k in dets_by_cam.keys()
        if isinstance(k, str)
        and not k.endswith("__wh")
        and not k.endswith("__bgr")
        and dets_by_cam.get(k)
    ]
    player_pts = []
    ball_rows = []
    used = []
    frames_by_cam = {}
    for cam in cam_ids:
        fr = dets_by_cam.get(f"{cam}__bgr")
        if fr is not None:
            frames_by_cam[cam] = fr
        calib = load_calib(cam)
        if calib is None:
            continue
        dets = dets_by_cam.get(cam) or []
        wh = dets_by_cam.get(f"{cam}__wh")
        mapped_any = False
        for d in dets:
            if _is_ball_det(d):
                mapped = map_ball_box(
                    calib,
                    d.bbox,
                    float(d.confidence),
                    frame_wh=wh,
                    apply_undistort=apply_undistort,
                )
                if mapped is None:
                    continue
                mapped_any = True
                ball_rows.append(mapped)
                continue
            if not player_det_ok(d):
                continue
            mapped = map_ball_box(
                calib,
                d.bbox,
                float(d.confidence),
                frame_wh=wh,
                apply_undistort=apply_undistort,
            )
            if mapped is None:
                continue
            mapped_any = True
            player_pts.append(
                {
                    "xy": mapped["xy"],
                    "team": -1,
                    "pid": -1,
                    "conf": float(mapped["conf"]),
                    "cam": cam,
                    "bbox": tuple(float(v) for v in d.bbox),
                }
            )
        if mapped_any:
            used.append(cam)

    if player_pts and frames_by_cam:
        label_player_pts(player_pts, frames_by_cam, team_session=team_session)

    players = _fuse_player_clusters(
        _cluster_players(player_pts, merge_m=merge_m),
        merge_m=merge_m,
    ) if player_pts else []
    if team_session is not None and players:
        players = team_session.stabilize_fused(players)

    ball_xy = None
    if ball_rows:
        fused = fuse_balls(ball_rows)
        if fused is not None:
            ball_xy = tuple(fused["xy"])
        else:
            best = max(ball_rows, key=lambda r: r["conf"])
            ball_xy = tuple(best["xy"])

    return {
        "players": players,
        "ball_xy": ball_xy,
        "n_cams": len(used),
        "cams": sorted(used),
        "source": "live",
    }
