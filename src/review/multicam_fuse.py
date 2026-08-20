"""Multi-cam Pitch 1 fuse for review: merge sibling cam exports by frame."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.mapping.match3_xy import fuse_balls

# Pitch-space merge radius for same person seen by two cams (meters)
PLAYER_MERGE_M = 1.8
MATCH3_CAMS = ("P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2")


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
    out = {}
    for cam, path in cam_csvs.items():
        df = pd.read_csv(path)
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
        # prefer exact frame_id when present
        exact = players[players["frame_id"] == int(frame_id)]
        use = exact if len(exact) else players
        for _, r in use.iterrows():
            conf = float(r["confidence"]) if "confidence" in use.columns else 0.5
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
    pts.sort(key=lambda p: p["conf"], reverse=True)
    clusters: list[list[dict]] = []
    for p in pts:
        placed = False
        for cl in clusters:
            if _dist(p["xy"], cl[0]["xy"]) <= merge_m:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    fused = []
    for i, cl in enumerate(clusters):
        xs = [c["xy"][0] for c in cl]
        ys = [c["xy"][1] for c in cl]
        teams = [c["team"] for c in cl if c["team"] >= 0]
        team = max(set(teams), key=teams.count) if teams else -1
        # stable-ish id from max-conf member
        best = max(cl, key=lambda c: c["conf"])
        fused.append(
            (
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                int(team),
                int(best["pid"]) if len(cl) == 1 else 10_000 + i,
            )
        )
    return fused


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
    }
