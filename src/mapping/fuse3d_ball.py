"""3D ground-plane ball fusion with per-camera reprojection gate."""
from __future__ import annotations

import numpy as np

from src.mapping.camera_projection import (
    ensure_projection,
    pixel_to_ray,
    ray_plane_intersect,
    reproj_err_px_3d,
)
from src.mapping.match3_xy import EMIT_CONF, combined_conf, load_calib

DEFAULT_REPROJ_PX = {
    "P7": 48.0,
    "P8": 48.0,
    "P9": 48.0,
    "P10": 48.0,
    "P1": 72.0,
    "P6": 72.0,
    "P_Goal1": 80.0,
    "P_Goal2": 80.0,
}
QUAD_CAMS = ("P7", "P8", "P9", "P10")
ALL_CAMS = ("P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2")


def reproj_threshold(cam: str, overrides: dict | None = None) -> float:
    if overrides and cam in overrides:
        return float(overrides[cam])
    return float(DEFAULT_REPROJ_PX.get(cam, 64.0))


def _ground_hits(rows: list[dict]) -> list[dict]:
    hits = []
    for row in rows:
        cam = str(row.get("cam", ""))
        calib = load_calib(cam)
        if calib is None:
            continue
        calib = ensure_projection(calib) or calib
        xy = row.get("xy")
        if xy is not None:
            hits.append(
                {
                    **row,
                    "calib": calib,
                    "ground_xy": (float(xy[0]), float(xy[1])),
                }
            )
            continue
        foot = row.get("foot_px")
        if foot is None:
            continue
        ray = pixel_to_ray(calib, float(foot[0]), float(foot[1]))
        if ray is None:
            continue
        pt = ray_plane_intersect(ray[0], ray[1], z=0.0)
        if pt is None:
            continue
        hits.append({**row, "calib": calib, "ground_xy": (float(pt[0]), float(pt[1]))})
    return hits


def _weighted_xy(hits: list[dict]) -> tuple[float, float]:
    wsum = 0.0
    x = y = 0.0
    for h in hits:
        w = float(h.get("weight") or h.get("conf") or 0.0)
        if w <= 0:
            continue
        gx, gy = h["ground_xy"]
        x += w * gx
        y += w * gy
        wsum += w
    if wsum <= 0:
        gx, gy = hits[0]["ground_xy"]
        return float(gx), float(gy)
    return float(x / wsum), float(y / wsum)


def prune_reproj_3d(
    hits: list[dict], xy, z: float, reproj_overrides: dict | None = None,
) -> list[dict]:
    xyz = (float(xy[0]), float(xy[1]), float(z))
    kept = []
    for h in hits:
        cam = str(h.get("cam", ""))
        foot = h.get("foot_px")
        err = reproj_err_px_3d(h["calib"], xyz, foot)
        if err <= reproj_threshold(cam, reproj_overrides):
            kept.append(h)
    if kept:
        return kept
    return [max(hits, key=lambda r: float(r.get("conf", 0.0)))]


def fuse_balls_3d(
    rows: list[dict],
    *,
    reproj_overrides: dict | None = None,
    min_cams: int = 2,
    emit_conf: float = EMIT_CONF,
) -> dict | None:
    """Triangulate ball on z=0 plane; reproj gate; emit if conf >= emit_conf."""
    hits = _ground_hits([r for r in rows if r])
    if not hits:
        return None
    xy = _weighted_xy(hits)
    inliers = prune_reproj_3d(hits, xy, 0.0, reproj_overrides)
    xy = _weighted_xy(inliers)
    conf = combined_conf([float(h["conf"]) for h in inliers])
    agree = len(inliers) >= int(min_cams)
    if conf < float(emit_conf):
        if len(inliers) == 1:
            solo = inliers[0]
            if float(solo["conf"]) >= float(emit_conf):
                gx, gy = solo["ground_xy"]
                return {
                    "xy": (gx, gy),
                    "z": 0.0,
                    "conf": float(solo["conf"]),
                    "cam": solo["cam"],
                    "n": 1,
                    "agree": False,
                    "reproj_inliers": 1,
                    "fuse_mode": "triangulate_3d",
                }
        return None
    win = max(inliers, key=lambda r: float(r.get("support", r.get("conf", 0.0))))
    return {
        "xy": xy,
        "z": 0.0,
        "conf": conf,
        "cam": win["cam"],
        "n": len(inliers),
        "agree": agree,
        "reproj_inliers": len(inliers),
        "fuse_mode": "triangulate_3d",
    }
