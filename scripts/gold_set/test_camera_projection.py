#!/usr/bin/env python3
"""Unit tests for camera_projection + fuse3d_ball."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.camera_projection import (  # noqa: E402
    ensure_projection,
    fit_projection_from_landmarks,
    project_world,
    reproj_err_px_3d,
    write_projection_to_calib,
)
from src.mapping.fuse3d_ball import fuse_balls_3d  # noqa: E402
from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402


def test_p10_projection_fit() -> None:
    rec = load_calib("P10")
    if rec is None:
        raise AssertionError("missing P10 calib")
    proj = fit_projection_from_landmarks(rec)
    if proj is None:
        raise AssertionError("P10 solvePnP failed")
    if float(proj["C"][2]) < 2.0:
        raise AssertionError(f"P10 camera height {proj['C'][2]} too low")


def test_p10_landmark_reproj() -> None:
    rec = load_calib("P10")
    if rec is None:
        raise AssertionError("missing P10")
    from src.mapping.camera_projection import project_pitch_to_pixel

    max_err = 0.0
    for ip, pp in zip(rec["image_points"], rec["pitch_points"]):
        rp = project_pitch_to_pixel(rec, float(pp[0]), float(pp[1]))
        if rp is None:
            raise AssertionError("project failed")
        dx = float(ip[0]) - rp[0]
        dy = float(ip[1]) - rp[1]
        max_err = max(max_err, (dx * dx + dy * dy) ** 0.5)
    if max_err > 35.0:
        raise AssertionError(f"P10 H-inv reproj {max_err:.2f}px > 35")


def test_fuse3d_solo_emit() -> None:
    rec = load_calib("P10")
    if rec is None:
        raise AssertionError("missing P10")
    img = rec["image_points"][0]
    fx, fy = float(img[0]), float(img[1])
    box = [fx - 5.0, fy - 18.0, 10.0, 18.0]
    row = map_ball_box(rec, box, 0.95, frame_wh=rec.get("image_wh"))
    if row is None:
        raise AssertionError("map failed")
    out = fuse_balls_3d([row], min_cams=2)
    if out is None or float(out["conf"]) < 0.80:
        raise AssertionError(f"solo 3d emit failed {out}")


def test_write_projection_all_cams() -> None:
    for cam in ("P10", "P7", "P8", "P9"):
        r = write_projection_to_calib(cam, dry_run=True)
        if not r.get("ok"):
            raise AssertionError(f"{cam} projection dry_run failed {r}")


def main() -> int:
    test_p10_projection_fit()
    test_p10_landmark_reproj()
    test_fuse3d_solo_emit()
    test_write_projection_all_cams()
    print("camera_projection + fuse3d ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
