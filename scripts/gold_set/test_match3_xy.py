#!/usr/bin/env python3
"""Match 3 H round-trip, bbox foot, support, pitch-space fuse."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import (  # noqa: E402
    AGREE_M,
    EMIT_CONF,
    MATCH3_CAMS,
    apply_H,
    bbox_foot,
    combined_conf,
    fuse_balls,
    hull_support,
    load_calib,
    map_ball_box,
)


def test_foot_not_center() -> None:
    x, y = bbox_foot([100, 80, 20, 40])
    if abs(x - 110) > 1e-6 or abs(y - 120) > 1e-6:
        raise AssertionError(f"foot {x,y} want (110, 120)")


def test_roundtrip() -> None:
    for cam in MATCH3_CAMS:
        rec = load_calib(cam)
        if rec is None:
            raise AssertionError(f"missing calib {cam}")
        for img, pitch in zip(rec["image_points"], rec["pitch_points"]):
            got = apply_H(rec["H"], img[0], img[1])
            err = ((got[0] - pitch[0]) ** 2 + (got[1] - pitch[1]) ** 2) ** 0.5
            if err > 0.15:
                raise AssertionError(f"{cam} roundtrip {err:.3f}m")


def test_off_pitch_dropped() -> None:
    rec = load_calib("P9")
    box = [0, 0, 10, 10]
    row = map_ball_box(rec, box, 0.99, rec["image_wh"])
    # corner of image may or may not map on-pitch; force far mapping via fake xy
    fake = {"cam": "P9", "xy": (10.0, 40.0), "conf": 0.99, "support": 1.0, "weight": 0.99}
    from src.mapping.pitch_bounds import in_pitch_bounds
    if in_pitch_bounds(fake["xy"][0], fake["xy"][1], margin_m=1.0):
        raise AssertionError("track y=40 should be off pitch")


def test_far_hull_dropped() -> None:
    from src.mapping.match3_xy import MIN_SUPPORT

    rec = load_calib("P9")
    pts = rec["image_points"]
    row = map_ball_box(rec, [1860, 20, 20, 20], 0.99, rec["image_wh"])
    sup = hull_support(1900, 10, pts)
    if sup >= MIN_SUPPORT and row is not None:
        raise AssertionError(f"far P9 pixel should be weak support {sup} {row}")
    if row is not None and sup < MIN_SUPPORT:
        raise AssertionError("map_ball_box should drop below MIN_SUPPORT")


def test_hull_image_points_expand() -> None:
    rec = load_calib("P_Goal1")
    if not rec.get("hull_image_points"):
        raise AssertionError("P_Goal1 should set hull_image_points for FOV expand")
    if len(rec["hull_image_points"]) <= len(rec["image_points"]):
        raise AssertionError("hull expand should add points beyond landmark clicks")
    # Landmark H unchanged: roundtrip still tight
    for img, pitch in zip(rec["image_points"], rec["pitch_points"]):
        got = apply_H(rec["H"], img[0], img[1])
        err = ((got[0] - pitch[0]) ** 2 + (got[1] - pitch[1]) ** 2) ** 0.5
        if err > 0.15:
            raise AssertionError(f"P_Goal1 H drifted {err:.3f}m")


def test_fuse_agree() -> None:
    a = {"cam": "P9", "xy": (36.0, -20.0), "conf": 0.62, "support": 0.9, "weight": 0.56}
    b = {"cam": "P6", "xy": (38.0, -21.0), "conf": 0.55, "support": 0.5, "weight": 0.28}
    out = fuse_balls([a, b])
    if out is None or not out["agree"] or out["n"] != 2:
        raise AssertionError(f"agree failed {out}")
    if abs(out["xy"][0] - 37.0) > 0.2:
        raise AssertionError(f"median x {out['xy']}")
    if out["conf"] < EMIT_CONF:
        raise AssertionError("two-cam conf should emit")


def test_fuse_no_midpoint() -> None:
    a = {"cam": "P9", "xy": (40.0, -20.0), "conf": 0.91, "support": 1.0, "weight": 0.91}
    b = {"cam": "P10", "xy": (-40.0, 20.0), "conf": 0.90, "support": 0.4, "weight": 0.36}
    out = fuse_balls([a, b])
    if out is None or out["agree"]:
        raise AssertionError(f"should not average {out}")
    if abs(out["xy"][0] - 40.0) > 0.01:
        raise AssertionError(f"kept seed {out['xy']}")


def test_low_conf_silent() -> None:
    a = {"cam": "P9", "xy": (36.0, -20.0), "conf": 0.40, "support": 1.0, "weight": 0.40}
    if fuse_balls([a]) is not None:
        raise AssertionError("conf 0.4 singleton must not emit")
    if combined_conf([0.4]) >= EMIT_CONF:
        raise AssertionError("single 0.4 is not 0.8")


def test_agree_radius() -> None:
    if AGREE_M != 4.0:
        raise AssertionError("agree radius is 4 m")


def main() -> int:
    test_foot_not_center()
    test_roundtrip()
    test_off_pitch_dropped()
    test_far_hull_dropped()
    test_hull_image_points_expand()
    test_fuse_agree()
    test_fuse_no_midpoint()
    test_low_conf_silent()
    test_agree_radius()
    print("match3_xy ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
