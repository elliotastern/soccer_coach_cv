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
    GHOST_CONF,
    HOLD_MAX_GAP,
    MATCH3_CAMS,
    MIN_SUPPORT,
    PLAYER_MIN_SUPPORT,
    apply_H,
    bbox_foot,
    combined_conf,
    fuse_balls,
    fuse_balls_with_hold,
    hull_support,
    load_calib,
    map_ball_box,
    map_player_box,
    prune_ghost_maps,
    prune_reproj_outliers,
    pitch_xy_to_pixel,
    reproj_err_px,
)


def test_foot_not_center() -> None:
    x, y = bbox_foot([100, 80, 20, 40])
    if abs(x - 110) > 1e-6 or abs(y - 120) > 1e-6:
        raise AssertionError(f"foot {x,y} want (110, 120)")


def test_foot_modes() -> None:
    box = [100, 80, 20, 40]
    if bbox_foot(box, "center") != (110.0, 100.0):
        raise AssertionError(bbox_foot(box, "center"))
    if bbox_foot(box, "inset25") != (110.0, 110.0):
        raise AssertionError(bbox_foot(box, "inset25"))
    if bbox_foot(box, "radius") != (110.0, 110.0):
        raise AssertionError(bbox_foot(box, "radius"))
    try:
        bbox_foot(box, "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_roundtrip() -> None:
    for cam in MATCH3_CAMS:
        rec = load_calib(cam)
        if rec is None:
            raise AssertionError(f"missing calib {cam}")
        errs = []
        for img, pitch in zip(rec["image_points"], rec["pitch_points"]):
            got = apply_H(rec["H"], img[0], img[1])
            err = ((got[0] - pitch[0]) ** 2 + (got[1] - pitch[1]) ** 2) ** 0.5
            errs.append(err)
        good = sum(1 for e in errs if e <= 0.25)
        if rec.get("undistort"):
            # Defish clicks can leave one outlier; H must still nail ≥3 junctions.
            if good < 3:
                raise AssertionError(f"{cam} only {good}/≥3 inliers≤0.25m {errs}")
            if sorted(errs)[2] > 0.25:
                raise AssertionError(f"{cam} 3rd-best err {sorted(errs)[2]:.3f}m")
        else:
            if max(errs) > 0.15:
                raise AssertionError(f"{cam} roundtrip {max(errs):.3f}m")


def test_undistort_params_on_defish_cams() -> None:
    from src.mapping.match3_xy import calib_undistort_params

    for cam in ("P7", "P8", "P9", "P10"):
        rec = load_calib(cam)
        if calib_undistort_params(rec) is None:
            raise AssertionError(f"{cam} should lock undistort with landmarks")
    for cam in ("P1", "P6", "P_Goal1", "P_Goal2"):
        rec = load_calib(cam)
        if calib_undistort_params(rec) is not None:
            raise AssertionError(f"{cam} should stay on raw H")


def test_undistort_px_matches_remap() -> None:
    from src.mapping.match3_xy import undistort_px

    rec = load_calib("P8")
    u = rec["undistort"]
    w, h = rec["image_wh"]
    k_mat = __import__("numpy").array(
        [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]], dtype="float64"
    )
    dist = __import__("numpy").array(
        [u["k1"], u["k2"], u["p1"], u["p2"], 0.0], dtype="float64"
    )
    import cv2
    import numpy as np

    new_k, _ = cv2.getOptimalNewCameraMatrix(k_mat, dist, (w, h), u["alpha"], (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        k_mat, dist, None, new_k, (w, h), cv2.CV_32FC1
    )
    for ux, uy in rec["image_points"]:
        ix, iy = int(round(ux)), int(round(uy))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        rx, ry = float(map1[iy, ix]), float(map2[iy, ix])
        if rx < 0 or ry < 0:
            continue
        gx, gy = undistort_px(rx, ry, w, h, u)
        err = ((gx - ux) ** 2 + (gy - uy) ** 2) ** 0.5
        if err > 2.0:
            raise AssertionError(f"P8 undistort_px err {err:.2f}px at {ux,uy}")


def test_map_ball_uses_undistort_for_raw_foot() -> None:
    """Raw pixel under a defished landmark click should map near its pitch point."""
    import cv2
    import numpy as np

    rec = load_calib("P8")
    u = rec["undistort"]
    w, h = rec["image_wh"]
    k_mat = np.array(
        [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.array([u["k1"], u["k2"], u["p1"], u["p2"], 0.0], dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k_mat, dist, (w, h), u["alpha"], (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        k_mat, dist, None, new_k, (w, h), cv2.CV_32FC1
    )
    ux, uy = rec["image_points"][0]
    pitch = rec["pitch_points"][0]
    ix, iy = int(round(ux)), int(round(uy))
    rx, ry = float(map1[iy, ix]), float(map2[iy, ix])
    # foot box: height 20 → foot at (rx, ry)
    box = [rx - 10, ry - 20, 20, 20]
    hit = map_ball_box(rec, box, 0.99, rec["image_wh"])
    if hit is None:
        raise AssertionError("defish map should hit for landmark raw foot")
    err = ((hit["xy"][0] - pitch[0]) ** 2 + (hit["xy"][1] - pitch[1]) ** 2) ** 0.5
    if err > 0.5:
        raise AssertionError(f"raw→undistort→H err {err:.3f}m")
    # Without undistort, same raw foot should be worse away from optical center
    bad = map_ball_box(rec, box, 0.99, rec["image_wh"], apply_undistort=False)
    if bad is not None:
        bad_err = ((bad["xy"][0] - pitch[0]) ** 2 + (bad["xy"][1] - pitch[1]) ** 2) ** 0.5
        if bad_err <= err + 0.05 and (abs(rx - w / 2) > 200 or abs(ry - h / 2) > 150):
            raise AssertionError(
                f"skipping undistort should hurt edge landmark ({bad_err:.3f} vs {err:.3f})"
            )


def test_defish_detect_no_double_warp() -> None:
    """Boxes in defished pixel space must not be foot-undistorted again at map."""
    rec = load_calib("P10")
    box = [960.0, 540.0, 16.0, 16.0]
    once = map_ball_box(rec, box, 0.95, rec["image_wh"], apply_undistort=False)
    twice = map_ball_box(rec, box, 0.95, rec["image_wh"], apply_undistort=True)
    if once is None or twice is None:
        return
    dx = abs(once["xy"][0] - twice["xy"][0])
    dy = abs(once["xy"][1] - twice["xy"][1])
    if dx < 0.02 and dy < 0.02:
        raise AssertionError("defish-detect pairing must differ from double-warp at map")


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
    # Far corner of frame in *defished* space (image_points space); skip
    # product undistort so we probe hull only.
    row = map_ball_box(
        rec, [1860, 20, 20, 20], 0.99, rec["image_wh"], apply_undistort=False
    )
    sup = hull_support(1900, 10, pts)
    if sup >= MIN_SUPPORT and row is not None:
        raise AssertionError(f"far P9 pixel should be weak support {sup} {row}")
    if row is not None and sup < MIN_SUPPORT:
        raise AssertionError("map_ball_box should drop below MIN_SUPPORT")


def test_p6_hull_image_points_expand() -> None:
    rec = load_calib("P6")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P6 hull_image_points should expand toward near touch")
    # Clear balls on P9-t00559 lived around py~700–900
    if hull_support(960.0, 850.0, hull) < MIN_SUPPORT:
        raise AssertionError("P6 expanded hull should cover near-touch ball zone")


def test_p1_hull_image_points_expand() -> None:
    rec = load_calib("P1")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P1 hull_image_points should expand toward near touch")
    # Kickoff P1 clear balls (4K→1080) lived around py~950–1040, px~40–520
    if hull_support(200.0, 1020.0, hull) < MIN_SUPPORT:
        raise AssertionError("P1 expanded hull should cover near-touch ball zone")


def test_p7_hull_image_points_expand() -> None:
    rec = load_calib("P7")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P7 hull_image_points should expand for player FOV")
    # Player-map H_p low_support feet clustered ~x1760–1870, y230–410
    if hull_support(1820.0, 300.0, hull) < PLAYER_MIN_SUPPORT:
        raise AssertionError("P7 expanded hull should cover right-edge player feet")


def test_p8_h_player_dual_homography() -> None:
    rec = load_calib("P8")
    if rec.get("H_player") is None:
        raise AssertionError("P8 should set H_player for depth-diverse player map")
    ball = map_ball_box(
        rec,
        [1389.6, 943.8, 30.4, 28.8],
        0.84,
        frame_wh=(1920, 1080),
        apply_undistort=False,
    )
    if ball is None:
        raise AssertionError("ball must map on H not H_player")
    ply = map_player_box(
        rec,
        [1473.2, 126.7, 53.8, 85.9],
        0.8,
        frame_wh=(1920, 1080),
        apply_undistort=False,
    )
    if ply is None:
        raise AssertionError("P8 player foot should map via H_player")


def test_p8_hull_image_points_expand() -> None:
    rec = load_calib("P8")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P8 hull_image_points should expand beyond landmarks")
    # Midfield ball feet from C1 FN audit should pass support gate
    mid_px, mid_py = 1088.0, 456.0
    if hull_support(mid_px, mid_py, hull) < MIN_SUPPORT:
        raise AssertionError("P8 expanded hull should cover midfield ball zone")


def test_p10_hull_image_points_expand() -> None:
    rec = load_calib("P10")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P10 hull_image_points should expand beyond landmarks")
    # Holdout P10 low_support clear feet clustered ~x140–915, y500–860
    if hull_support(500.0, 750.0, hull) < MIN_SUPPORT:
        raise AssertionError("P10 expanded hull should cover lower-FOV clear-ball zone")
    if hull_support(200.0, 850.0, hull) < MIN_SUPPORT:
        raise AssertionError("P10 expanded hull should cover near-touch left zone")


def test_p_goal2_hull_image_points_expand() -> None:
    rec = load_calib("P_Goal2")
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec["image_points"]):
        raise AssertionError("P_Goal2 hull_image_points should expand for player FOV")
    if hull_support(1776.7, 303.6, hull) < PLAYER_MIN_SUPPORT:
        raise AssertionError("P_Goal2 expanded hull should cover right-edge player feet")


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


def test_agree_m_override_tightens_cluster() -> None:
    """Optional agree_m A/B must not change product AGREE_M default."""
    a = {
        "cam": "P9",
        "xy": (0.0, 0.0),
        "conf": 0.90,
        "support": 1.0,
        "weight": 0.90,
    }
    b = {
        "cam": "P10",
        "xy": (3.5, 0.0),
        "conf": 0.88,
        "support": 0.9,
        "weight": 0.79,
    }
    wide = fuse_balls([a, b], agree_m=4.0)
    tight = fuse_balls([a, b], agree_m=2.5)
    if wide is None or not wide.get("agree"):
        raise AssertionError(f"4m should agree {wide}")
    if tight is None or tight.get("agree"):
        raise AssertionError(f"2.5m should solo {tight}")
    if AGREE_M != 4.0:
        raise AssertionError("product AGREE_M must stay 4.0")


def test_f1_soft_dual_fallback() -> None:
    """Weak agree cluster (combined < 0.80) must fall through to strong out-of-cluster solo."""
    a = {
        "cam": "P1",
        "xy": (1.0, 2.0),
        "conf": 0.50,
        "support": 1.0,
        "weight": 0.50,
    }
    b = {
        "cam": "P10",
        "xy": (2.0, 2.5),
        "conf": 0.40,
        "support": 0.9,
        "weight": 0.36,
    }
    strong = {
        "cam": "P6",
        "xy": (-15.0, 5.0),
        "conf": 0.91,
        "support": 0.4,
        "weight": 0.364,
    }
    out = fuse_balls([a, b, strong])
    if out is None or out.get("agree") or out["cam"] != "P6":
        raise AssertionError(f"F1 soft dual should solo P6 {out}")
    silent = fuse_balls(
        [a, b, strong], soft_dual_fallback=False, solo_max_conf=False
    )
    if silent is not None:
        raise AssertionError(f"baseline soft dual must stay silent {silent}")


def test_f2_solo_max_conf() -> None:
    """Disagree: weak high-support seed must not block strong low-support (F2)."""
    weak_seed = {
        "cam": "P1",
        "xy": (10.0, 0.0),
        "conf": 0.50,
        "support": 1.0,
        "weight": 0.50,
    }
    strong = {
        "cam": "P6",
        "xy": (-10.0, 5.0),
        "conf": 0.91,
        "support": 0.4,
        "weight": 0.364,
    }
    out = fuse_balls([weak_seed, strong])
    if out is None or out.get("agree") or out["cam"] != "P6":
        raise AssertionError(f"F2 should emit strong P6 {out}")
    old = fuse_balls([weak_seed, strong], solo_max_conf=False, soft_dual_fallback=False)
    if old is not None:
        raise AssertionError(f"seed-only solo must stay silent at 0.50 {old}")


def test_f0_hold() -> None:
    prev = {
        "xy": (3.0, 4.0),
        "conf": 0.90,
        "cam": "P6",
        "n": 1,
        "agree": False,
    }
    held = fuse_balls_with_hold(prev, [], frames_since_emit=1)
    if held is None or not held.get("hold") or held["cam"] != "P6":
        raise AssertionError(f"F0 hold failed {held}")
    if HOLD_MAX_GAP < 1:
        raise AssertionError("HOLD_MAX_GAP")
    expired = fuse_balls_with_hold(prev, [], frames_since_emit=HOLD_MAX_GAP + 1)
    if expired is not None:
        raise AssertionError(f"hold must expire {expired}")
    weak_prev = {**prev, "conf": 0.50}
    if fuse_balls_with_hold(weak_prev, [], frames_since_emit=1) is not None:
        raise AssertionError("hold must require prev conf >= EMIT_CONF")


def test_f0b_soft_hold_renew() -> None:
    """After hold expires, high-support soft near prev renews; low-support ghost does not."""
    prev = {
        "xy": (-10.5, 10.4),
        "conf": 0.90,
        "cam": "P10",
        "n": 1,
        "agree": False,
    }
    soft = {
        "cam": "P10",
        "xy": (-10.6, 10.1),
        "conf": 0.76,
        "support": 1.0,
        "weight": 0.76,
    }
    ghost = {
        "cam": "P7",
        "xy": (-20.1, -16.2),
        "conf": 0.77,
        "support": 0.43,
        "weight": 0.33,
    }
    gap = HOLD_MAX_GAP + 5
    ok = fuse_balls_with_hold(
        prev, [soft, ghost], gap, soft_hold_renew=True, soft_hold_min_conf=0.55
    )
    if ok is None or not ok.get("soft_renew"):
        raise AssertionError(f"soft renew failed {ok}")
    if abs(ok["xy"][0] - soft["xy"][0]) > 1e-6:
        raise AssertionError(f"renew should use soft xy {ok}")
    if float(ok["conf"]) < 0.89:
        raise AssertionError(f"renew must keep prev emit conf {ok}")
    no = fuse_balls_with_hold(
        prev, [ghost], gap, soft_hold_renew=True, soft_hold_min_conf=0.55
    )
    if no is not None:
        raise AssertionError(f"low-support far ghost must not renew {no}")
    off = fuse_balls_with_hold(prev, [soft], gap, soft_hold_renew=False)
    if off is not None:
        raise AssertionError(f"default soft renew off {off}")


def test_solo_min_support_blocks_ballcap() -> None:
    """P7 low-hull ballcap blocked; P10 low-hull edge ball still solos."""
    from src.mapping.match3_xy import SOLO_MIN_SUPPORT, SOLO_STRICT_CAMS, fuse_balls

    if SOLO_MIN_SUPPORT < 0.50:
        raise AssertionError(f"SOLO_MIN_SUPPORT {SOLO_MIN_SUPPORT}")
    if "P7" not in SOLO_STRICT_CAMS:
        raise AssertionError(f"SOLO_STRICT_CAMS {SOLO_STRICT_CAMS}")
    ballcap = {
        "cam": "P7",
        "xy": (-20.1, -16.2),
        "conf": 0.81,
        "support": 0.43,
        "weight": 0.35,
    }
    if fuse_balls([ballcap]) is not None:
        raise AssertionError("P7 ballcap solo must be blocked")
    edge = {
        "cam": "P10",
        "xy": (-1.1, 15.5),
        "conf": 0.85,
        "support": 0.28,
        "weight": 0.24,
    }
    ok_edge = fuse_balls([edge])
    if ok_edge is None or ok_edge["cam"] != "P10":
        raise AssertionError(f"P10 edge solo must emit {ok_edge}")
    real = {
        "cam": "P10",
        "xy": (-10.0, 10.0),
        "conf": 0.85,
        "support": 1.0,
        "weight": 0.85,
    }
    ok = fuse_balls([real])
    if ok is None or ok["cam"] != "P10":
        raise AssertionError(f"high-support solo failed {ok}")
    # Cascade: P7 ballcap must not shadow a real ≥0.80 map on another cam
    shadowed = fuse_balls([ballcap, real])
    if shadowed is None or shadowed["cam"] != "P10":
        raise AssertionError(f"cascade past P7 ballcap failed {shadowed}")


def test_f3_ghost_prune() -> None:
    """Weak far ghost must not join / derail; strong far cam kept."""
    if GHOST_CONF > 0.50:
        raise AssertionError(f"GHOST_CONF {GHOST_CONF}")
    anchor = {
        "cam": "P6",
        "xy": (-13.0, 7.0),
        "conf": 0.91,
        "support": 1.0,
        "weight": 0.91,
    }
    ghost = {
        "cam": "P1",
        "xy": (12.0, 0.0),
        "conf": 0.25,
        "support": 1.0,
        "weight": 0.25,
    }
    pruned = prune_ghost_maps([anchor, ghost])
    if len(pruned) != 1 or pruned[0]["cam"] != "P6":
        raise AssertionError(f"F3 should drop P1 ghost {pruned}")
    out = fuse_balls([anchor, ghost])
    if out is None or out.get("agree") or out["cam"] != "P6":
        raise AssertionError(f"F3 fuse should solo P6 {out}")
    # Near weak ally still allowed for dual combine path
    near = {
        "cam": "P10",
        "xy": (-12.5, 7.2),
        "conf": 0.40,
        "support": 0.8,
        "weight": 0.32,
    }
    both = prune_ghost_maps([anchor, near])
    if len(both) != 2:
        raise AssertionError(f"near weak should keep {both}")


def test_f3b_low_support_far_prune() -> None:
    """Low-hull far ballcap must not seed when a high-support map exists."""
    from src.mapping.match3_xy import prune_low_support_far

    real = {
        "cam": "P10",
        "xy": (-10.0, 10.0),
        "conf": 0.76,
        "support": 1.0,
        "weight": 0.76,
    }
    ballcap = {
        "cam": "P7",
        "xy": (-20.0, -16.0),
        "conf": 0.77,
        "support": 0.43,
        "weight": 0.33,
    }
    pruned = prune_low_support_far([real, ballcap])
    if len(pruned) != 1 or pruned[0]["cam"] != "P10":
        raise AssertionError(f"F3b should drop P7 ballcap {pruned}")
    # Emit-eligible far map kept
    strong = {**ballcap, "conf": 0.90, "weight": 0.39}
    keep = prune_low_support_far([real, strong])
    if len(keep) != 2:
        raise AssertionError(f"emit-eligible far should keep {keep}")


def test_reproj_roundtrip_low_err() -> None:
    """map_ball_box foot_px → pitch → back-project stays within a few px."""
    rec = load_calib("P10")
    if rec is None:
        raise AssertionError("missing P10")
    img = rec["image_points"][0]
    fx, fy = float(img[0]), float(img[1])
    box = [fx - 5.0, fy - 18.0, 10.0, 18.0]
    mapped = map_ball_box(rec, box, 0.99, frame_wh=rec.get("image_wh"))
    if mapped is None:
        raise AssertionError("map_ball_box failed on P10 landmark foot")
    err = reproj_err_px(rec, mapped["xy"], mapped["foot_px"])
    if err > 5.0:
        raise AssertionError(f"P10 mapped reproj err {err:.2f}px > 5")


def test_f4_solo_unchanged() -> None:
    row = {
        "cam": "P10",
        "xy": (0.0, 0.0),
        "conf": 0.9,
        "support": 1.0,
        "weight": 0.9,
        "foot_px": (960.0, 540.0),
    }
    out = prune_reproj_outliers([row], enabled=True)
    if len(out) != 1:
        raise AssertionError(f"solo row must pass F4 {out}")


def test_f4_drops_far_ghost() -> None:
    rec = load_calib("P10")
    if rec is None:
        raise AssertionError("missing P10")
    img = rec["image_points"][0]
    fx, fy = float(img[0]), float(img[1])
    box = [fx - 5.0, fy - 18.0, 10.0, 18.0]
    good = map_ball_box(rec, box, 0.92, frame_wh=rec.get("image_wh"))
    if good is None:
        raise AssertionError("good map failed")
    bad = {
        "cam": "P10",
        "xy": good["xy"],
        "conf": 0.88,
        "support": 1.0,
        "weight": 0.88,
        "foot_px": (good["foot_px"][0] + 200.0, good["foot_px"][1] + 200.0),
    }
    pruned = prune_reproj_outliers([good, bad], enabled=True, max_px=48.0)
    if len(pruned) != 1 or float(pruned[0]["conf"]) != 0.92:
        raise AssertionError(f"F4 should drop bad reproj row {pruned}")
    out = fuse_balls(
        [good, bad],
        reproj_prune=True,
        ghost_prune=True,
    )
    if out is None or out.get("agree"):
        raise AssertionError(f"F4 fuse should solo good cam {out}")


def test_soft_f4_demotes_agree_not_drop() -> None:
    """Soft F4 keeps emitting but refuses agree when reproj fails."""
    import numpy as np
    from src.mapping import match3_xy as m

    a = {
        "cam": "P10",
        "xy": (0.0, 0.0),
        "conf": 0.9,
        "support": 1.0,
        "weight": 0.9,
        "foot_px": (100.0, 100.0),
    }
    b = {
        "cam": "P9",
        "xy": (1.0, 0.0),
        "conf": 0.85,
        "support": 1.0,
        "weight": 0.85,
        "foot_px": (400.0, 100.0),
    }
    old = m.load_calib

    def _fake(cam: str):
        return {"camera": cam, "H": np.eye(3), "image_wh": [1920, 1080]}

    m.load_calib = _fake  # type: ignore
    try:
        out = fuse_balls(
            [a, b],
            ghost_prune=False,
            reproj_agree_gate=True,
            reproj_max_px=5.0,
        )
    finally:
        m.load_calib = old  # type: ignore
    if out is None:
        raise AssertionError("soft F4 should still emit")
    if out.get("agree"):
        raise AssertionError(f"soft F4 should demote agree→solo {out}")
    if out.get("n", 1) != 1:
        raise AssertionError(f"expected solo n=1 got {out}")


def main() -> int:
    test_foot_not_center()
    test_foot_modes()
    test_roundtrip()
    test_undistort_params_on_defish_cams()
    test_undistort_px_matches_remap()
    test_map_ball_uses_undistort_for_raw_foot()
    test_defish_detect_no_double_warp()
    test_off_pitch_dropped()
    test_far_hull_dropped()
    test_hull_image_points_expand()
    test_p_goal2_hull_image_points_expand()
    test_p7_hull_image_points_expand()
    test_p8_h_player_dual_homography()
    test_p8_hull_image_points_expand()
    test_p6_hull_image_points_expand()
    test_p1_hull_image_points_expand()
    test_p10_hull_image_points_expand()
    test_fuse_agree()
    test_fuse_no_midpoint()
    test_low_conf_silent()
    test_agree_radius()
    test_agree_m_override_tightens_cluster()
    test_f1_soft_dual_fallback()
    test_f2_solo_max_conf()
    test_f0_hold()
    test_f0b_soft_hold_renew()
    test_solo_min_support_blocks_ballcap()
    test_f3_ghost_prune()
    test_f3b_low_support_far_prune()
    test_reproj_roundtrip_low_err()
    test_f4_solo_unchanged()
    test_f4_drops_far_ghost()
    test_soft_f4_demotes_agree_not_drop()
    print("match3_xy ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
