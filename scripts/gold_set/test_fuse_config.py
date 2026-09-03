#!/usr/bin/env python3
"""Unit tests for fuse_config + sibling ball row merge."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.fuse_config import default_fuse_config, fuse_cams_list  # noqa: E402
from src.review.multicam_fuse import (  # noqa: E402
    ball_map_rows_at_frame,
    merge_live_and_sibling_ball_rows,
)


def test_fuse_cams_quad_default() -> None:
    cams = fuse_cams_list(default_fuse_config())
    if cams != ["P7", "P8", "P9", "P10"]:
        raise AssertionError(f"quad default wrong {cams}")


def test_fuse_cams_all() -> None:
    cams = fuse_cams_list({**default_fuse_config(), "cams": "all"})
    if len(cams) != 8 or "P1" not in cams or "P_Goal2" not in cams:
        raise AssertionError(f"all cams wrong {cams}")


def test_sibling_ball_rows_skip_current() -> None:
    df = pd.DataFrame(
        [
            {"frame_id": 10, "Player_ID": -1, "Location_X": 1.0, "Location_Y": 2.0, "confidence": 0.9},
            {"frame_id": 10, "Player_ID": 1, "Location_X": 0.0, "Location_Y": 0.0, "confidence": 0.8},
        ]
    )
    tables = {"P7": df, "P10": df.copy()}
    rows = ball_map_rows_at_frame(tables, 10, skip_cam="P10")
    cams = {r["cam"] for r in rows}
    if "P10" in cams or "P7" not in cams:
        raise AssertionError(f"skip_cam failed {cams}")


def test_merge_live_prefers_current_cam() -> None:
    live = [{"xy": (0.0, 0.0), "conf": 0.95, "cam": "P10"}]
    sib = [{"xy": (5.0, 5.0), "conf": 0.5, "cam": "P7"}]
    out = merge_live_and_sibling_ball_rows(live, sib, current_cam="P10")
    if len(out) != 2:
        raise AssertionError(f"merge count {len(out)}")


def test_hybrid_prefers_2d_when_3d_solo() -> None:
    from src.mapping.fuse_product import fuse_ball_product

    cfg = {
        "mode": "triangulate_3d",
        "ukf_enabled": False,
        "fallback_pitch_merge": True,
        "reproj_max_px": {},
    }
    rows = [
        {"xy": (0.0, 0.0), "conf": 0.95, "weight": 0.95, "support": 0.95, "cam": "P10"},
        {"xy": (0.5, 0.5), "conf": 0.90, "weight": 0.90, "support": 0.90, "cam": "P7"},
    ]
    emit, _, gap, _, _ = fuse_ball_product(rows, None, 0, cfg=cfg)
    if emit is None or gap != 0:
        raise AssertionError(f"hybrid 2-cam failed emit={emit}")
    if emit.get("fuse_mode") != "triangulate_3d+f0f3_fallback" and not emit.get("agree"):
        raise AssertionError(f"expected 2d fallback or 3d agree got {emit.get('fuse_mode')} agree={emit.get('agree')}")


def test_hybrid_prefers_2d_agree() -> None:
    from src.mapping.fuse_product import pick_3d_hybrid

    solo_3d = {"xy": (0, 0), "conf": 0.95, "agree": False, "fuse_mode": "triangulate_3d"}
    agree_2d = {"xy": (1, 1), "conf": 0.90, "agree": True}
    out = pick_3d_hybrid(solo_3d, agree_2d)
    if out is None or not out.get("agree"):
        raise AssertionError(f"expected 2d agree pick {out}")


def test_hybrid_prefers_3d_agree() -> None:
    from src.mapping.fuse_product import pick_3d_hybrid

    agree_3d = {"xy": (0, 0), "conf": 0.90, "agree": True, "fuse_mode": "triangulate_3d"}
    solo_2d = {"xy": (1, 1), "conf": 0.95, "agree": False}
    out = pick_3d_hybrid(agree_3d, solo_2d)
    if out is None or not out.get("agree") or out.get("fuse_mode") != "triangulate_3d":
        raise AssertionError(f"expected 3d agree pick {out}")


def test_hybrid_fallback_recovers_solo() -> None:
    from src.mapping.fuse_product import fuse_ball_product, pick_3d_hybrid

    cfg = {
        "mode": "triangulate_3d",
        "ukf_enabled": False,
        "fallback_pitch_merge": True,
        "reproj_max_px": {},
    }
    row = {
        "xy": (1.0, 2.0),
        "conf": 0.95,
        "weight": 0.95,
        "support": 0.95,
        "cam": "P10",
    }
    emit, _, gap, _, _ = fuse_ball_product([row], None, 0, cfg=cfg)
    if emit is None or gap != 0:
        raise AssertionError(f"hybrid solo failed emit={emit} gap={gap}")
    d3 = {"xy": (0.0, 0.0), "conf": 0.9, "agree": True, "fuse_mode": "triangulate_3d"}
    d2 = {"xy": (1.0, 1.0), "conf": 0.85, "agree": False}
    pick = pick_3d_hybrid(d3, d2)
    if pick is not d3:
        raise AssertionError("smart pick should prefer 3d when agree")


def main() -> int:
    test_fuse_cams_quad_default()
    test_fuse_cams_all()
    test_sibling_ball_rows_skip_current()
    test_merge_live_prefers_current_cam()
    test_hybrid_prefers_2d_when_3d_solo()
    test_hybrid_prefers_2d_agree()
    test_hybrid_prefers_3d_agree()
    test_hybrid_fallback_recovers_solo()
    print("fuse_config ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
