#!/usr/bin/env python3
"""Unit checks for selection loop (no model)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_top_left_multicam_selection_loop import (  # noqa: E402
    pick_gold_cams_only,
    pick_prefer_p10,
    pick_prefer_p10_if_close,
    variant_specs,
)


def test_pickers():
    active = {
        "P7": [([0, 0, 10, 10], 0.9, 10.0)],
        "P10": [([1, 1, 10, 10], 0.8, 10.0)],
        "P1": [([2, 2, 10, 10], 0.95, 10.0)],
    }
    cam, _ = pick_prefer_p10(active)
    assert cam == "P10"
    cam2, _ = pick_prefer_p10_if_close(active, 0.05)
    assert cam2 == "P7"  # 0.9 > 0.8+0.05
    cam3, _ = pick_prefer_p10_if_close(active, 0.20)
    assert cam3 == "P10"
    cam4, _ = pick_gold_cams_only(active)
    assert cam4 in ("P7", "P10")


def test_variant_count():
    assert len(variant_specs()) >= 10


if __name__ == "__main__":
    test_pickers()
    test_variant_count()
    print("ok")
