#!/usr/bin/env python3
"""T1: Match 3 thr is all-cam 0.30 (no Match 2 P7@0.60)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from multicam_select_policy import (  # noqa: E402
    MATCH3_THR_BY_CAM,
    TOP_LEFT_THR_BY_CAM,
    filter_active,
    thr_for_cam,
)


def test_match3_thr_no_p7_bump() -> None:
    if thr_for_cam(MATCH3_THR_BY_CAM, "P7") != 0.30:
        raise AssertionError("Match 3 P7 thr must be 0.30")
    if thr_for_cam(MATCH3_THR_BY_CAM, "P1") != 0.30:
        raise AssertionError("Match 3 default thr must be 0.30")
    if thr_for_cam(TOP_LEFT_THR_BY_CAM, "P7") != 0.60:
        raise AssertionError("Match 2 product P7 thr must stay 0.60")


def test_filter_lets_p7_through_at_030() -> None:
    dets = {
        "P7": [
            [([0, 0, 10, 10], 0.45, 10.0)],
        ],
        "P1": [
            [([0, 0, 10, 10], 0.90, 12.0)],
        ],
    }
    m2 = filter_active(dets, 0, ["P1", "P7"], TOP_LEFT_THR_BY_CAM)
    m3 = filter_active(dets, 0, ["P1", "P7"], MATCH3_THR_BY_CAM)
    if "P7" in m2:
        raise AssertionError("Match 2 should drop P7 @0.45")
    if "P7" not in m3:
        raise AssertionError("Match 3 must keep P7 @0.45")
    if "P1" not in m3:
        raise AssertionError("Match 3 must keep P1")


def main() -> int:
    test_match3_thr_no_p7_bump()
    test_filter_lets_p7_through_at_030()
    print("ok match3 thr T1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
