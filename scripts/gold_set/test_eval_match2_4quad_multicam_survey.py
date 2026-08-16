#!/usr/bin/env python3
"""Unit checks for 4quad survey helpers (no model)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from multicam_select_policy import (  # noqa: E402
    QUAD_SLOTS,
    SURVEY_CAMS,
    TOP_LEFT_POLICY_ID,
    thr_for_cam,
)
from eval_match2_4quad_multicam_survey import select_share  # noqa: E402


def test_policy():
    assert TOP_LEFT_POLICY_ID == "p7_thr060_others030"
    assert thr_for_cam({"_default": 0.3, "P7": 0.6}, "P7") == 0.6
    assert thr_for_cam({"_default": 0.3, "P7": 0.6}, "P10") == 0.3
    assert len(QUAD_SLOTS) == 4
    assert "Cam4plus" in SURVEY_CAMS


def test_select_share():
    n = 3
    dets = {cam: [[] for _ in range(n)] for cam in SURVEY_CAMS}
    dets["P10"][0] = [([0, 0, 10, 10], 0.9, 10.0)]
    dets["Cam4plus"][1] = [([0, 0, 10, 10], 0.95, 10.0)]
    s = select_share(dets, n, {"_default": 0.30})
    assert s["selection_counts"].get("P10", 0) == 1
    assert s["selection_counts"].get("Cam4plus", 0) == 1


if __name__ == "__main__":
    test_policy()
    test_select_share()
    print("ok")
