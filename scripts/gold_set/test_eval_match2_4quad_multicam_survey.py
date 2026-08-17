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
    TOP_LEFT_PCAM_ONLY_POLICY_ID,
    TOP_LEFT_POLICY_ID,
    thr_for_cam,
)
from eval_match2_4quad_multicam_survey import select_share  # noqa: E402


def test_policy():
    assert TOP_LEFT_POLICY_ID == "pool8_largest_ball_p7_thr060"
    assert thr_for_cam({"_default": 0.3, "P7": 0.6}, "P7") == 0.6
    assert thr_for_cam({"_default": 0.3, "P7": 0.6}, "P10") == 0.3
    assert len(QUAD_SLOTS) == 4
    assert "Cam4plus" in SURVEY_CAMS
    assert QUAD_SLOTS[0]["locked_pick"] == "largest_ball"


def test_pick_product_floors_p7():
    from multicam_select_policy import pick_product

    # P7 larger ball but below 0.60 floor → Cam4plus wins
    cam, pred = pick_product(
        {
            "P7": [([0, 0, 40, 40], 0.55, 40.0)],
            "Cam4plus": [([0, 0, 26, 26], 0.70, 26.0)],
        }
    )
    assert cam == "Cam4plus"
    assert pred[2] == 26.0
    # P7 clears floor and is larger → P7 wins
    cam2, pred2 = pick_product(
        {
            "P7": [([0, 0, 40, 40], 0.65, 40.0)],
            "Cam4plus": [([0, 0, 26, 26], 0.90, 26.0)],
        }
    )
    assert cam2 == "P7"
    assert pred2[2] == 40.0


def test_select_share():
    n = 3
    dets = {cam: [[] for _ in range(n)] for cam in SURVEY_CAMS}
    dets["P10"][0] = [([0, 0, 10, 10], 0.9, 10.0)]
    dets["Cam4plus"][1] = [([0, 0, 10, 10], 0.95, 10.0)]
    s = select_share(dets, n, {"_default": 0.30})
    assert s["selection_counts"].get("P10", 0) == 1
    assert s["selection_counts"].get("Cam4plus", 0) == 1


def test_sticky_hysteresis_and_emit():
    from multicam_select_policy import StickyCamPicker, challenger_beats

    a = ([0, 0, 30, 30], 0.9, 30.0)
    b = ([0, 0, 40, 40], 0.85, 40.0)  # clearly larger
    assert challenger_beats(a, b) is True
    assert challenger_beats(b, a) is False

    st = StickyCamPicker(k=3, n=2)
    # lock onto Cam5
    cam, pred = st.step("Cam5plus", a)
    assert cam == "Cam5plus"
    assert st.emit(cam, pred) == (None, None)  # need N=2
    cam, pred = st.step("Cam5plus", a)
    assert st.emit(cam, pred)[0] == "Cam5plus"

    # P1 larger but only 2/3 challenger frames → stay Cam5
    cam, _ = st.step("P1", b)
    assert cam == "Cam5plus"
    cam, _ = st.step("P1", b)
    assert cam == "Cam5plus"
    cam, pred = st.step("P1", b)
    assert cam == "P1"
    assert st.emit(cam, pred) == (None, None)  # hold_streak reset
    cam, pred = st.step("P1", b)
    assert st.emit(cam, pred)[0] == "P1"


if __name__ == "__main__":
    test_policy()
    test_pick_product_floors_p7()
    test_select_share()
    test_sticky_hysteresis_and_emit()
    print("ok")
