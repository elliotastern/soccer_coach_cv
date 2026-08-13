#!/usr/bin/env python3
"""Unit checks for Match 2 v10 video-system scoring (no model)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import (
    iou_xywh,
    match_tp,
    metrics_block,
    pick_selected,
    pr,
)


def test_iou_and_match():
    gt = [[10.0, 10.0, 20.0, 20.0]]
    hit = [([11.0, 11.0, 20.0, 20.0], 0.91, 20.0)]
    miss = [([200.0, 200.0, 20.0, 20.0], 0.91, 20.0)]
    assert iou_xywh(gt[0], hit[0][0]) >= 0.5
    tp, fp, fn = match_tp(gt, hit)
    assert (tp, fp, fn) == (1, 0, 0)
    tp, fp, fn = match_tp(gt, miss)
    assert (tp, fp, fn) == (0, 1, 1)


def test_pick_selected_max_conf():
    cam, pred = pick_selected(
        {
            "Cam5plus": [([0, 0, 40, 40], 0.70, 40.0)],
            "Cam4plus": [([0, 0, 20, 20], 0.92, 20.0)],
        },
        "max_conf",
    )
    assert cam == "Cam4plus"
    assert pred[1] == 0.92


def test_metrics_pass_and_hollow():
    ok = metrics_block(tp=8, fp=0, fn=2, n_emitted=8)
    assert ok["P_emit"] == 1.0
    assert ok["poc_pass_P_emit"] is True
    assert ok["hollow"] is False
    hollow = metrics_block(tp=2, fp=0, fn=8, n_emitted=2)
    assert hollow["hollow"] is True
    none = metrics_block(tp=0, fp=0, fn=10, n_emitted=0)
    assert none["P_emit"] is None
    p, r = pr(8, 2, 0)
    assert abs(p - 0.8) < 1e-9


def main() -> int:
    test_iou_and_match()
    test_pick_selected_max_conf()
    test_metrics_pass_and_hollow()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
