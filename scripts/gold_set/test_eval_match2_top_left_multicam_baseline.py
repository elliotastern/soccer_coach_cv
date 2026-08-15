#!/usr/bin/env python3
"""Unit checks for multicam Top Left baseline (no model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import (  # noqa: E402
    P_CAMS,
    filter_rows,
    load_top_left_gt,
    score_max_conf,
    select_frame,
    write_baseline_report,
    write_consensus_report,
)


def test_six_pcams():
    assert P_CAMS == ["P1", "P6", "P7", "P8", "P10", "P12"]


def test_load_gt_and_filter():
    xml = """<?xml version="1.0"?>
<annotations>
  <track id="1" label="ball" source="manual">
    <box frame="0" xtl="10" ytl="10" xbr="20" ybr="20" outside="0"/>
  </track>
</annotations>
"""
    path = Path(tempfile.mkdtemp()) / "a.xml"
    path.write_text(xml, encoding="utf-8")
    gt = load_top_left_gt(path)
    assert gt[0][0] == (10.0, 10.0, 10.0, 10.0)
    rows = [((1, 2, 3, 4), 0.4, 3.0), ((5, 6, 7, 8), 0.2, 3.0)]
    assert len(filter_rows(rows, 0.3)) == 1


def test_select_min_cams():
    cam_rows = {
        "P10": [([0, 0, 10, 10], 0.9, 10.0)],
        "P1": [],
    }
    cam, _pred = select_frame(cam_rows, min_cams=1)
    assert cam == "P10"
    cam2, pred2 = select_frame(cam_rows, min_cams=2)
    assert cam2 is None and pred2 is None


def test_score_and_reports():
    dets = {cam: [[] for _ in range(300)] for cam in P_CAMS}
    # frame 0: P10 ball
    dets["P10"][0] = [([10.0, 10.0, 10.0, 10.0], 0.85, 10.0)]
    dets["P1"][0] = [([100.0, 100.0, 10.0, 10.0], 0.4, 10.0)]
    gt = {0: [(10.0, 10.0, 10.0, 10.0)]}
    a = score_max_conf(dets, gt, thr=0.30, min_cams=1)
    assert a["selection_counts"].get("P10", 0) >= 1
    soft = score_max_conf(dets, gt, thr=0.15, min_cams=2)
    assert soft["n_selected"] >= 1
    out = Path(tempfile.mkdtemp())
    write_baseline_report(
        out / "b",
        {
            "baseline_a": a,
            "baseline_b": score_max_conf(dets, gt, thr=0.30, min_cams=1, emit_thr=0.80),
        },
    )
    write_consensus_report(
        out / "c", {"baseline_a": a, "soft_consensus": soft}
    )
    assert (out / "b" / "baseline.md").is_file()
    assert (out / "c" / "consensus.md").is_file()


if __name__ == "__main__":
    test_six_pcams()
    test_load_gt_and_filter()
    test_select_min_cams()
    test_score_and_reports()
    print("ok")
