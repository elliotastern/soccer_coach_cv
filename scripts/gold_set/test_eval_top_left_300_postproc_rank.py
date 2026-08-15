#!/usr/bin/env python3
"""Unit checks for top_left_300 postproc rank (no model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_top_left_300_postproc_rank import (  # noqa: E402
    load_top_left_gt,
    rank_rows,
    score_at,
    technique_specs,
    write_report,
)


def test_thirty_techniques():
    specs = technique_specs()
    assert len(specs) == 30
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids))
    assert "baseline_topk2" in ids
    assert "sahi_fallback" in ids
    assert "sahi_recover_always" in ids
    assert "D1_sparse_logit" in ids


def test_load_gt_prefers_manual(tmp_path: Path | None = None):
    xml = """<?xml version="1.0"?>
<annotations>
  <track id="1" label="ball" source="auto">
    <box frame="0" xtl="10" ytl="10" xbr="20" ybr="20" outside="0"/>
  </track>
  <track id="2" label="ball" source="manual">
    <box frame="0" xtl="100" ytl="100" xbr="120" ybr="120" outside="0"/>
  </track>
  <track id="3" label="ball" source="auto">
    <box frame="1" xtl="1" ytl="1" xbr="5" ybr="5" outside="1"/>
  </track>
</annotations>
"""
    path = Path(tempfile.mkdtemp()) / "annotations.xml"
    path.write_text(xml, encoding="utf-8")
    gt = load_top_left_gt(path)
    assert len(gt[0]) == 1
    assert gt[0][0] == (100.0, 100.0, 20.0, 20.0)
    assert 1 not in gt


def test_score_and_rank():
    gt = {0: [(0.0, 0.0, 10.0, 10.0)], 1: [(50.0, 50.0, 10.0, 10.0)]}
    good = [
        [([0.0, 0.0, 10.0, 10.0], 0.9)],
        [([50.0, 50.0, 10.0, 10.0], 0.85)],
    ]
    bad = [
        [([200.0, 200.0, 10.0, 10.0], 0.9)],
        [],
    ]
    g = score_at(gt, good, 0.3)
    b = score_at(gt, bad, 0.3)
    assert g["tp"] == 2 and g["fp"] == 0 and g["fn"] == 0
    assert b["tp"] == 0 and b["fp"] == 1 and b["fn"] == 2
    ranked = rank_rows(
        [
            {"id": "bad", "at_0_3": b, "at_0_8": b},
            {"id": "good", "at_0_3": g, "at_0_8": g},
        ]
    )
    assert ranked[0]["id"] == "good"


def test_write_report():
    out = Path(tempfile.mkdtemp())
    payload = {
        "title": "t",
        "gold_xml": "x",
        "n_frames": 2,
        "n_gt_boxes": 1,
        "ranked": [
            {
                "id": "a",
                "title": "A",
                "why": "w",
                "family": "postproc",
                "at_0_3": {
                    "f1": 0.5,
                    "recall": 0.5,
                    "precision": 0.5,
                    "tp": 1,
                    "fp": 1,
                    "fn": 1,
                },
                "at_0_8": {"recall": 0.0, "P_emit": None},
            }
        ],
    }
    md = write_report(out, payload)
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "`a`" in text


if __name__ == "__main__":
    test_thirty_techniques()
    test_load_gt_prefers_manual()
    test_score_and_rank()
    test_write_report()
    print("ok")
