#!/usr/bin/env python3
"""Unit checks: TTA unflip and train-only winner pick."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_v10_postprocess_ablation import BASELINE, pick_winner, unflip_bbox, unflip_det
from src.state.types import Detection


def test_unflip_bbox_roundtrip():
    width = 3840.0
    box = (100.0, 50.0, 40.0, 30.0)
    flipped = unflip_bbox(box, width)
    assert abs(flipped[0] - (width - 100.0 - 40.0)) < 1e-6
    back = unflip_bbox(flipped, width)
    assert abs(back[0] - box[0]) < 1e-6
    assert back[2] == box[2]


def test_unflip_det_preserves_conf():
    det = Detection(class_id=1, confidence=0.91, bbox=(10.0, 20.0, 30.0, 30.0), class_name="ball")
    out = unflip_det(det, 100.0)
    assert out.confidence == 0.91
    assert out.class_name == "ball"
    assert abs(out.bbox[0] - 60.0) < 1e-6


def _row(name, recall, precision, fp):
    return {
        "name": name,
        "train": {"0.3": {"recall": recall, "precision": precision, "fp": fp}},
    }


def test_pick_rejects_extra_fps():
    rows = [
        _row(BASELINE, 0.90, 0.97, 2),
        _row("sahi_recover_always", 0.95, 0.91, 5),
        _row("topk3", 0.92, 0.96, 2),
    ]
    win = pick_winner(rows)
    assert win["name"] == "topk3"


def test_pick_keeps_baseline_if_no_gain():
    rows = [
        _row(BASELINE, 0.92, 0.97, 2),
        _row("topk1", 0.80, 1.00, 0),
        _row("sahi_fallback_only", 0.92, 0.97, 2),
    ]
    win = pick_winner(rows)
    assert win["name"] == BASELINE


def main() -> int:
    test_unflip_bbox_roundtrip()
    test_unflip_det_preserves_conf()
    test_pick_rejects_extra_fps()
    test_pick_keeps_baseline_if_no_gain()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
