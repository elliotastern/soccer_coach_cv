#!/usr/bin/env python3
"""Unit checks for ball_sahi_next_test (no model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from run_ball_postprocessing_test import write_html  # noqa: E402
from run_ball_sahi_next_test import (  # noqa: E402
    cluster_diou_nms,
    variant_specs,
    wbf_merge,
)
from src.state.types import Detection  # noqa: E402


def test_ten_variants():
    specs = variant_specs()
    assert len(specs) == 10
    ids = [s["id"] for s in specs]
    assert ids[0] == "D1_sparse_logit"
    assert ids[-1] == "D10_entropy_crop"


def test_wbf_and_diou():
    a = Detection(1, 0.9, (100, 100, 20, 20), "ball")
    b = Detection(1, 0.8, (105, 102, 20, 20), "ball")
    fused = wbf_merge([a, b])
    assert len(fused) == 1
    kept = cluster_diou_nms([a, b], dist_px=15)
    assert len(kept) == 1


def test_html():
    specs = variant_specs()
    variants = [
        {
            "id": s["id"],
            "title": s["title"],
            "why": s["why"],
            "n_frames": 10,
            "n_raw_hits": 7,
            "n_emit_hold": 2,
            "raw_rate": 0.7,
            "emit_rate": 0.2,
            "mean_emit_conf": 0.85,
            "overlay": f"overlay/{s['id']}.mp4",
        }
        for s in specs
    ]
    out = Path(tempfile.mkdtemp())
    html = write_html(
        out,
        {
            "title": "ball_sahi_next_test",
            "clock": "0:26–0:31",
            "checkpoint": "ckpt",
            "page_note": "next",
            "variants": variants,
        },
    ).read_text()
    assert "ball_sahi_next_test" in html
    assert "D1_sparse_logit" in html


def main() -> int:
    test_ten_variants()
    test_wbf_and_diou()
    test_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
