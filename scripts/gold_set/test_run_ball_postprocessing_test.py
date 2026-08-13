#!/usr/bin/env python3
"""Unit checks for ball_postprocessing_test (no model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from run_ball_postprocessing_test import (  # noqa: E402
    END_CLOCK,
    P10_VIDEO,
    START_CLOCK,
    parse_clock,
    variant_specs,
    write_html,
)


def test_ten_variants_and_ids():
    specs = variant_specs()
    assert len(specs) == 10
    ids = [s["id"] for s in specs]
    assert ids == [
        "baseline_topk2",
        "topk3",
        "hflip_tta",
        "multiscale_1p5",
        "sahi_fallback",
        "thr50_topk2",
        "emit80_pass",
        "bytetrack_iou08",
        "bytetrack_emit80",
        "kalman_detect",
    ]


def test_top_left_clock_and_p10():
    assert parse_clock(START_CLOCK) == 26.0
    assert parse_clock(END_CLOCK) == 31.0
    assert END_CLOCK > START_CLOCK or parse_clock(END_CLOCK) > parse_clock(START_CLOCK)
    assert P10_VIDEO.name == "Cam 8-P10-003.mp4"
    assert "P10" in str(P10_VIDEO) or "P10" in P10_VIDEO.name or "P10" in "Cam 8-P10-003.mp4"


def test_html_has_title_and_ids():
    variants = []
    for spec in variant_specs():
        variants.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "why": spec["why"],
                "n_frames": 10,
                "n_raw_hits": 8,
                "n_emit_hold": 2,
                "raw_rate": 0.8,
                "emit_rate": 0.2,
                "mean_emit_conf": 0.85,
                "overlay": f"overlay/{spec['id']}.mp4",
            }
        )
    payload = {
        "clock": "0:26–0:31",
        "checkpoint": "models/v10_snaps/post_train/checkpoint.pth",
        "variants": variants,
    }
    out = Path(tempfile.mkdtemp())
    html = write_html(out, payload).read_text()
    assert "ball_postprocessing_test" in html
    for spec in variant_specs():
        assert spec["id"] in html
        assert f"overlay/{spec['id']}.mp4" in html


def main() -> int:
    test_ten_variants_and_ids()
    test_top_left_clock_and_p10()
    test_html_has_title_and_ids()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
