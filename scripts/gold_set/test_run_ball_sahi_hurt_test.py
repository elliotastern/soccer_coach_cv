#!/usr/bin/env python3
"""Unit checks for ball_sahi_hurt_test (no model)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from run_ball_postprocessing_test import write_html  # noqa: E402
from run_ball_sahi_hurt_test import variant_specs  # noqa: E402


def test_ten_hurt_variants():
    specs = variant_specs()
    assert len(specs) == 10
    ids = [s["id"] for s in specs]
    assert ids[0] == "sahi_recover_always"
    assert "sahi_always_tta" in ids
    assert "sahi_dense_tiles" in ids
    for spec in specs:
        assert spec["cfg"].use_sahi is True


def test_html_title():
    variants = [
        {
            "id": s["id"],
            "title": s["title"],
            "why": s["why"],
            "n_frames": 10,
            "n_raw_hits": 9,
            "n_emit_hold": 4,
            "raw_rate": 0.9,
            "emit_rate": 0.4,
            "mean_emit_conf": 0.7,
            "overlay": f"overlay/{s['id']}.mp4",
        }
        for s in variant_specs()
    ]
    out = Path(tempfile.mkdtemp())
    html = write_html(
        out,
        {
            "title": "ball_sahi_hurt_test",
            "clock": "0:26–0:31",
            "checkpoint": "models/v10_snaps/post_train/checkpoint.pth",
            "page_note": "hurt gallery",
            "variants": variants,
        },
    ).read_text()
    assert "ball_sahi_hurt_test" in html
    assert "sahi_recover_always" in html


def main() -> int:
    test_ten_hurt_variants()
    test_html_title()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
