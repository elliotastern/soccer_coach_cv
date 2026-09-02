#!/usr/bin/env python3
"""Unit tests for H-consistency helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from report_match3_h_consistency import (  # noqa: E402
    landmark_roundtrip_m,
    pairwise_spans_m,
)


def test_landmark_roundtrip_identity() -> None:
    pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    rec = {
        "H": np.eye(3).tolist(),
        "image_points": pts,
        "pitch_points": pts,
        "landmark_names": ["a", "b", "c", "d"],
    }
    rt = landmark_roundtrip_m(rec)
    assert rt is not None
    assert rt["rt_max_m"] < 1e-6
    assert rt["rt_pass_l1"] is True


def test_pairwise_spans() -> None:
    spans = pairwise_spans_m([(0.0, 0.0), (3.0, 4.0)])
    assert len(spans) == 1
    assert abs(spans[0] - 5.0) < 1e-6


if __name__ == "__main__":
    test_landmark_roundtrip_identity()
    test_pairwise_spans()
    print("ok")
