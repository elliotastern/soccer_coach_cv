#!/usr/bin/env python3
"""4K association radius must scale; 80px was dropping ByteTrack matches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.tracker import Tracker


def test_assoc_px_scales_4k():
    tr = Tracker()
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    assert tr.assoc_px(frame_4k) >= 150.0
    frame_hd = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert abs(tr.assoc_px(frame_hd) - 80.0) < 1e-6


def test_explicit_match_px():
    tr = Tracker(match_px=80.0)
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    assert tr.assoc_px(frame_4k) == 80.0


def main() -> int:
    test_assoc_px_scales_4k()
    test_explicit_match_px()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
