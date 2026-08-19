"""Pitch rectangle bounds helpers (meters, Pitch 1 center-origin)."""
from __future__ import annotations

import json
from pathlib import Path

_PITCH1 = Path(__file__).resolve().parents[2] / "docs/product/PITCH1_DIMENSIONS.json"
_REC = json.loads(_PITCH1.read_text(encoding="utf-8"))
PITCH_LENGTH_M = float(_REC["length_m"])
PITCH_WIDTH_M = float(_REC["width_m"])


def in_pitch_bounds(
    x: float,
    y: float,
    margin_m: float = 0.0,
    pitch_length: float = PITCH_LENGTH_M,
    pitch_width: float = PITCH_WIDTH_M,
) -> bool:
    """True if (x, y) is inside the pitch rectangle expanded by margin_m."""
    half_l = pitch_length / 2.0 + margin_m
    half_w = pitch_width / 2.0 + margin_m
    return -half_l <= x <= half_l and -half_w <= y <= half_w
