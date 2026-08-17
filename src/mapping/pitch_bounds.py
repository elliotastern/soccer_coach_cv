"""Pitch rectangle bounds helpers (meters, FIFA center-origin)."""
from __future__ import annotations


def in_pitch_bounds(
    x: float,
    y: float,
    margin_m: float = 0.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> bool:
    """True if (x, y) is inside the pitch rectangle expanded by margin_m."""
    half_l = pitch_length / 2.0 + margin_m
    half_w = pitch_width / 2.0 + margin_m
    return -half_l <= x <= half_l and -half_w <= y <= half_w
