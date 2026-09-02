#!/usr/bin/env python3
"""Unit tests for ball_ukf pitch-space filter."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.ball_ukf import BallPitchUKF  # noqa: E402


def test_ukf_update_and_coast() -> None:
    ukf = BallPitchUKF()
    m1 = {"xy": (0.0, 0.0), "conf": 0.95}
    out1 = ukf.step(m1)
    if out1 is None:
        raise AssertionError("first update failed")
    out2 = ukf.step(None)
    if out2 is None or not out2.get("coast"):
        raise AssertionError(f"coast failed {out2}")
    out3 = ukf.step({"xy": (1.0, 0.0), "conf": 0.95})
    if out3 is None or out3.get("coast"):
        raise AssertionError(f"second update failed {out3}")


def main() -> int:
    test_ukf_update_and_coast()
    print("ball_ukf ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
