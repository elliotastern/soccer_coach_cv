#!/usr/bin/env python3
"""Smoke: v13 residual builder contracts (clamp, holdout, paths)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from build_ball_finetune_v11 import M3_TRAIN_MAX, clamp_box  # noqa: E402
from build_ball_finetune_v13_residual import (  # noqa: E402
    EMIT_CONF,
    HARD_COPIES,
    M3_HOLDOUT_HI,
    M3_HOLDOUT_LO,
    N_V12_ANCHOR_TRAIN,
    OUT,
    SIDE_TIGHT,
    SOFT_PREF_HI,
    SOFT_PREF_LO,
    TINY_PASTES,
    center_tight,
    conf_band,
    is_residual,
)


def test_clamp_and_tight() -> None:
    ok = clamp_box([-20, 100, 38, 26], 1920, 1080)
    if ok is None or ok[0] != 0:
        raise AssertionError(ok)
    tight = center_tight([100, 200, 40, 20], SIDE_TIGHT)
    if abs(tight[2] - SIDE_TIGHT) > 1e-6 or abs(tight[3] - SIDE_TIGHT) > 1e-6:
        raise AssertionError(tight)
    cx = 100 + 20
    cy = 200 + 10
    if abs(tight[0] + SIDE_TIGHT / 2 - cx) > 1e-6:
        raise AssertionError(("cx", tight, cx))
    if abs(tight[1] + SIDE_TIGHT / 2 - cy) > 1e-6:
        raise AssertionError(("cy", tight, cy))


def test_holdout_and_policy() -> None:
    if M3_TRAIN_MAX != 119:
        raise AssertionError(M3_TRAIN_MAX)
    if M3_HOLDOUT_LO != 120 or M3_HOLDOUT_HI != 194:
        raise AssertionError((M3_HOLDOUT_LO, M3_HOLDOUT_HI))
    if EMIT_CONF != 0.80 or SOFT_PREF_LO != 0.50 or SOFT_PREF_HI != 0.79:
        raise AssertionError("soft band")
    if HARD_COPIES != 2 or TINY_PASTES != 2:
        raise AssertionError("lighter than v12")
    if N_V12_ANCHOR_TRAIN < 50:
        raise AssertionError("need modest v12 anchor")
    if not str(OUT).endswith("ball_finetune_v13_residual"):
        raise AssertionError(OUT)
    if conf_band(0.61) != "050_079" or conf_band(None) != "no_det":
        raise AssertionError("conf_band")
    if not is_residual(0.61, True, "P10"):
        raise AssertionError("soft residual")
    if is_residual(0.85, True, "P10"):
        raise AssertionError("emit should drop")
    if not is_residual(None, False, "P8"):
        raise AssertionError("P8 no_det extra")
    if is_residual(None, False, "P10"):
        raise AssertionError("P10 non-clear no_det should skip")


def main() -> int:
    test_clamp_and_tight()
    test_holdout_and_policy()
    print("ok v13 residual contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
