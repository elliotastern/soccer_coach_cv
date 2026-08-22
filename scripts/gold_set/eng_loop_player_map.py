#!/usr/bin/env python3
"""Eng-loop: player MAP quality (landmarks/H/support) — not fuse knobs.

Gate: every component ≥ 9.0 / 10.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import (  # noqa: E402
    MIN_SUPPORT,
    PLAYER_MIN_SUPPORT,
    diagnose_map_foot,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_map"
GATE = 9.0
CAMS = ["P10", "P9", "P7", "P8"]


def _score_bool(ok: bool, partial: float = 4.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    # 01 calib present
    missing = [c for c in CAMS if load_calib(c) is None]
    scores["01_calib_present"] = _score_bool(not missing, 2.0)
    notes["01_calib_present"] = f"missing={missing}"

    # 02 ball MIN_SUPPORT locked
    src = (ROOT / "src/mapping/match3_xy.py").read_text(encoding="utf-8")
    locked = "MIN_SUPPORT = 0.20" in src and MIN_SUPPORT == 0.20
    scores["02_ball_minsupport_locked"] = _score_bool(locked, 0.0)
    notes["02_ball_minsupport_locked"] = f"MIN_SUPPORT={MIN_SUPPORT}"

    # 03 player softer than ball
    scores["03_player_softer_hull"] = _score_bool(
        PLAYER_MIN_SUPPORT < MIN_SUPPORT and PLAYER_MIN_SUPPORT >= 0.10 - 1e-9, 3.0
    )
    notes["03_player_softer_hull"] = f"PLAYER_MIN_SUPPORT={PLAYER_MIN_SUPPORT}"

    # 04 P7/P8 hull present after H_p (P7 must expand; P8 already had)
    p7 = json.loads((ROOT / "reports/eval_match3/match3_pitch_calib/P7_manual.json").read_text())
    p8 = json.loads((ROOT / "reports/eval_match3/match3_pitch_calib/P8_manual.json").read_text())
    p7_hull = p7.get("hull_image_points") or []
    p8_hull = p8.get("hull_image_points") or []
    hull_ok = len(p7_hull) > len(p7.get("image_points") or []) and len(p8_hull) >= len(
        p8.get("image_points") or []
    )
    scores["04_hp_hull_expanded"] = _score_bool(hull_ok, 4.0)
    notes["04_hp_hull_expanded"] = f"P7_hull={len(p7_hull)} P8_hull={len(p8_hull)}"

    # 05 L1: full refit must not be live if it broke ball — expect HEAD landmarks or documented revert
    l1_path = OUT / "l1_p8_refit.json"
    l1 = json.loads(l1_path.read_text()) if l1_path.is_file() else {}
    l1_ok = l1.get("status") in {
        "reverted_full_refit_ball_kill_switch",
        "promoted_H_player_dual",
        "promoted_H_player_bottom_anchor",
        "rejected_l2",
        "nudged",
        "kept_HEAD",
    } or (
        float(p8.get("roundtrip_max_m") or 99) <= 0.15
        and l1.get("status") == "promoted"
    )
    scores["05_l1_killswitch_honored"] = _score_bool(l1_ok, 3.0)
    notes["05_l1_killswitch_honored"] = f"l1_status={l1.get('status')} names={p8.get('landmark_names')}"

    # 06 ball kill switch: P8 fr3022-style ball foot still maps
    calib8 = load_calib("P8")
    ball = map_ball_box(
        calib8,
        [1389.6, 943.8, 30.4, 28.8],
        0.84,
        frame_wh=(1920, 1080),
        apply_undistort=False,
    )
    ball_ok = ball is not None and abs(ball["xy"][0] - 25.0) < 6.0
    scores["06_ball_map_killswitch"] = _score_bool(ball_ok, 0.0)
    notes["06_ball_map_killswitch"] = f"ball={None if ball is None else ball['xy']}"

    # 07–10 pack metrics
    d1_path = OUT / "d1_pack_reasons.json"
    d2_path = OUT / "d2_collapse.json"
    d1 = json.loads(d1_path.read_text()) if d1_path.is_file() else {}
    d2 = json.loads(d2_path.read_text()) if d2_path.is_file() else {}
    summ = d1.get("summary") or {}
    mapped_frac = float(summ.get("mapped_frac") or 0.0)
    reasons = summ.get("pack_reasons") or {}
    n_ok = int(summ.get("n_ok") or reasons.get("ok") or 0)
    n_fail = int(summ.get("n_map_fail") or 0)
    low = int(reasons.get("low_support") or 0)
    off = int(reasons.get("off_pitch") or 0)

    scores["07_mapped_frac"] = (
        10.0 if mapped_frac >= 0.80 else max(0.0, 10.0 * mapped_frac / 0.80)
    )
    notes["07_mapped_frac"] = f"mapped_frac={mapped_frac} ok={n_ok} fail={n_fail}"

    # low_support should be minority after H_p
    low_rate = low / max(n_ok + n_fail, 1)
    scores["08_low_support_rate"] = (
        10.0 if low_rate <= 0.08 else max(0.0, 10.0 - 50.0 * (low_rate - 0.08))
    )
    notes["08_low_support_rate"] = f"low={low} rate={low_rate:.3f}"

    p8_reasons = (summ.get("per_cam_reasons") or {}).get("P8") or {}
    p8_off = int(p8_reasons.get("off_pitch") or 0)
    p8_ok = int(p8_reasons.get("ok") or 0)
    p8_off_rate = p8_off / max(p8_off + p8_ok, 1)
    p8_hp = bool(p8.get("H_player"))
    off_gate = 0.38 if p8_hp else 0.25
    scores["09_p8_off_pitch"] = (
        10.0 if p8_off_rate <= off_gate else max(0.0, 10.0 - 30.0 * (p8_off_rate - off_gate))
    )
    notes["09_p8_off_pitch"] = (
        f"P8 off={p8_off} ok={p8_ok} rate={p8_off_rate:.3f} gate={off_gate} H_player={p8_hp}"
    )

    n_crush = int(d2.get("n_crush_flags") or 0)
    scores["10_collapse_flags"] = 10.0 if n_crush <= 2 else max(0.0, 10.0 - 2.0 * n_crush)
    notes["10_collapse_flags"] = f"n_crush_flags={n_crush}"

    # 11 defish fingerprint P7–P10
    fish_ok = True
    for cam in CAMS:
        rec = json.loads(
            (ROOT / f"reports/eval_match3/match3_pitch_calib/{cam}_manual.json").read_text()
        )
        if not rec.get("undistort"):
            fish_ok = False
    scores["11_defish_calib"] = _score_bool(fish_ok, 4.0)
    notes["11_defish_calib"] = f"undistort_on_all_quad={fish_ok}"

    # 12 docs present
    plan = ROOT / "docs/product/MATCH3_PLAYER_MAP_IMPROVE_PLAN.md"
    dead = OUT / "DEAD_ENDS.md"
    prompt = OUT / "PROMPT.md"
    docs_ok = plan.is_file() and dead.is_file() and prompt.is_file()
    scores["12_docs_ledger"] = _score_bool(docs_ok, 3.0)
    notes["12_docs_ledger"] = f"plan={plan.is_file()} dead={dead.is_file()} prompt={prompt.is_file()}"

    hard = [
        scores["02_ball_minsupport_locked"],
        scores["06_ball_map_killswitch"],
        scores["07_mapped_frac"],
        scores["08_low_support_rate"],
        scores["10_collapse_flags"],
    ]
    scores["13_product_map_ready"] = float(np.mean(hard))
    notes["13_product_map_ready"] = f"mean_hard={scores['13_product_map_ready']:.1f}"

    d3_path = OUT / "d3_p8_quadrant.json"
    d3 = json.loads(d3_path.read_text()) if d3_path.is_file() else {}
    bottom_frac = float((d3.get("summary") or {}).get("bottom_mapped_frac") or 0.0)
    scores["14_p8_bottom_mapped"] = (
        10.0 if bottom_frac >= 0.5 else max(0.0, 10.0 * bottom_frac / 0.5)
    )
    notes["14_p8_bottom_mapped"] = f"bottom_mapped_frac={bottom_frac}"

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
        "constants": {
            "MIN_SUPPORT": MIN_SUPPORT,
            "PLAYER_MIN_SUPPORT": PLAYER_MIN_SUPPORT,
        },
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
