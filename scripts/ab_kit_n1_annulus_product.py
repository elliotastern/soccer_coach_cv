#!/usr/bin/env python3
"""N1 product A/B: annulus jersey_feature vs center sticky on Match4 theory obs.

Gate: score >= baseline_center_sticky + 0.5 AND retain drop <= 0.02.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.perception.team_core as team_core  # noqa: E402
from scripts.ab_jersey_theories5 import (  # noqa: E402
    OBS,
    apply_strategy,
    build_tracks,
    load_obs,
)
from scripts.ab_jersey_undershirt_ideas10 import extract  # noqa: E402

CROPS = ROOT / "reports/eval_match3/improve_eng_loop/kit_undershirt_ab/m4_torso_crops.npz"
OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n1_annulus_product_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n1_annulus_product_ab.md"
GATE_SCORE_DELTA = 0.5
GATE_RETAIN_DROP = 0.02


def parity_check(n: int = 80) -> dict:
    """Product jersey_feature (annulus on) ≈ ideas10 extract(annulus30)."""
    z = np.load(CROPS, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("c")][:n]
    matched = compared = 0
    max_abs = 0.0
    for k in keys:
        crop = z[k]
        ab = extract(crop, "annulus30")
        prod = team_core.jersey_feature(crop)
        if ab is None and prod is None:
            continue
        compared += 1
        if ab is None or prod is None:
            continue
        d = float(np.max(np.abs(ab[:2] - prod[:2])))
        max_abs = max(max_abs, d)
        if d <= 1e-3:
            matched += 1
    return {
        "n_compared": compared,
        "n_matched_bw": matched,
        "max_abs_bw": round(max_abs, 6),
        "pass": compared > 0 and matched / max(compared, 1) >= 0.95 and max_abs <= 1e-3,
    }


def main() -> None:
    assert team_core.USE_JERSEY_ANNULUS is True
    parity = parity_check()
    print("parity", json.dumps(parity), flush=True)

    bundle = load_obs(OBS)
    tracks = [t for t in build_tracks(bundle["rows"]) if t]
    baseline = apply_strategy("baseline_center_sticky", tracks)
    annulus = apply_strategy("T1_annulus_sticky", tracks)
    delta = float(annulus["score"] - baseline["score"])
    retain_drop = float(baseline["retain_stress"] - annulus["retain_stress"])
    worked = delta >= GATE_SCORE_DELTA and retain_drop <= GATE_RETAIN_DROP and parity["pass"]
    payload = {
        "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "gate": {
            "score_delta_min": GATE_SCORE_DELTA,
            "retain_drop_max": GATE_RETAIN_DROP,
        },
        "product_flags": {
            "USE_JERSEY_ANNULUS": team_core.USE_JERSEY_ANNULUS,
            "JERSEY_ANNULUS_OUTER": team_core.JERSEY_ANNULUS_OUTER,
            "USE_JERSEY_CENTER": team_core.USE_JERSEY_CENTER,
        },
        "parity_vs_ideas10_annulus30": parity,
        "n_rows": len(bundle["rows"]),
        "n_tracks": len(tracks),
        "baseline_center_sticky": baseline,
        "T1_annulus_sticky": annulus,
        "score_delta": round(delta, 3),
        "retain_drop": round(retain_drop, 4),
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N1 product annulus A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ baseline+{GATE_SCORE_DELTA} and retain drop ≤ {GATE_RETAIN_DROP}; "
        "parity vs ideas10 annulus30.",
        "",
        f"- score: baseline **{baseline['score']:.1f}** → annulus **{annulus['score']:.1f}** "
        f"(Δ {delta:+.2f})",
        f"- off50: {baseline['off50']:.1f} → {annulus['off50']:.1f}",
        f"- retain: {baseline['retain_stress']:.3f} → {annulus['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {baseline['flip_rate']:.3f} → {annulus['flip_rate']:.3f}",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{annulus['pct0']:.1f}/{annulus['pct1']:.1f}",
        f"- parity pass: {parity['pass']} (matched {parity['n_matched_bw']}/{parity['n_compared']}, "
        f"max|Δbw|={parity['max_abs_bw']})",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({"worked": worked, "score_delta": delta, "retain_drop": retain_drop}, indent=2))


if __name__ == "__main__":
    main()
