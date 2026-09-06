#!/usr/bin/env python3
"""N7 A/B: gated dual→white (white≥blue) vs ungated dual_soft on theory obs.

Gate: score ≥ T1+0.3 AND retain drop ≤ 0.02 AND flip_rate ≤ T1.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.perception.team_core as team_core  # noqa: E402
from scripts.ab_jersey_theories5 import (  # noqa: E402
    OBS,
    apply_strategy,
    build_tracks,
    fit_feats,
    label_feat,
    load_obs,
)

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n7_gated_dual_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n7_gated_dual_ab.md"
GATE_SCORE_DELTA = 0.3
GATE_RETAIN_DROP = 0.02


def _score(tracks, *, gated: bool) -> dict:
    """Sticky annulus score; dual_soft ungated vs gated via flag + label_feat."""
    team_core.USE_GATED_DUAL_TO_WHITE = gated
    team_core.USE_DUAL_TO_WHITE = True
    feat_key, stress_key = "fann", "sann"
    flat_rows = [r for tr in tracks for r in tr]
    feats = [r.get(feat_key) for r in flat_rows]
    fit = fit_feats(feats)
    if fit is None:
        return {"error": "fit_failed"}
    cents, radius = fit
    n_dual = n_gate = 0
    for f in feats:
        if f is None:
            continue
        b, w = float(f[0]), float(f[1])
        if b >= team_core.DUAL_COLOR_FRAC and w >= team_core.DUAL_COLOR_FRAC:
            n_dual += 1
            if w >= b:
                n_gate += 1

    def lab(f):
        if f is None:
            return -1, 0.0
        if gated:
            tid, conf = label_feat(f, cents, radius, dual_soft=False)
            b, w = float(f[0]), float(f[1])
            if (
                b >= team_core.DUAL_COLOR_FRAC
                and w >= team_core.DUAL_COLOR_FRAC
                and w >= b
            ):
                return 1, 0.70
            return int(tid), float(conf)
        return label_feat(f, cents, radius, dual_soft=True)

    lengths = [len(tr) for tr in tracks]
    labs_all: list[int] = []
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = flat_rows[k : k + n]
        k += n
        instant = [lab(r.get(feat_key)) for r in chunk]
        held = instant[0][0]
        for i, (tid, conf) in enumerate(instant):
            if i == 0:
                held = tid
            else:
                if held >= 0 and tid != held and conf < 0.90:
                    tid = held
                elif held < 0:
                    held = tid
                else:
                    held = tid if conf >= 0.90 else held
                    tid = held
            labs_all.append(tid)

    same = tot = 0
    for r, labi in zip(flat_rows, labs_all):
        if labi < 0:
            continue
        sf = r.get(stress_key)
        if sf is None:
            continue
        stid, _ = lab(sf)
        if stid < 0:
            continue
        tot += 1
        if stid == labi:
            same += 1
    retain = same / max(tot, 1)
    flips = edges = 0
    k = 0
    for L in lengths:
        tl = labs_all[k : k + L]
        k += L
        for a, b in zip(tl, tl[1:]):
            if a < 0 or b < 0:
                continue
            edges += 1
            if a != b:
                flips += 1
    flip_rate = flips / max(edges, 1)
    n0 = sum(1 for t in labs_all if t == 0)
    n1 = sum(1 for t in labs_all if t == 1)
    n = n0 + n1
    p0 = 100.0 * n0 / max(n, 1)
    off = abs(p0 - 50.0)
    bal = float(min(n0, n1) / max(max(n0, n1), 1))
    by_fr: dict[int, list[int]] = defaultdict(list)
    for r, labi in zip(flat_rows, labs_all):
        if labi >= 0:
            by_fr[r["fr_i"]].append(labi)
    frame_offs = [
        abs(labs.count(0) / len(labs) - 0.5)
        for labs in by_fr.values()
        if len(labs) >= 6
    ]
    mean_frame_off = float(np.mean(frame_offs)) if frame_offs else 0.0
    score = 100.0 * (
        0.30 * (1.0 - min(off / 20.0, 1.0))
        + 0.20 * (1.0 - min(mean_frame_off / 0.25, 1.0))
        + 0.20 * retain
        + 0.20 * (1.0 - flip_rate)
        + 0.10 * bal
    )
    return {
        "gated": gated,
        "n_dual": n_dual,
        "n_dual_white_ge_blue": n_gate,
        "pct0": round(p0, 2),
        "pct1": round(100.0 - p0, 2) if n else 0.0,
        "off50": round(off, 2),
        "mean_frame_off": round(mean_frame_off, 4),
        "retain_stress": round(retain, 4),
        "flip_rate": round(flip_rate, 4),
        "balance": round(bal, 4),
        "coverage": round(n / max(len(labs_all), 1), 4),
        "n0": n0,
        "n1": n1,
        "score": round(score, 2),
    }


def main() -> None:
    bundle = load_obs(OBS)
    tracks = [t for t in build_tracks(bundle["rows"]) if t]
    # Fair baseline: ungated dual (product pre-N7)
    baseline = _score(tracks, gated=False)
    n7 = _score(tracks, gated=True)
    # Restore product default after A/B (will set False if fail below)
    team_core.USE_GATED_DUAL_TO_WHITE = False
    t1_ref = apply_strategy("T1_annulus_sticky", tracks)
    delta = float(n7["score"] - baseline["score"])
    retain_drop = float(baseline["retain_stress"] - n7["retain_stress"])
    flips_ok = float(n7["flip_rate"]) <= float(baseline["flip_rate"])
    worked = delta >= GATE_SCORE_DELTA and retain_drop <= GATE_RETAIN_DROP and flips_ok
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "score_delta_min": GATE_SCORE_DELTA,
            "retain_drop_max": GATE_RETAIN_DROP,
            "flip_rate_lte_baseline": True,
        },
        "n_rows": len(bundle["rows"]),
        "n_tracks": len(tracks),
        "baseline_ungated_dual": baseline,
        "N7_gated_dual": n7,
        "T1_apply_strategy_ref": t1_ref,
        "score_delta": round(delta, 3),
        "retain_drop": round(retain_drop, 4),
        "flips_ok": flips_ok,
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N7 gated dual→white A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ ungated+{GATE_SCORE_DELTA}, flips ≤ baseline, retain drop ≤ {GATE_RETAIN_DROP}.",
        "",
        f"- dual crops: {n7['n_dual']} (white≥blue: {n7['n_dual_white_ge_blue']})",
        f"- score: ungated **{baseline['score']:.1f}** → N7 **{n7['score']:.1f}** (Δ {delta:+.2f})",
        f"- off50: {baseline['off50']:.1f} → {n7['off50']:.1f}",
        f"- retain: {baseline['retain_stress']:.3f} → {n7['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {baseline['flip_rate']:.3f} → {n7['flip_rate']:.3f}",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{n7['pct0']:.1f}/{n7['pct1']:.1f}",
        f"- T1 apply_strategy ref: {t1_ref['score']:.1f}",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            {
                "worked": worked,
                "score_delta": delta,
                "retain_drop": retain_drop,
                "flips_ok": flips_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
