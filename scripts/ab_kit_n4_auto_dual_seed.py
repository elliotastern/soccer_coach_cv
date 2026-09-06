#!/usr/bin/env python3
"""N4 A/B: auto dual→white seed bank + freeze vs T1 annulus+online sticky.

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

from scripts.ab_jersey_theories5 import (  # noqa: E402
    OBS,
    apply_strategy,
    build_tracks,
    label_feat,
    load_obs,
)
from src.perception.team_core import (  # noqa: E402
    fit_auto_dual_seed_centroids,
    is_blue_seed_feat,
    is_white_seed_feat,
)

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n4_auto_dual_seed_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n4_auto_dual_seed_ab.md"
GATE_SCORE_DELTA = 0.3
GATE_RETAIN_DROP = 0.02


def apply_n4_auto_dual_freeze(tracks: list[list[dict]]) -> dict:
    """Annulus feats → dual/blue seed fit → freeze → dual_soft + sticky."""
    feat_key = "fann"
    stress_key = "sann"
    flat_rows = [r for tr in tracks for r in tr]
    feats = [r.get(feat_key) for r in flat_rows if r.get(feat_key) is not None]
    n_blue = sum(1 for f in feats if is_blue_seed_feat(f))
    n_white = sum(1 for f in feats if is_white_seed_feat(f))
    fit = fit_auto_dual_seed_centroids(feats)
    if fit is None:
        return {
            "name": "N4_auto_dual_freeze",
            "error": "fit_failed",
            "n_blue_seeds": n_blue,
            "n_white_seeds": n_white,
        }
    cents, radius = fit

    lengths = [len(tr) for tr in tracks]
    labs_all: list[int] = []
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = flat_rows[k : k + n]
        k += n
        instant = []
        for r in chunk:
            tid, conf = label_feat(r.get(feat_key), cents, radius, dual_soft=True)
            instant.append((tid, conf, r))
        held = instant[0][0]
        for i, (tid, conf, r) in enumerate(instant):
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
    for r, lab in zip(flat_rows, labs_all):
        if lab < 0:
            continue
        sf = r.get(stress_key)
        if sf is None:
            continue
        stid, _ = label_feat(sf, cents, radius, dual_soft=True)
        if stid < 0:
            continue
        tot += 1
        if stid == lab:
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
    p1 = 100.0 * n1 / max(n, 1)
    off = abs(p0 - 50.0)
    bal = float(min(n0, n1) / max(max(n0, n1), 1))
    cover = n / max(len(labs_all), 1)
    by_fr: dict[int, list[int]] = defaultdict(list)
    for r, lab in zip(flat_rows, labs_all):
        if lab >= 0:
            by_fr[r["fr_i"]].append(lab)
    frame_offs = []
    for labs in by_fr.values():
        if len(labs) < 6:
            continue
        s0 = labs.count(0) / len(labs)
        frame_offs.append(abs(s0 - 0.5))
    mean_frame_off = float(np.mean(frame_offs)) if frame_offs else 0.0
    bad_frames = sum(1 for x in frame_offs if x > 0.15) / max(len(frame_offs), 1)
    score = 100.0 * (
        0.30 * (1.0 - min(off / 20.0, 1.0))
        + 0.20 * (1.0 - min(mean_frame_off / 0.25, 1.0))
        + 0.20 * retain
        + 0.20 * (1.0 - flip_rate)
        + 0.10 * bal
    )
    return {
        "name": "N4_auto_dual_freeze",
        "n_blue_seeds": n_blue,
        "n_white_seeds": n_white,
        "pct0": round(p0, 2),
        "pct1": round(p1, 2),
        "off50": round(off, 2),
        "mean_frame_off": round(mean_frame_off, 4),
        "bad_frame_frac": round(bad_frames, 4),
        "retain_stress": round(retain, 4),
        "flip_rate": round(flip_rate, 4),
        "balance": round(bal, 4),
        "coverage": round(cover, 4),
        "n0": n0,
        "n1": n1,
        "score": round(score, 2),
    }


def main() -> None:
    bundle = load_obs(OBS)
    tracks = [t for t in build_tracks(bundle["rows"]) if t]
    baseline = apply_strategy("T1_annulus_sticky", tracks)
    n4 = apply_n4_auto_dual_freeze(tracks)
    if "error" in n4:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "worked": False,
            "error": n4,
            "T1_annulus_sticky": baseline,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
        OUT_MD.write_text("# N4 auto dual-seed A/B\n\n**Worked: `False`** (fit failed)\n")
        print(json.dumps({"worked": False, "error": n4}, indent=2))
        return
    delta = float(n4["score"] - baseline["score"])
    retain_drop = float(baseline["retain_stress"] - n4["retain_stress"])
    flips_ok = float(n4["flip_rate"]) <= float(baseline["flip_rate"])
    worked = (
        delta >= GATE_SCORE_DELTA
        and retain_drop <= GATE_RETAIN_DROP
        and flips_ok
    )
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "score_delta_min": GATE_SCORE_DELTA,
            "retain_drop_max": GATE_RETAIN_DROP,
            "flip_rate_lte_baseline": True,
        },
        "n_rows": len(bundle["rows"]),
        "n_tracks": len(tracks),
        "T1_annulus_sticky": baseline,
        "N4_auto_dual_freeze": n4,
        "score_delta": round(delta, 3),
        "retain_drop": round(retain_drop, 4),
        "flips_ok": flips_ok,
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N4 auto dual-seed freeze A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ T1+{GATE_SCORE_DELTA}, flips ≤ T1, retain drop ≤ {GATE_RETAIN_DROP}.",
        "",
        f"- seeds: blue={n4['n_blue_seeds']} white/dual={n4['n_white_seeds']}",
        f"- score: T1 **{baseline['score']:.1f}** → N4 **{n4['score']:.1f}** (Δ {delta:+.2f})",
        f"- off50: {baseline['off50']:.1f} → {n4['off50']:.1f}",
        f"- retain: {baseline['retain_stress']:.3f} → {n4['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {baseline['flip_rate']:.3f} → {n4['flip_rate']:.3f}",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{n4['pct0']:.1f}/{n4['pct1']:.1f}",
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
