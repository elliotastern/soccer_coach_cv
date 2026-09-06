#!/usr/bin/env python3
"""N8 A/B: birth wait (gray until age≥3 or ≥2 cams agree) vs T1 sticky.

Unlike T5 hard age-lock: once committed, normal sticky flips only (conf≥0.90).
Gate: score ≥ T1+0.3 AND flip_rate ≤ T1 AND retain drop ≤ 0.02.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_jersey_theories5 import (  # noqa: E402
    OBS,
    apply_strategy,
    build_tracks,
    fit_feats,
    label_feat,
    load_obs,
)

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n8_birth_wait_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n8_birth_wait_ab.md"
GATE_SCORE_DELTA = 0.3
GATE_RETAIN_DROP = 0.02
BIRTH_WAIT_MIN_AGE = 3
BIRTH_WAIT_MIN_CAMS = 2


def _score_n8(tracks: list[list[dict]], *, birth_wait: bool) -> dict:
    feat_key, stress_key = "fann", "sann"
    flat_rows = [r for tr in tracks for r in tr]
    feats = [r.get(feat_key) for r in flat_rows]
    fit = fit_feats(feats)
    if fit is None:
        return {"error": "fit_failed"}
    cents, radius = fit

    instant = []
    frame_cam_team: dict[tuple[int, str], list[int]] = defaultdict(list)
    for r in flat_rows:
        tid, conf = label_feat(r.get(feat_key), cents, radius, dual_soft=True)
        instant.append((tid, conf, r))
        if tid >= 0:
            frame_cam_team[(r["fr_i"], r["cam"])].append(tid)
    frame_cam_maj = {
        k: Counter(ts).most_common(1)[0][0] for k, ts in frame_cam_team.items()
    }

    lengths = [len(tr) for tr in tracks]
    labs_all: list[int] = []
    n_waited = n_early_cam = 0
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = instant[k : k + n]
        k += n
        held = -1
        for i, (tid, conf, r) in enumerate(chunk):
            age = i
            # multi-cam agree count for this candidate (frame-level maj proxy)
            fr_i = r["fr_i"]
            majs = [
                frame_cam_maj[(fr_i, cam)]
                for cam in {x["cam"] for x in flat_rows if x["fr_i"] == fr_i}
                if (fr_i, cam) in frame_cam_maj
            ]
            agree = sum(1 for m in majs if tid >= 0 and m == tid)

            if birth_wait and held < 0:
                if age < BIRTH_WAIT_MIN_AGE and agree < BIRTH_WAIT_MIN_CAMS:
                    tid = -1
                    n_waited += 1
                elif agree >= BIRTH_WAIT_MIN_CAMS and age < BIRTH_WAIT_MIN_AGE:
                    n_early_cam += 1
                    held = tid
                else:
                    held = tid
            else:
                # normal sticky once committed (no T5 hard age lock)
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
    name = "N8_birth_wait" if birth_wait else "baseline_no_birth_wait"
    return {
        "name": name,
        "birth_wait": birth_wait,
        "n_waited": n_waited,
        "n_early_cam_commit": n_early_cam,
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
    t1 = apply_strategy("T1_annulus_sticky", tracks)
    base = _score_n8(tracks, birth_wait=False)
    n8 = _score_n8(tracks, birth_wait=True)
    delta = float(n8["score"] - t1["score"])
    retain_drop = float(t1["retain_stress"] - n8["retain_stress"])
    flips_ok = float(n8["flip_rate"]) <= float(t1["flip_rate"])
    worked = (
        delta >= GATE_SCORE_DELTA
        and flips_ok
        and retain_drop <= GATE_RETAIN_DROP
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
        "baseline_no_birth_wait": base,
        "N8_birth_wait": n8,
        "T1_annulus_sticky": t1,
        "score_delta": round(delta, 3),
        "retain_drop": round(retain_drop, 4),
        "flips_ok": flips_ok,
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N8 birth-wait A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ T1+{GATE_SCORE_DELTA}, flips ≤ T1, retain drop ≤ {GATE_RETAIN_DROP}.",
        "",
        f"- gray waits: {n8['n_waited']} · early cam commits: {n8['n_early_cam_commit']}",
        f"- score: T1 **{t1['score']:.1f}** → N8 **{n8['score']:.1f}** (Δ {delta:+.2f})",
        f"- off50: {t1['off50']:.1f} → {n8['off50']:.1f}",
        f"- retain: {t1['retain_stress']:.3f} → {n8['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {t1['flip_rate']:.3f} → {n8['flip_rate']:.3f}",
        f"- coverage: {base['coverage']:.3f} → {n8['coverage']:.3f}",
        f"- share: {t1['pct0']:.1f}/{t1['pct1']:.1f} → "
        f"{n8['pct0']:.1f}/{n8['pct1']:.1f}",
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
                "coverage": n8["coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
