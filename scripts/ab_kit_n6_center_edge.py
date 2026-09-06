#!/usr/bin/env python3
"""N6 A/B: center-vs-edge veto (edge much bluer → core feats) vs plain annulus.

Primary gate on Match4 theory obs: rebuild feats via product jersey_feature
with veto on/off using cached det frames (stride), same sticky as T1.

Fallback if cache missing: torso-crop bank sticky A/B.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
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
from scripts.ab_jersey_undershirt_variants import paint_opposite_ring  # noqa: E402
from src.perception.team_core import torso_crop  # noqa: E402

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n6_center_edge_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n6_center_edge_ab.md"
CACHE = ROOT / "reports/eval_match3/team_id_strategy_grid/m4_90s_det_plain.json"
GATE_SCORE_DELTA = 0.3
GATE_RETAIN_DROP = 0.02
RING = 0.24
MAX_BUILD_FRAMES = 60
FRAME_STRIDE = 3


def _score_tracks(tracks, feat_key: str, stress_key: str) -> dict:
    flat_rows = [r for tr in tracks for r in tr]
    feats = [r.get(feat_key) for r in flat_rows]
    fit = fit_feats(feats)
    if fit is None:
        return {"error": "fit_failed"}
    cents, radius = fit
    lengths = [len(tr) for tr in tracks]
    labs_all: list[int] = []
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = flat_rows[k : k + n]
        k += n
        instant = [label_feat(r.get(feat_key), cents, radius, dual_soft=True) for r in chunk]
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
    off = abs(p0 - 50.0)
    bal = float(min(n0, n1) / max(max(n0, n1), 1))
    by_fr: dict[int, list[int]] = defaultdict(list)
    for r, lab in zip(flat_rows, labs_all):
        if lab >= 0:
            by_fr[r["fr_i"]].append(lab)
    frame_offs = []
    for labs in by_fr.values():
        if len(labs) < 6:
            continue
        frame_offs.append(abs(labs.count(0) / len(labs) - 0.5))
    mean_frame_off = float(np.mean(frame_offs)) if frame_offs else 0.0
    score = 100.0 * (
        0.30 * (1.0 - min(off / 20.0, 1.0))
        + 0.20 * (1.0 - min(mean_frame_off / 0.25, 1.0))
        + 0.20 * retain
        + 0.20 * (1.0 - flip_rate)
        + 0.10 * bal
    )
    return {
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


def _feat_pair(crop: np.ndarray, veto: bool):
    team_core.USE_CENTER_EDGE_VETO = veto
    team_core.USE_JERSEY_ANNULUS = True
    f = team_core.jersey_feature(crop)
    stressed = paint_opposite_ring(crop, RING)
    s = team_core.jersey_feature(stressed)
    return f, s


def build_rows_from_cache() -> list[dict] | None:
    if not CACHE.is_file():
        return None
    raw = json.loads(CACHE.read_text())
    frames = list(raw["frames"])[::FRAME_STRIDE][:MAX_BUILD_FRAMES]
    rows = []
    for i, fr in enumerate(frames):
        row = raw["data"][str(fr)]
        for cam in raw["cams"]:
            hex_bgr = (row.get("bgr") or {}).get(cam)
            if not hex_bgr:
                continue
            img = cv2.imdecode(
                np.frombuffer(bytes.fromhex(hex_bgr), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                continue
            wh = (img.shape[1], img.shape[0])
            for d in row["dets"].get(cam) or []:
                if str(d.get("class_name", "player")).lower() == "ball":
                    continue
                crop = torso_crop(img, d["bbox"], cam=cam, frame_wh=wh)
                if crop is None:
                    continue
                f0, s0 = _feat_pair(crop, veto=False)
                f1, s1 = _feat_pair(crop, veto=True)
                if f0 is None and f1 is None:
                    continue
                rows.append(
                    {
                        "cam": cam,
                        "fr": int(fr),
                        "fr_i": i,
                        "bbox": [float(v) for v in d["bbox"]],
                        "f_base": f0,
                        "s_base": s0,
                        "f_n6": f1,
                        "s_n6": s1,
                    }
                )
        if i % 10 == 0:
            print(f"  build fr={fr} rows={len(rows)}", flush=True)
    return rows


def main() -> None:
    print("building N6 feature rows from det cache…", flush=True)
    rows = build_rows_from_cache()
    if not rows:
        print("cache build failed", flush=True)
        OUT_JSON.write_text(json.dumps({"worked": False, "error": "no_rows"}) + "\n")
        return
    # Restore product defaults after build
    team_core.USE_CENTER_EDGE_VETO = True
    team_core.USE_JERSEY_ANNULUS = True

    # Map into theory-style tracks using rebuilt rows only
    tracks = build_tracks(rows)
    tracks = [t for t in tracks if t]
    # Remap keys for scorer
    for tr in tracks:
        for r in tr:
            r["fann"] = r.get("f_base")
            r["sann"] = r.get("s_base")
    baseline = _score_tracks(tracks, "fann", "sann")
    for tr in tracks:
        for r in tr:
            r["fann"] = r.get("f_n6")
            r["sann"] = r.get("s_n6")
    n6 = _score_tracks(tracks, "fann", "sann")
    # Reference T1 on full theory obs (unchanged)
    bundle = load_obs(OBS)
    t1 = apply_strategy("T1_annulus_sticky", [t for t in build_tracks(bundle["rows"]) if t])

    delta = float(n6["score"] - baseline["score"])
    retain_drop = float(baseline["retain_stress"] - n6["retain_stress"])
    flips_ok = float(n6["flip_rate"]) <= float(baseline["flip_rate"])
    worked = delta >= GATE_SCORE_DELTA and retain_drop <= GATE_RETAIN_DROP and flips_ok
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "harness": "m4_90s_det_plain rebuild sticky",
        "n_rows": len(rows),
        "n_tracks": len(tracks),
        "gate": {
            "score_delta_min": GATE_SCORE_DELTA,
            "retain_drop_max": GATE_RETAIN_DROP,
            "flip_rate_lte_baseline": True,
        },
        "baseline_annulus_sticky": baseline,
        "N6_center_edge_veto": n6,
        "T1_theory_obs_ref": t1,
        "score_delta": round(delta, 3),
        "retain_drop": round(retain_drop, 4),
        "flips_ok": flips_ok,
        "worked": worked,
        "n_veto_diff": sum(
            1
            for r in rows
            if r.get("f_base") is not None
            and r.get("f_n6") is not None
            and float(np.max(np.abs(r["f_base"][:2] - r["f_n6"][:2]))) > 1e-4
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N6 center-vs-edge veto A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ annulus+sticky+{GATE_SCORE_DELTA}, flips ≤ baseline, retain drop ≤ {GATE_RETAIN_DROP}.",
        "",
        f"- rows={len(rows)} tracks={len(tracks)} veto_changed_feats={payload['n_veto_diff']}",
        f"- score: annulus **{baseline['score']:.1f}** → N6 **{n6['score']:.1f}** (Δ {delta:+.2f})",
        f"- off50: {baseline['off50']:.1f} → {n6['off50']:.1f}",
        f"- retain: {baseline['retain_stress']:.3f} → {n6['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {baseline['flip_rate']:.3f} → {n6['flip_rate']:.3f}",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{n6['pct0']:.1f}/{n6['pct1']:.1f}",
        f"- theory T1 ref score: {t1['score']:.1f}",
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
                "n_veto_diff": payload["n_veto_diff"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
