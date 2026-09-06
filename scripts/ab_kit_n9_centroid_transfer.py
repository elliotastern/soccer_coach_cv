#!/usr/bin/env python3
"""N9 A/B: Match3→Match4 HSV/hist affine centroid transfer vs T1 online fit.

Source: Match3 full-cam det cache → annulus feats → k=2 fit.
Dest: Match4 theory obs annulus bank → affine map centroids → freeze + sticky.

Gate: score ≥ T1+0.3 AND flip_rate ≤ T1 AND retain drop ≤ 0.02.
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

from scripts.ab_jersey_theories5 import (  # noqa: E402
    OBS,
    apply_strategy,
    build_tracks,
    label_feat,
    load_obs,
)
from scripts.ab_jersey_undershirt_ideas10 import extract  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    TEAM_MIN_CROPS,
    fit_match_centroids,
    transfer_match_centroids,
    torso_crop,
)

M3_DET = ROOT / "reports/eval_match3/improve_eng_loop/kit_ref_ab/m3_60f_fullcam_det_plain.json"
M3_FEAT_CACHE = ROOT / "reports/eval_match3/improve_eng_loop/kit_n9_m3_annulus_feats.npz"
OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n9_centroid_transfer_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n9_centroid_transfer_ab.md"
GATE_SCORE_DELTA = 0.3
GATE_RETAIN_DROP = 0.02
FRAME_STRIDE = 2
MAX_M3_FEATS = 800


def _build_m3_annulus_feats() -> list[np.ndarray]:
    if M3_FEAT_CACHE.is_file():
        z = np.load(M3_FEAT_CACHE, allow_pickle=True)
        return [np.asarray(f, np.float32) for f in z["feats"]]
    print(f"building M3 annulus feats from {M3_DET.name}…", flush=True)
    raw = json.loads(M3_DET.read_text())
    frames = list(raw["frames"])[::FRAME_STRIDE]
    feats: list[np.ndarray] = []
    for fr in frames:
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
                f = extract(crop, "annulus30")
                if f is None:
                    continue
                feats.append(np.asarray(f, np.float32))
                if len(feats) >= MAX_M3_FEATS:
                    break
            if len(feats) >= MAX_M3_FEATS:
                break
        if len(feats) >= MAX_M3_FEATS:
            break
    M3_FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(M3_FEAT_CACHE, feats=np.stack(feats, axis=0))
    print(f"cached {len(feats)} M3 feats → {M3_FEAT_CACHE.name}", flush=True)
    return feats


def _score_frozen(
    tracks: list[list[dict]],
    cents: np.ndarray,
    radius: float,
    name: str,
) -> dict:
    feat_key, stress_key = "fann", "sann"
    flat_rows = [r for tr in tracks for r in tr]
    lengths = [len(tr) for tr in tracks]
    labs_all: list[int] = []
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = flat_rows[k : k + n]
        k += n
        instant = [
            label_feat(r.get(feat_key), cents, radius, dual_soft=True) for r in chunk
        ]
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
        "name": name,
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
    m3_feats = _build_m3_annulus_feats()
    m3_fit = fit_match_centroids(m3_feats, min_crops=min(TEAM_MIN_CROPS, 40))
    if m3_fit is None:
        raise SystemExit("Match3 centroid fit failed")
    m3_cents, m3_rad = m3_fit

    bundle = load_obs(OBS)
    tracks = [t for t in build_tracks(bundle["rows"]) if t]
    m4_feats = [
        r.get("fann") for tr in tracks for r in tr if r.get("fann") is not None
    ]
    t1 = apply_strategy("T1_annulus_sticky", tracks)

    # Naive copy (no lighting adapt)
    naive = _score_frozen(tracks, m3_cents, m3_rad, "N9_naive_copy_m3")

    xfer = transfer_match_centroids(m3_cents, m3_feats, m4_feats, src_radius=m3_rad)
    if xfer is None:
        raise SystemExit("transfer_match_centroids failed")
    x_cents, x_rad = xfer
    n9 = _score_frozen(tracks, x_cents, x_rad, "N9_affine_transfer")

    delta = float(n9["score"] - t1["score"])
    retain_drop = float(t1["retain_stress"] - n9["retain_stress"])
    flips_ok = float(n9["flip_rate"]) <= float(t1["flip_rate"])
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
        "n_m3_feats": len(m3_feats),
        "n_m4_feats": len(m4_feats),
        "m3_radius": round(float(m3_rad), 4),
        "xfer_radius": round(float(x_rad), 4),
        "n_rows": len(bundle["rows"]),
        "n_tracks": len(tracks),
        "T1_annulus_sticky": t1,
        "N9_naive_copy_m3": naive,
        "N9_affine_transfer": n9,
        "score_delta": round(delta, 3),
        "naive_delta_vs_t1": round(float(naive["score"] - t1["score"]), 3),
        "retain_drop": round(retain_drop, 4),
        "flips_ok": flips_ok,
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N9 Match3→Match4 centroid transfer A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: score ≥ T1+{GATE_SCORE_DELTA}, flips ≤ T1, retain drop ≤ {GATE_RETAIN_DROP}.",
        "",
        f"- source: Match3 annulus feats={len(m3_feats)} → dest Match4={len(m4_feats)}",
        f"- score: T1 **{t1['score']:.1f}** → N9 transfer **{n9['score']:.1f}** "
        f"(Δ {delta:+.2f}) · naive copy {naive['score']:.1f}",
        f"- off50: {t1['off50']:.1f} → {n9['off50']:.1f}",
        f"- retain: {t1['retain_stress']:.3f} → {n9['retain_stress']:.3f} "
        f"(drop {retain_drop:+.3f})",
        f"- flips: {t1['flip_rate']:.3f} → {n9['flip_rate']:.3f}",
        f"- share: {t1['pct0']:.1f}/{t1['pct1']:.1f} → "
        f"{n9['pct0']:.1f}/{n9['pct1']:.1f}",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            {
                "worked": worked,
                "score_delta": delta,
                "naive_delta": float(naive["score"] - t1["score"]),
                "retain_drop": retain_drop,
                "flips_ok": flips_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
