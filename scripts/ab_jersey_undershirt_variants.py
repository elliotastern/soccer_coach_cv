#!/usr/bin/env python3
"""A/B top undershirt-resistant jersey sampling on Match 4 real torso crops.

Outer kit / undershirt can disagree (e.g. white jersey + blue sleeves). Variants
re-weight torso pixels toward the crop center; stress test paints opposite color
on the outer ring of real crops.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import (  # noqa: E402
    FEAT_BASE,
    HUE_BINS,
    KIT_DIM,
    MIN_CROP_STD,
    MIN_JERSEY_FRAC,
    TEAM_MIN_CROPS,
    _adaptive_non_green,
    assign_feature,
    feature_distance,
    fit_match_centroids,
    jersey_feature,
    torso_crop,
)

CACHE = ROOT / "reports/eval_match3/team_id_strategy_grid/m4_90s_det_plain.json"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_undershirt_ab"
CROPS_NPZ = OUT / "m4_torso_crops.npz"

# name -> kwargs for weighted feature
VARIANTS = [
    ("baseline", {"mode": "baseline"}),
    ("gauss_soft", {"mode": "gauss", "sig": 0.40}),
    ("gauss_tight", {"mode": "gauss", "sig": 0.22}),
    ("hard_center_50", {"mode": "hard", "frac": 0.50}),
    ("hard_center_35", {"mode": "hard", "frac": 0.35}),
    ("raised_cosine", {"mode": "raised"}),
    ("sleeve_x_gauss", {"mode": "gauss_xy", "sig_x": 0.22, "sig_y": 0.55}),
    ("top50_center_w", {"mode": "topk", "sig": 0.35, "keep": 0.50}),
    ("gauss_sat_gate", {"mode": "gauss_sat", "sig": 0.35}),
    ("gauss_hue_trim", {"mode": "gauss_hue", "sig": 0.35, "hue_tol": 25.0}),
    ("inner_ellipse_55", {"mode": "ellipse", "r_max": 0.55}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-crops", action="store_true")
    p.add_argument("--max-crops", type=int, default=600)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--ring", type=float, default=0.24)
    return p.parse_args()


def _xy_norm(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    return xn.astype(np.float32), yn.astype(np.float32)


def _spatial_weights(h: int, w: int, cfg: dict) -> np.ndarray:
    mode = cfg["mode"]
    xn, yn = _xy_norm(h, w)
    r2 = xn * xn + yn * yn
    r = np.sqrt(np.maximum(r2, 0.0))
    if mode == "baseline":
        return np.ones((h, w), dtype=np.float32)
    if mode == "gauss":
        sig = float(cfg["sig"])
        return np.exp(-0.5 * r2 / (sig * sig)).astype(np.float32)
    if mode == "gauss_xy":
        sx, sy = float(cfg["sig_x"]), float(cfg["sig_y"])
        return np.exp(-0.5 * ((xn / sx) ** 2 + (yn / sy) ** 2)).astype(np.float32)
    if mode == "hard":
        f = float(cfg["frac"])
        return ((np.abs(xn) <= f) & (np.abs(yn) <= f)).astype(np.float32)
    if mode == "raised":
        wgt = np.clip(1.0 - r, 0.0, 1.0)
        return (wgt * wgt).astype(np.float32)
    if mode == "ellipse":
        return (r <= float(cfg["r_max"])).astype(np.float32)
    if mode in ("topk", "gauss_sat", "gauss_hue"):
        sig = float(cfg.get("sig", 0.35))
        return np.exp(-0.5 * r2 / (sig * sig)).astype(np.float32)
    raise ValueError(f"unknown mode {mode}")


def _feat_from_keep(
    hsv: np.ndarray,
    keep: np.ndarray,
    wgt: np.ndarray,
) -> np.ndarray | None:
    ww = wgt * keep.astype(np.float32)
    mass = float(ww.sum())
    if mass < 18.0 or float(keep.mean()) < MIN_JERSEY_FRAC:
        return None
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = keep & (h >= 85) & (h <= 145) & (s >= 35)
    purple = keep & (h >= 125) & (h <= 170) & (s >= 30)
    white = keep & (s <= 55) & (v >= 125)
    yellow = keep & (h >= 12) & (h <= 40) & (s >= 45) & (v >= 60)
    bp = blue | purple
    n = max(mass, 1e-6)
    hist, _ = np.histogram(
        h.astype(np.float32),
        bins=HUE_BINS,
        range=(0.0, 180.0),
        weights=ww,
    )
    hist = hist.astype(np.float32)
    hist = hist / (float(hist.sum()) + 1e-6)
    s_mean = float((s.astype(np.float32) * ww).sum() / n)
    v_mean = float((v.astype(np.float32) * ww).sum() / n)
    base = np.array(
        [
            float((bp.astype(np.float32) * ww).sum() / n),
            float((white.astype(np.float32) * ww).sum() / n),
            float((yellow.astype(np.float32) * ww).sum() / n),
            s_mean,
            v_mean,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, hist])


def jersey_feature_variant(crop: np.ndarray, cfg: dict) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    if cfg["mode"] == "baseline":
        return jersey_feature(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = _adaptive_non_green(hsv)
    std = float(crop.std())
    if std < MIN_CROP_STD:
        whiteish = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 125)
        if float(whiteish.mean()) < 0.45:
            return None
    hgt, wdt = crop.shape[:2]
    wgt = _spatial_weights(hgt, wdt, cfg)
    mode = cfg["mode"]
    if mode == "topk":
        flat = wgt.reshape(-1)
        thr = float(np.quantile(flat, 1.0 - float(cfg["keep"])))
        wgt = (wgt >= thr).astype(np.float32) * wgt
    if mode == "gauss_sat":
        xn, yn = _xy_norm(hgt, wdt)
        core = (np.abs(xn) <= 0.25) & (np.abs(yn) <= 0.25) & keep
        if int(core.sum()) >= 8:
            s_core = float(np.median(hsv[:, :, 1][core]))
            s = hsv[:, :, 1].astype(np.float32)
            if s_core <= 55:
                # white-ish core: down-weight saturated sleeve pixels
                wgt = wgt * (1.0 - 0.85 * np.clip((s - 55.0) / 100.0, 0.0, 1.0))
            else:
                # colored core: down-weight pale undershirt/white flashes
                wgt = wgt * (1.0 - 0.85 * np.clip((70.0 - s) / 70.0, 0.0, 1.0))
    if mode == "gauss_hue":
        xn, yn = _xy_norm(hgt, wdt)
        core = (np.abs(xn) <= 0.25) & (np.abs(yn) <= 0.25) & keep
        if int(core.sum()) >= 8:
            h0 = float(np.median(hsv[:, :, 0][core].astype(np.float32)))
            dh = np.abs(hsv[:, :, 0].astype(np.float32) - h0)
            dh = np.minimum(dh, 180.0 - dh)
            wgt = wgt * (dh <= float(cfg["hue_tol"])).astype(np.float32)
    return _feat_from_keep(hsv, keep, wgt)


def paint_opposite_ring(crop: np.ndarray, ring: float) -> np.ndarray:
    """Paint outer ring with opposite of center kit (simulates mismatched undershirt)."""
    out = crop.copy()
    h, w = out.shape[:2]
    xn, yn = _xy_norm(h, w)
    r = np.sqrt(xn * xn + yn * yn)
    core = r <= 0.35
    edge = r >= (1.0 - float(ring))
    if int(core.sum()) < 6:
        return out
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    s_m = float(np.median(hsv[:, :, 1][core]))
    h_m = float(np.median(hsv[:, :, 0][core]))
    # white-ish core → blue paint on ring; blue-ish core → white paint
    if s_m <= 55 or not (85 <= h_m <= 145):
        paint = np.array([210, 90, 40], dtype=np.uint8)  # blue BGR
    else:
        paint = np.array([235, 235, 235], dtype=np.uint8)
    out[edge] = paint
    return out


def extract_crops(max_crops: int, frame_stride: int) -> list[np.ndarray]:
    print(f"loading cache {CACHE}…", flush=True)
    raw = json.loads(CACHE.read_text())
    frames = list(raw["frames"])[:: max(1, frame_stride)]
    crops: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        if len(crops) >= max_crops:
            break
        row = raw["data"][str(fr)]
        for cam in raw["cams"]:
            hex_bgr = (row.get("bgr") or {}).get(cam)
            if not hex_bgr:
                continue
            img = cv2.imdecode(
                np.frombuffer(bytes.fromhex(hex_bgr), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is None:
                continue
            wh = (img.shape[1], img.shape[0])
            for d in row["dets"].get(cam) or []:
                if str(d.get("class_name", "player")).lower() == "ball":
                    continue
                crop = torso_crop(img, d["bbox"], cam=cam, frame_wh=wh)
                if crop is None or crop.size < 80:
                    continue
                if jersey_feature(crop) is None:
                    continue
                crops.append(crop.copy())
                if len(crops) >= max_crops:
                    break
            if len(crops) >= max_crops:
                break
        if i % 20 == 0:
            print(f"  fr={fr} crops={len(crops)}", flush=True)
    return crops


def save_crops(crops: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"c{i}": c for i, c in enumerate(crops)})
    meta = {"n": len(crops), "source": str(CACHE)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_crops(path: Path) -> list[np.ndarray]:
    z = np.load(path)
    keys = sorted(z.files, key=lambda k: int(k[1:]))
    return [z[k] for k in keys]


def labels_for_feats(feats: list[np.ndarray]) -> tuple[np.ndarray, float, list[int]] | None:
    fit = fit_match_centroids(feats, min_crops=TEAM_MIN_CROPS)
    if fit is None:
        return None
    cents, radius = fit
    labs = []
    for f in feats:
        tid, _ = assign_feature(f, cents, radius)
        labs.append(int(tid))
    return cents, radius, labs


def score_variant(
    name: str,
    cfg: dict,
    crops: list[np.ndarray],
    ring: float,
) -> dict:
    clean_feats: list[np.ndarray] = []
    stress_feats: list[np.ndarray] = []
    ok_idx: list[int] = []
    for i, crop in enumerate(crops):
        fc = jersey_feature_variant(crop, cfg)
        if fc is None:
            continue
        fs = jersey_feature_variant(paint_opposite_ring(crop, ring), cfg)
        if fs is None:
            continue
        clean_feats.append(fc)
        stress_feats.append(fs)
        ok_idx.append(i)
    n = len(clean_feats)
    row: dict = {"variant": name, "n_feat": n, "n_drop": len(crops) - n}
    if n < TEAM_MIN_CROPS:
        row["error"] = "too_few_feats"
        return row
    fit_c = labels_for_feats(clean_feats)
    if fit_c is None:
        row["error"] = "fit_failed"
        return row
    cents, radius, labs_c = fit_c
    labs_s = [int(assign_feature(f, cents, radius)[0]) for f in stress_feats]
    flips = sum(1 for a, b in zip(labs_c, labs_s) if a != b and a >= 0 and b >= 0)
    both_sure = sum(1 for a, b in zip(labs_c, labs_s) if a >= 0 and b >= 0)
    drift = float(
        np.mean(
            [
                np.linalg.norm(a[:KIT_DIM] - b[:KIT_DIM])
                for a, b in zip(clean_feats, stress_feats)
            ]
        )
    )
    sep = float(
        abs(cents[0, 0] - cents[1, 0]) + abs(cents[0, 1] - cents[1, 1])
    )
    n0 = sum(1 for t in labs_c if t == 0)
    n1 = sum(1 for t in labs_c if t == 1)
    bal = float(min(n0, n1) / max(max(n0, n1), 1))
    # polarity: team0 should be bluer
    polarity_ok = float(cents[0, 0] - cents[0, 1]) > float(cents[1, 0] - cents[1, 1])
    retain = 1.0 - (flips / max(both_sure, 1))
    coverage = float(n) / float(max(len(crops), 1))
    # composite: undershirt-robust + separable + balanced + covers most crops
    score = 100.0 * (
        0.45 * retain
        + 0.18 * min(sep / 0.8, 1.0)
        + 0.12 * bal
        + 0.10 * (1.0 - min(drift, 1.0))
        + 0.15 * coverage
    )
    if not polarity_ok:
        score *= 0.85
    row.update(
        {
            "retain": round(retain, 4),
            "flip_rate": round(1.0 - retain, 4),
            "drift_kit": round(drift, 4),
            "sep_bw": round(sep, 4),
            "balance": round(bal, 4),
            "coverage": round(coverage, 4),
            "n0": n0,
            "n1": n1,
            "polarity_ok": bool(polarity_ok),
            "score": round(score, 2),
            "mean_pair_dist": round(
                float(
                    np.mean(
                        [
                            feature_distance(clean_feats[i], stress_feats[i])
                            for i in range(n)
                        ]
                    )
                ),
                4,
            ),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.rebuild_crops or not CROPS_NPZ.is_file():
        crops = extract_crops(args.max_crops, args.frame_stride)
        save_crops(crops, CROPS_NPZ)
        print(f"saved {len(crops)} crops → {CROPS_NPZ}", flush=True)
    else:
        crops = load_crops(CROPS_NPZ)
        print(f"loaded {len(crops)} crops", flush=True)
    rows = []
    for name, cfg in VARIANTS:
        print(f"scoring {name}…", flush=True)
        rows.append(score_variant(name, cfg, crops, args.ring))
    rows_sorted = sorted(rows, key=lambda r: (-float(r.get("score", -1)), r["variant"]))
    payload = {
        "n_crops": len(crops),
        "ring": args.ring,
        "cache": str(CACHE),
        "ranking": rows_sorted,
        "winner": rows_sorted[0]["variant"] if rows_sorted else None,
    }
    out_path = OUT / "ab_jersey_undershirt_variants.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    md = ["# Jersey undershirt variant A/B", "", f"n_crops={len(crops)} ring={args.ring}", ""]
    md.append("| rank | variant | score | retain | flip | drift | sep | bal | cover |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_sorted, 1):
        if "error" in r:
            md.append(f"| {i} | {r['variant']} | err | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {i} | `{r['variant']}` | {r['score']:.1f} | {r['retain']:.3f} | "
            f"{r['flip_rate']:.3f} | {r['drift_kit']:.3f} | {r['sep_bw']:.3f} | "
            f"{r['balance']:.3f} | {r['coverage']:.3f} |"
        )
    (OUT / "ab_jersey_undershirt_variants.md").write_text("\n".join(md) + "\n")
    print(json.dumps(payload["ranking"][:5], indent=2))
    print(f"winner={payload['winner']} → {out_path}")


if __name__ == "__main__":
    main()
