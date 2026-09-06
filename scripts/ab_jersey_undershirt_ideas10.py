#!/usr/bin/env python3
"""10 undershirt-aware jersey ideas — A/B on Match4 real torso crops.

Outer kit can disagree with undershirt (white jersey + blue sleeves).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_jersey_undershirt_variants import paint_opposite_ring  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    HUE_BINS,
    MIN_CROP_STD,
    MIN_JERSEY_FRAC,
    TEAM_MIN_CROPS,
    _adaptive_non_green,
    assign_feature,
    feature_distance,
    fit_match_centroids,
)

CROPS = ROOT / "reports/eval_match3/improve_eng_loop/kit_undershirt_ab/m4_torso_crops.npz"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_undershirt_ideas10"
RING = 0.24
TRACK_LEN = 5

IDEAS = [
    (1, "center50_only", "Sample only center 50×50% (ignore sleeve/collar).", "center50", "cluster", "none"),
    (2, "annulus_zero_30", "Zero-weight outer 30% ring (undershirt zone).", "annulus30", "cluster", "none"),
    (3, "center_vs_edge_vote", "Center white-dom + edge blue-dom → white kit.", "center_edge", "center_wins", "none"),
    (4, "low_sat_body", "Up-weight low-S pixels (white body vs saturated sleeve).", "low_sat", "cluster", "none"),
    (5, "outer_highsat_down", "Down-weight high-S only in outer third.", "outer_sat", "cluster", "none"),
    (6, "dual_soft_white", "Both blue&white ≥0.35 on center → white @0.70.", "center50", "dual_soft", "none"),
    (7, "dual_unsure_vote5", "Dual-color → unsure; vote5 fills identity.", "center50", "dual_unsure", "vote5"),
    (8, "edge_blue_ignore", "If outer much bluer than center, use center-only fracs.", "edge_ignore", "cluster", "none"),
    (9, "median5_sticky", "Center50 + sticky hold (anti-flicker).", "center50", "cluster", "sticky"),
    (10, "dual_soft_vote5_bal", "Center50 + dual-soft + vote5 + soft 50/50 nudge.", "center50", "dual_soft", "vote5_bal"),
]


def _xy_norm(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    xn = ((xx - cx) / max(cx, 1.0)).astype(np.float32)
    yn = ((yy - cy) / max(cy, 1.0)).astype(np.float32)
    return xn, yn


def _feat_w(hsv, keep, wgt):
    ww = wgt * keep.astype(np.float32)
    mass = float(ww.sum())
    if mass < 18.0 or float(keep.mean()) < MIN_JERSEY_FRAC:
        return None
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = keep & (h >= 85) & (h <= 145) & (s >= 35)
    purple = keep & (h >= 125) & (h <= 170) & (s >= 30)
    white = keep & (s <= 55) & (v >= 125)
    yellow = keep & (h >= 12) & (h <= 40) & (s >= 45) & (v >= 60)
    n = max(mass, 1e-6)
    hist, _ = np.histogram(h.astype(np.float32), bins=HUE_BINS, range=(0.0, 180.0), weights=ww)
    hist = hist.astype(np.float32)
    hist = hist / (float(hist.sum()) + 1e-6)
    bp = blue | purple
    base = np.array(
        [
            float((bp.astype(np.float32) * ww).sum() / n),
            float((white.astype(np.float32) * ww).sum() / n),
            float((yellow.astype(np.float32) * ww).sum() / n),
            float((s.astype(np.float32) * ww).sum() / n),
            float((v.astype(np.float32) * ww).sum() / n),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, hist])


def _prep(crop):
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = _adaptive_non_green(hsv)
    if float(crop.std()) < MIN_CROP_STD:
        whiteish = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 125)
        if float(whiteish.mean()) < 0.45:
            return None
    return hsv, keep


def extract(crop, mode: str):
    prep = _prep(crop)
    if prep is None:
        return None
    hsv, keep = prep
    hgt, wdt = crop.shape[:2]
    xn, yn = _xy_norm(hgt, wdt)
    r = np.sqrt(xn * xn + yn * yn)

    if mode == "center50":
        return _feat_w(hsv, keep, ((np.abs(xn) <= 0.50) & (np.abs(yn) <= 0.50)).astype(np.float32))
    if mode == "annulus30":
        return _feat_w(hsv, keep, (r <= 0.70).astype(np.float32))
    if mode == "low_sat":
        s = hsv[:, :, 1].astype(np.float32)
        wgt = np.clip(1.0 - s / 180.0, 0.15, 1.0)
        wgt *= ((np.abs(xn) <= 0.55) & (np.abs(yn) <= 0.55)).astype(np.float32)
        return _feat_w(hsv, keep, wgt)
    if mode == "outer_sat":
        s = hsv[:, :, 1].astype(np.float32)
        wgt = np.ones((hgt, wdt), np.float32)
        outer = r >= 0.55
        wgt[outer] = np.clip(1.0 - s[outer] / 140.0, 0.1, 1.0)
        return _feat_w(hsv, keep, wgt)
    if mode == "edge_ignore":
        f_core = _feat_w(hsv, keep, (r <= 0.45).astype(np.float32))
        f_edge = _feat_w(hsv, keep, (r >= 0.65).astype(np.float32))
        if f_core is None:
            return None
        if f_edge is not None and float(f_edge[0] - f_core[0]) > 0.18:
            return f_core
        return _feat_w(hsv, keep, (r <= 0.70).astype(np.float32))
    if mode == "center_edge":
        f_c = _feat_w(hsv, keep, (r <= 0.40).astype(np.float32))
        f_e = _feat_w(hsv, keep, (r >= 0.60).astype(np.float32))
        if f_c is None:
            return None
        return {"feat": f_c, "edge": f_e}
    return None


def assign_one(feat, edge, cents, radius, mode: str):
    if feat is None:
        return -1, 0.0
    tid, conf = assign_feature(feat, cents, radius)
    b, w = float(feat[0]), float(feat[1])
    if mode == "dual_soft" and b >= 0.35 and w >= 0.35:
        return 1, 0.70
    if mode == "dual_unsure" and b >= 0.35 and w >= 0.35:
        return -1, 0.0
    if mode == "center_wins" and edge is not None:
        if float(feat[1]) >= float(feat[0]) + 0.05 and float(edge[0]) >= float(edge[1]) + 0.10:
            return 1, 0.72
    return int(tid), float(conf)


def temporalize(labs, confs, mode: str):
    n = len(labs)
    out = list(labs)
    if mode == "none":
        return out
    # fake tracks of TRACK_LEN
    for start in range(0, n - n % TRACK_LEN, TRACK_LEN):
        sl = slice(start, start + TRACK_LEN)
        tl = list(labs[sl])
        tc = list(confs[sl])
        cur = list(tl)
        if mode in ("vote5", "vote5_bal"):
            for i in range(TRACK_LEN):
                w = [tl[j] for j in range(max(0, i - 4), i + 1) if tl[j] >= 0]
                cur[i] = Counter(w).most_common(1)[0][0] if w else tl[i]
        if mode == "sticky":
            held = tl[0]
            for i in range(TRACK_LEN):
                window = [tl[j] for j in range(max(0, i - 4), i + 1) if tl[j] >= 0]
                if not window:
                    cur[i] = held if held >= 0 else tl[i]
                    continue
                voted, vn = Counter(window).most_common(1)[0]
                if held < 0:
                    held = voted
                elif voted != held and vn >= 3 and tc[i] >= 0.85:
                    held = voted
                cur[i] = held
        out[sl] = cur
    if mode == "vote5_bal":
        order = np.argsort(confs)
        for i in order:
            n0 = sum(1 for t in out if t == 0)
            n1 = sum(1 for t in out if t == 1)
            tot = n0 + n1
            if tot == 0 or abs(n0 / tot - 0.5) <= 0.08:
                break
            if n0 > n1 and out[i] == 0 and confs[i] <= 0.72:
                out[i] = 1
            elif n1 > n0 and out[i] == 1 and confs[i] <= 0.72:
                out[i] = 0
    return out


def track_flip_rate(labs):
    flips = edges = 0
    for start in range(0, len(labs) - len(labs) % TRACK_LEN, TRACK_LEN):
        tl = labs[start : start + TRACK_LEN]
        for a, b in zip(tl, tl[1:]):
            if a < 0 or b < 0:
                continue
            edges += 1
            if a != b:
                flips += 1
    return float(flips / max(edges, 1))


def load_crops(path: Path):
    z = np.load(path)
    keys = sorted(z.files, key=lambda k: int(k[1:]))
    return [z[k] for k in keys]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops = load_crops(CROPS)
    n = len(crops) - (len(crops) % TRACK_LEN)
    crops = crops[:n]
    print(f"crops={n}", flush=True)

    rows = []
    for iid, name, idea, feat_m, assign_m, temp_m in IDEAS:
        print(f"[{iid}] {name}…", flush=True)
        raw = [extract(c, feat_m) for c in crops]
        feats, edges = [], []
        for r in raw:
            if isinstance(r, dict):
                feats.append(r["feat"])
                edges.append(r.get("edge"))
            else:
                feats.append(r)
                edges.append(None)
        ok = [f for f in feats if f is not None]
        if len(ok) < TEAM_MIN_CROPS:
            rows.append({"id": iid, "name": name, "error": "too_few"})
            continue
        fit = fit_match_centroids(ok, min_crops=TEAM_MIN_CROPS)
        if fit is None:
            rows.append({"id": iid, "name": name, "error": "fit_failed"})
            continue
        cents, radius = fit
        labs, confs = [], []
        for f, e in zip(feats, edges):
            tid, conf = assign_one(f, e, cents, radius, assign_m)
            labs.append(tid)
            confs.append(conf)
        labs = temporalize(labs, confs, temp_m)

        # stress retain
        same = tot = 0
        for i, c in enumerate(crops):
            if labs[i] < 0:
                continue
            rs = extract(paint_opposite_ring(c, RING), feat_m)
            if isinstance(rs, dict):
                fe, ee = rs["feat"], rs.get("edge")
            else:
                fe, ee = rs, None
            if fe is None:
                continue
            tid, _ = assign_one(fe, ee, cents, radius, assign_m)
            if tid < 0:
                continue
            tot += 1
            if tid == labs[i]:
                same += 1
        retain = same / max(tot, 1)

        n0 = sum(1 for t in labs if t == 0)
        n1 = sum(1 for t in labs if t == 1)
        nu = sum(1 for t in labs if t < 0)
        labeled = n0 + n1
        p0 = 100.0 * n0 / max(labeled, 1)
        p1 = 100.0 * n1 / max(labeled, 1)
        off = abs(p0 - 50.0)
        bal = float(min(n0, n1) / max(max(n0, n1), 1))
        flip = track_flip_rate(labs)
        cover = labeled / max(len(labs), 1)
        score = 100.0 * (
            0.35 * (1.0 - min(off / 20.0, 1.0))
            + 0.25 * retain
            + 0.20 * (1.0 - flip)
            + 0.10 * bal
            + 0.10 * cover
        )
        row = {
            "id": iid,
            "name": name,
            "idea": idea,
            "pct0": round(p0, 2),
            "pct1": round(p1, 2),
            "off50": round(off, 2),
            "retain_stress": round(retain, 4),
            "flip_rate": round(flip, 4),
            "balance": round(bal, 4),
            "coverage": round(cover, 4),
            "n_unsure": nu,
            "score": round(score, 2),
        }
        rows.append(row)
        print(
            f"  score={row['score']} share={row['pct0']}/{row['pct1']} "
            f"retain={row['retain_stress']} flip={row['flip_rate']}",
            flush=True,
        )

    ranked = sorted([r for r in rows if "score" in r], key=lambda r: -r["score"])
    payload = {"n_crops": n, "ring": RING, "ranking": ranked, "winner": ranked[0]["name"] if ranked else None}
    (OUT / "ab_undershirt_ideas10.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# 10 undershirt jersey ideas — tested",
        "",
        f"Match4 real torsos n={n}; stress = opposite-color outer ring ({RING}).",
        "",
        f"**Winner: `{payload['winner']}`**",
        "",
        "| rank | # | idea | score | share | off50 | retain | flips |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        md.append(
            f"| {i} | {r['id']} | `{r['name']}` | {r['score']:.1f} | "
            f"{r['pct0']:.1f}/{r['pct1']:.1f} | {r['off50']:.1f} | "
            f"{r['retain_stress']:.3f} | {r['flip_rate']:.3f} |"
        )
    md += ["", "## Ideas", ""]
    for iid, name, idea, *_ in IDEAS:
        md.append(f"{iid}. **{name}** — {idea}")
    (OUT / "ab_undershirt_ideas10.md").write_text("\n".join(md) + "\n")
    print("WINNER", payload["winner"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
