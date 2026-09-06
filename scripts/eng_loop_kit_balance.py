#!/usr/bin/env python3
"""Eng-loop: improve kit team share toward 50/50 on Match 4 traj obs (fast).

Diagnosis: hard_center_50 crops are already ~62% blue>white; P10 ~74/26.
Trajectory cannot invent white mass. This A/Bs assignment / rebalance levers.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_jersey_traj_variants import build_tracks, load_obs, track_flips  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    TEAM_MIN_CROPS,
    assign_feature,
    fit_match_centroids,
)

OBS = ROOT / "reports/eval_match3/improve_eng_loop/kit_traj_ab/m4_traj_obs.npz"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_balance_eng_loop"
FIT_CAP = 900
RNG = np.random.RandomState(0)


def fit_sub(feats: list[np.ndarray]):
    if len(feats) > FIT_CAP:
        idx = RNG.choice(len(feats), FIT_CAP, replace=False)
        feats = [feats[i] for i in idx]
    return fit_match_centroids(feats, min_crops=TEAM_MIN_CROPS)


def score_labels(labs: list[int], track_lab_lists: list[list[int]]) -> dict:
    n0 = sum(1 for t in labs if t == 0)
    n1 = sum(1 for t in labs if t == 1)
    nu = sum(1 for t in labs if t < 0)
    n = n0 + n1
    if n == 0:
        return {"error": "empty", "score": -1}
    p0 = 100.0 * n0 / n
    p1 = 100.0 * n1 / n
    dev = abs(p0 - 50.0)
    bal = float(min(n0, n1) / max(n0, n1))
    flips = edges = 0
    for tl in track_lab_lists:
        f, e = track_flips(tl)
        flips += f
        edges += e
    flip_rate = float(flips / max(edges, 1))
    cover = float(n / max(len(labs), 1))
    score = 100.0 * (
        0.55 * (1.0 - min(dev / 20.0, 1.0))
        + 0.20 * bal
        + 0.15 * (1.0 - flip_rate)
        + 0.10 * cover
    )
    return {
        "n0": n0,
        "n1": n1,
        "n_unsure": nu,
        "pct0": round(p0, 2),
        "pct1": round(p1, 2),
        "dev_from_50_pp": round(dev, 2),
        "balance": round(bal, 4),
        "flip_rate": round(flip_rate, 4),
        "coverage": round(cover, 4),
        "score": round(score, 2),
    }


def equalize_by_conf(labs, confs, target_dev=0.02, drop=False):
    labs = list(labs)
    order = np.argsort(confs)
    for i in order:
        n0 = sum(1 for t in labs if t == 0)
        n1 = sum(1 for t in labs if t == 1)
        n = n0 + n1
        if n == 0 or abs(n0 / n - 0.5) <= target_dev:
            break
        if n0 > n1 and labs[i] == 0:
            labs[i] = -1 if drop else 1
        elif n1 > n0 and labs[i] == 1:
            labs[i] = -1 if drop else 0
    return labs


def vote3(labs: list[int]) -> list[int]:
    out = []
    for i in range(len(labs)):
        w = [labs[j] for j in range(max(0, i - 2), i + 1) if labs[j] >= 0]
        out.append(Counter(w).most_common(1)[0][0] if w else labs[i])
    return out


def split_tracks(flat: list[int], lengths: list[int]) -> list[list[int]]:
    out, k = [], 0
    for n in lengths:
        out.append(flat[k : k + n])
        k += n
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("load obs…", flush=True)
    bundle = load_obs(OBS)
    tracks = [t for t in build_tracks(bundle["by_fr"], 0.30, 120.0) if t]
    lengths = [len(t) for t in tracks]
    flat_obs = [o for t in tracks for o in t]
    feats = [o["feat"] for o in flat_obs]
    cams = [o["cam"] for o in flat_obs]
    fr_is = [int(o["fr_i"]) for o in flat_obs]
    print(f"tracks={len(tracks)} obs={len(flat_obs)}", flush=True)

    cam_n = Counter(cams)
    pools = {
        "all": feats,
        "no_p10": [f for f, c in zip(feats, cams) if c != "P10"],
        "inv_cam": [],
    }
    # inv_cam: upsample underrepresented cams into fit bag (cap)
    inv_bag = []
    for f, c in zip(feats, cams):
        reps = max(1, int(round(max(cam_n.values()) / max(cam_n[c], 1))))
        inv_bag.extend([f] * min(reps, 3))
    pools["inv_cam"] = inv_bag

    fits = {}
    for k, bag in pools.items():
        print(f"fit {k} n={len(bag)}…", flush=True)
        fits[k] = fit_sub(bag)
        if fits[k] is None:
            del fits[k]

    # precompute labels per fit
    base = {}
    for k, (cents, radius) in fits.items():
        print(f"assign {k}…", flush=True)
        labs, confs = [], []
        for f in feats:
            tid, conf = assign_feature(f, cents, radius)
            labs.append(int(tid))
            confs.append(float(conf))
        base[k] = (np.array(labs), np.array(confs), cents, radius)

    def dual_mask(mode: str):
        b = np.array([float(f[0]) for f in feats])
        w = np.array([float(f[1]) for f in feats])
        if mode == "dual_to_white":
            return (b >= 0.35) & (w >= 0.35)
        if mode == "dual_to_unsure":
            return (b >= 0.35) & (w >= 0.35)
        if mode == "dual_unsure_soft":
            return (np.minimum(b, w) >= 0.30) & (np.abs(b - w) < 0.12)
        if mode == "prefer_white_close":
            return (w >= b - 0.05) & (w >= 0.28)
        return np.zeros(len(feats), dtype=bool)

    variants = []

    def add(name, fit_key, labs, confs):
        tl = split_tracks(list(labs), lengths)
        row = score_labels(list(labs), tl)
        row["variant"] = name
        row["fit"] = fit_key
        variants.append(row)
        print(
            f"  {name}: {row['pct0']:.1f}/{row['pct1']:.1f} off={row['dev_from_50_pp']:.1f} "
            f"flip={row['flip_rate']:.3f} score={row['score']:.1f}",
            flush=True,
        )

    # baseline + levers
    for fit_key in fits:
        labs0, confs0, _, _ = base[fit_key]
        add(f"{fit_key}__baseline", fit_key, labs0, confs0)

        labs = labs0.copy()
        m = dual_mask("dual_to_white")
        labs[m] = 1
        add(f"{fit_key}__dual_to_white", fit_key, labs, confs0)

        labs = labs0.copy()
        m = dual_mask("dual_to_unsure")
        labs[m] = -1
        add(f"{fit_key}__dual_to_unsure", fit_key, labs, confs0)

        labs = labs0.copy()
        m = dual_mask("prefer_white_close")
        labs[m] = 1
        add(f"{fit_key}__prefer_white_close", fit_key, labs, confs0)

        # dual white + vote3
        labs = labs0.copy()
        labs[dual_mask("dual_to_white")] = 1
        tl = [vote3(t) for t in split_tracks(list(labs), lengths)]
        flat = [x for t in tl for x in t]
        add(f"{fit_key}__dual_white+vote3", fit_key, np.array(flat), confs0)

        # per-frame equalize weak
        by_fr = defaultdict(list)
        for i, fr in enumerate(fr_is):
            by_fr[fr].append(i)
        labs = labs0.copy()
        for idxs in by_fr.values():
            sub_l = [int(labs[i]) for i in idxs]
            sub_c = [float(confs0[i]) for i in idxs]
            sub2 = equalize_by_conf(sub_l, sub_c, target_dev=0.05)
            for i, lab in zip(idxs, sub2):
                labs[i] = lab
        add(f"{fit_key}__equalize_frame", fit_key, labs, confs0)

        # dual white then frame equalize
        labs = labs0.copy()
        labs[dual_mask("dual_to_white")] = 1
        for idxs in by_fr.values():
            sub_l = [int(labs[i]) for i in idxs]
            sub_c = [float(confs0[i]) for i in idxs]
            sub2 = equalize_by_conf(sub_l, sub_c, target_dev=0.05)
            for i, lab in zip(idxs, sub2):
                labs[i] = lab
        add(f"{fit_key}__dual_white+eq_frame", fit_key, labs, confs0)

        # majority drop weak → unsure
        labs = np.array(equalize_by_conf(list(labs0), list(confs0), target_dev=0.05, drop=True))
        add(f"{fit_key}__majority_drop_weak", fit_key, labs, confs0)

        # global equalize (metric ceiling / gaming check)
        labs = np.array(equalize_by_conf(list(labs0), list(confs0), target_dev=0.02, drop=False))
        add(f"{fit_key}__equalize_global", fit_key, labs, confs0)

    by_score = sorted(variants, key=lambda r: -float(r.get("score", -1)))
    by_50 = sorted(variants, key=lambda r: (r.get("dev_from_50_pp", 99), -r.get("score", 0)))
    base_row = next(r for r in variants if r["variant"] == "all__baseline")
    # practical: not pure global equalize; coverage >= 0.75
    practical = [
        r
        for r in by_score
        if "equalize_global" not in r["variant"] and r.get("coverage", 0) >= 0.75
    ]
    pick = practical[0] if practical else by_score[0]

    note = (
        "Feature mass is blue-heavy (~62% blue>white; P10 74/26). Do not chase 50/50 by "
        "global label flipping alone (metric gaming). Highest-confidence product path: "
        "(1) keep hard_center_50 jersey sampling, (2) dual-color rule: if blue&white fracs "
        "both high on center crop → label white (undershirt), (3) optional per-frame soft "
        "equalize of weakest majority labels, (4) add more white kit-ref samples / downweight "
        "P10 in centroid fit. Trajectory vote helps flips, not share."
    )

    payload = {
        "before": base_row,
        "practical_pick": pick,
        "closest_50": by_50[0],
        "ranking_by_score": by_score[:20],
        "ranking_by_50_50": by_50[:20],
        "confidence_next_path": 0.9,
        "next_path": note,
        "diagnosis": {
            "blue_gt_white_frac": 0.620,
            "p10": "73.9/26.1",
            "p8": "49.3/50.7",
            "whiteish_but_team0": 474,
        },
    }
    (OUT / "loop_kit_balance.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# Kit balance eng-loop",
        "",
        "## Diagnosis",
        "- Crops already **62% blue>white**; **P10 74/26**, P8 ~50/50.",
        "- High-conf labels are *more* blue-skewed. Traj cannot invent white mass.",
        "",
        f"## Practical pick `{pick['variant']}` (confidence 0.90 on path)",
        "",
        "| metric | before | after |",
        "|---|---:|---:|",
        f"| share | {base_row['pct0']:.1f}/{base_row['pct1']:.1f} | {pick['pct0']:.1f}/{pick['pct1']:.1f} |",
        f"| off 50/50 | {base_row['dev_from_50_pp']:.1f} | {pick['dev_from_50_pp']:.1f} |",
        f"| flips | {base_row['flip_rate']:.3f} | {pick['flip_rate']:.3f} |",
        f"| coverage | {base_row['coverage']:.3f} | {pick['coverage']:.3f} |",
        "",
        "## Top by score",
        "",
        "| rank | variant | pct0 | pct1 | off50 | flips | cover | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(by_score[:15], 1):
        md.append(
            f"| {i} | `{r['variant']}` | {r['pct0']:.1f} | {r['pct1']:.1f} | "
            f"{r['dev_from_50_pp']:.1f} | {r['flip_rate']:.3f} | {r['coverage']:.3f} | {r['score']:.1f} |"
        )
    md += ["", "## Next path", "", note, ""]
    (OUT / "loop_kit_balance.md").write_text("\n".join(md) + "\n")
    print("PRACTICAL", pick["variant"], pick)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
