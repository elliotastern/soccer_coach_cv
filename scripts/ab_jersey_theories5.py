#!/usr/bin/env python3
"""A/B the five ≥8/10 jersey theories on Match4 real dets.

T1 annulus+sticky · T2 P10-downweight fit · T3 freeze fit · T4 multi-cam agree gate · T5 birth-bal+age lock
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_jersey_undershirt_ideas10 import extract  # noqa: E402
from scripts.ab_jersey_undershirt_variants import paint_opposite_ring  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    TEAM_MIN_CROPS,
    assign_feature,
    fit_match_centroids,
    torso_crop,
)
CACHE = ROOT / "reports/eval_match3/team_id_strategy_grid/m4_90s_det_plain.json"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_theories5_ab"
OBS = OUT / "m4_theory_obs.npz"
RING = 0.24


def _xy_norm_iou(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    return float(inter / max(aw * ah + bw * bh - inter, 1e-6))


def _center(b):
    x, y, w, h = [float(v) for v in b]
    return x + 0.5 * w, y + 0.5 * h


def build_obs(max_frames: int = 0, frame_stride: int = 2) -> dict:
    print(f"loading {CACHE}…", flush=True)
    raw = json.loads(CACHE.read_text())
    frames = list(raw["frames"])[:: max(1, frame_stride)]
    if max_frames > 0:
        frames = frames[:max_frames]
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
                f50 = extract(crop, "center50")
                fann = extract(crop, "annulus30")
                if f50 is None and fann is None:
                    continue
                # stress retain probe features
                stressed = paint_opposite_ring(crop, RING)
                s50 = extract(stressed, "center50")
                sann = extract(stressed, "annulus30")
                rows.append(
                    {
                        "cam": cam,
                        "fr": int(fr),
                        "fr_i": i,
                        "bbox": [float(v) for v in d["bbox"]],
                        "f50": f50,
                        "fann": fann,
                        "s50": s50,
                        "sann": sann,
                        "crop": crop,  # keep for nothing — drop to save mem
                    }
                )
        if i % 30 == 0:
            print(f"  fr={fr} rows={len(rows)}", flush=True)
    # drop crops from save
    for r in rows:
        r.pop("crop", None)
    return {"frames": frames, "cams": list(raw["cams"]), "rows": rows}


def save_obs(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = bundle["rows"]
    n = len(rows)

    def stack(key):
        arr = []
        mask = np.zeros(n, dtype=np.uint8)
        for i, r in enumerate(rows):
            f = r.get(key)
            if f is None:
                arr.append(np.zeros(15, np.float32))
            else:
                mask[i] = 1
                arr.append(np.asarray(f, np.float32))
        return np.stack(arr), mask

    f50, m50 = stack("f50")
    fann, mann = stack("fann")
    s50, ms50 = stack("s50")
    sann, msann = stack("sann")
    np.savez_compressed(
        path,
        cams=np.array([r["cam"] for r in rows]),
        frs=np.array([r["fr"] for r in rows], np.int32),
        fr_is=np.array([r["fr_i"] for r in rows], np.int32),
        bboxes=np.array([r["bbox"] for r in rows], np.float32),
        f50=f50,
        fann=fann,
        s50=s50,
        sann=sann,
        m50=m50,
        mann=mann,
        ms50=ms50,
        msann=msann,
        frames=np.array(bundle["frames"], np.int32),
    )
    path.with_suffix(".json").write_text(
        json.dumps({"n": n, "n_frames": len(bundle["frames"])}, indent=2) + "\n"
    )


def load_obs(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    n = len(z["cams"])
    rows = []
    for i in range(n):
        rows.append(
            {
                "cam": str(z["cams"][i]),
                "fr": int(z["frs"][i]),
                "fr_i": int(z["fr_is"][i]),
                "bbox": z["bboxes"][i].tolist(),
                "f50": z["f50"][i] if z["m50"][i] else None,
                "fann": z["fann"][i] if z["mann"][i] else None,
                "s50": z["s50"][i] if z["ms50"][i] else None,
                "sann": z["sann"][i] if z["msann"][i] else None,
            }
        )
    return {"frames": [int(x) for x in z["frames"]], "rows": rows}


def build_tracks(rows: list[dict], iou_thr=0.25, max_center=120.0):
    active: dict[str, list] = {}
    tracks: dict[int, list] = {}
    next_id = 1
    by_fr: dict[int, list] = defaultdict(list)
    for r in rows:
        by_fr[r["fr_i"]].append(r)
    for fr_i in sorted(by_fr):
        by_cam: dict[str, list] = defaultdict(list)
        for r in by_fr[fr_i]:
            by_cam[r["cam"]].append(r)
        new_active = {}
        for cam, dets in by_cam.items():
            prev = active.get(cam) or []
            used_p, used_d = set(), set()
            pairs = []
            for pi, p in enumerate(prev):
                for di, d in enumerate(dets):
                    ov = _xy_norm_iou(p["bbox"], d["bbox"])
                    if ov >= iou_thr:
                        pairs.append((1.0 + ov, pi, di))
                    else:
                        ax, ay = _center(p["bbox"])
                        bx, by = _center(d["bbox"])
                        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                        if dist <= max_center:
                            pairs.append((0.5 * (1.0 - dist / max_center), pi, di))
            pairs.sort(reverse=True)
            alive = []
            for sc, pi, di in pairs:
                if pi in used_p or di in used_d:
                    continue
                used_p.add(pi)
                used_d.add(di)
                tid = prev[pi]["tid"]
                row = {**dets[di], "tid": tid}
                tracks[tid].append(row)
                alive.append({"tid": tid, "bbox": dets[di]["bbox"]})
            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                tid = next_id
                next_id += 1
                tracks[tid] = [{**d, "tid": tid}]
                alive.append({"tid": tid, "bbox": d["bbox"]})
            new_active[cam] = alive
        active = new_active
    return list(tracks.values())


def fit_feats(feats: list, weights: list[float] | None = None):
    bag = []
    if weights is None:
        bag = [f for f in feats if f is not None]
    else:
        for f, w in zip(feats, weights):
            if f is None:
                continue
            reps = max(1, int(round(float(w) * 3)))
            bag.extend([f] * min(reps, 4))
    if len(bag) > 900:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(bag), 900, replace=False)
        bag = [bag[i] for i in idx]
    return fit_match_centroids(bag, min_crops=TEAM_MIN_CROPS)


def label_feat(f, cents, radius, dual_soft=True):
    if f is None or cents is None:
        return -1, 0.0
    tid, conf = assign_feature(f, cents, radius)
    b, w = float(f[0]), float(f[1])
    if dual_soft and b >= 0.35 and w >= 0.35:
        return 1, 0.70
    return int(tid), float(conf)


def apply_strategy(name: str, tracks: list[list[dict]], rows_by_key=None) -> dict:
    """Return flat labels in track-major order + metrics inputs."""
    # gather feats for fit
    feat_key = "fann" if name == "T1_annulus_sticky" else "f50"
    stress_key = "sann" if name == "T1_annulus_sticky" else "s50"

    flat_rows = [r for tr in tracks for r in tr]
    feats = [r.get(feat_key) for r in flat_rows]
    cams = [r["cam"] for r in flat_rows]

    # fit
    if name == "T2_p10_downweight":
        w = [0.4 if c == "P10" else 1.0 for c in cams]
        fit = fit_feats(feats, w)
    elif name == "T3_freeze_first30":
        # fit only early frames then freeze
        early = [r.get(feat_key) for r in flat_rows if r["fr_i"] <= 30]
        fit = fit_feats(early)
    else:
        fit = fit_feats(feats)
    if fit is None:
        return {"name": name, "error": "fit_failed"}
    cents, radius = fit

    lengths = [len(tr) for tr in tracks]
    labs_all = []
    confs_all = []
    # per-frame cam majorities for T4
    frame_cam_team: dict[tuple[int, str], list[int]] = defaultdict(list)

    # first pass instant labels
    instant = []
    for r in flat_rows:
        tid, conf = label_feat(r.get(feat_key), cents, radius, dual_soft=True)
        instant.append((tid, conf, r))
        if tid >= 0:
            frame_cam_team[(r["fr_i"], r["cam"])].append(tid)

    frame_cam_maj = {}
    for k, ts in frame_cam_team.items():
        frame_cam_maj[k] = Counter(ts).most_common(1)[0][0]

    # per-frame overall share for T5
    frame_labs: dict[int, list[int]] = defaultdict(list)
    for tid, conf, r in instant:
        if tid >= 0:
            frame_labs[r["fr_i"]].append(tid)

    # track-wise temporal
    k = 0
    for tr in tracks:
        n = len(tr)
        chunk = instant[k : k + n]
        k += n
        labs = [t for t, _, _ in chunk]
        confs = [c for _, c, _ in chunk]
        held = labs[0]
        age = 0
        out = []
        for i, (tid, conf, r) in enumerate(chunk):
            # T4: multi-cam agree gate
            if name == "T4_multicam_agree" and i > 0 and held >= 0:
                fr_i = r["fr_i"]
                majs = [
                    frame_cam_maj[(fr_i, cam)]
                    for cam in {x["cam"] for x in flat_rows if x["fr_i"] == fr_i}
                    if (fr_i, cam) in frame_cam_maj
                ]
                # cams agreeing with new tid
                if tid != held and tid >= 0:
                    agree = sum(1 for m in majs if m == tid)
                    if agree < 2 and conf < 0.92:
                        tid = held

            # sticky / age lock
            if name in (
                "T1_annulus_sticky",
                "baseline_center_sticky",
                "T2_p10_downweight",
                "T3_freeze_first30",
                "T5_birth_agelock",
                "T4_multicam_agree",
            ):
                if i == 0:
                    held = tid
                    age = 0
                else:
                    age += 1
                    if name == "T5_birth_agelock" and age >= 2 and held >= 0:
                        # hard lock unless very strong
                        if tid != held and conf < 0.95:
                            tid = held
                    else:
                        # sticky streak-ish: keep held unless high conf
                        if held >= 0 and tid != held and conf < 0.90:
                            tid = held
                        elif held < 0:
                            held = tid
                        else:
                            held = tid if conf >= 0.90 else held
                            tid = held
                # T5 birth balance on first obs of track
                if name == "T5_birth_agelock" and i == 0 and tid >= 0:
                    fl = frame_labs.get(r["fr_i"], [])
                    if len(fl) >= 6:
                        n0 = fl.count(0)
                        n1 = fl.count(1)
                        if n0 + n1 > 0:
                            share0 = n0 / (n0 + n1)
                            if share0 > 0.58 and tid == 0 and conf <= 0.72:
                                tid = 1
                            elif share0 < 0.42 and tid == 1 and conf <= 0.72:
                                tid = 0
                        held = tid

            out.append(tid)
            labs_all.append(tid)
            confs_all.append(conf)
        # end track

    # stress retain
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

    # flips
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
    # frame skew severity
    by_fr: dict[int, list[int]] = defaultdict(list)
    k = 0
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not OBS.is_file():
        bundle = build_obs(frame_stride=2)
        save_obs(bundle, OBS)
        print(f"saved {len(bundle['rows'])} → {OBS}", flush=True)
    else:
        bundle = load_obs(OBS)
        print(f"loaded {len(bundle['rows'])} rows", flush=True)

    tracks = build_tracks(bundle["rows"])
    tracks = [t for t in tracks if t]
    print(f"tracks={len(tracks)} mean_len={np.mean([len(t) for t in tracks]):.2f}", flush=True)

    strategies = [
        "baseline_center_sticky",
        "T1_annulus_sticky",
        "T2_p10_downweight",
        "T3_freeze_first30",
        "T4_multicam_agree",
        "T5_birth_agelock",
    ]
    rows = []
    for name in strategies:
        print(f"scoring {name}…", flush=True)
        rows.append(apply_strategy(name, tracks))

    ranked = sorted([r for r in rows if "score" in r], key=lambda r: -r["score"])
    payload = {
        "n_rows": len(bundle["rows"]),
        "n_tracks": len(tracks),
        "ranking": ranked,
        "winner": ranked[0]["name"] if ranked else None,
    }
    (OUT / "ab_theories5.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# Five jersey theories — A/B",
        "",
        f"n_rows={payload['n_rows']} tracks={payload['n_tracks']}",
        "",
        f"**Winner: `{payload['winner']}`**",
        "",
        "| rank | strategy | score | share | off50 | frame_off | bad_fr | retain | flips |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        md.append(
            f"| {i} | `{r['name']}` | {r['score']:.1f} | {r['pct0']:.1f}/{r['pct1']:.1f} | "
            f"{r['off50']:.1f} | {r['mean_frame_off']:.3f} | {r['bad_frame_frac']:.3f} | "
            f"{r['retain_stress']:.3f} | {r['flip_rate']:.3f} |"
        )
    (OUT / "ab_theories5.md").write_text("\n".join(md) + "\n")
    print(json.dumps(ranked, indent=2))
    print("WINNER", payload["winner"])


if __name__ == "__main__":
    main()
