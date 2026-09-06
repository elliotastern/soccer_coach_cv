#!/usr/bin/env python3
"""A/B trajectory / history team labeling vs 50/50 on Match 4 det cache.

Builds IoU tracks (cache has no ByteTrack ids), then compares 10 history variants
against per-frame labeling. Jersey features use hard_center_50 (undershirt A/B winner).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_jersey_undershirt_variants import (  # noqa: E402
    jersey_feature_variant,
)
from src.perception.team_core import (  # noqa: E402
    TEAM_MIN_CROPS,
    assign_feature,
    fit_match_centroids,
    torso_crop,
    tracklet_median_feature,
)

CACHE = ROOT / "reports/eval_match3/team_id_strategy_grid/m4_90s_det_plain.json"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_traj_ab"
OBS_PATH = OUT / "m4_traj_obs.npz"
FEAT_CFG = {"mode": "hard", "frac": 0.50}

# control + 10 trajectory variations
VARIANTS = [
    ("per_frame", {"kind": "per_frame"}),
    ("sticky_0.70", {"kind": "sticky", "flip_conf": 0.70}),
    ("sticky_0.85", {"kind": "sticky", "flip_conf": 0.85}),
    ("vote_last3", {"kind": "vote", "k": 3}),
    ("vote_last5", {"kind": "vote", "k": 5}),
    ("vote_last8", {"kind": "vote", "k": 8}),
    ("feat_median_5", {"kind": "feat_median", "k": 5}),
    ("feat_median_all", {"kind": "feat_median", "k": 0}),
    ("feat_ema_0.30", {"kind": "feat_ema", "alpha": 0.30}),
    ("feat_ema_0.15", {"kind": "feat_ema", "alpha": 0.15}),
    ("vote5_sticky085", {"kind": "vote_sticky", "k": 5, "flip_conf": 0.85}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-obs", action="store_true")
    p.add_argument("--iou", type=float, default=0.30)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all cache frames")
    return p.parse_args()


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1e-6))


def extract_obs(max_frames: int) -> dict:
    print(f"loading {CACHE}…", flush=True)
    raw = json.loads(CACHE.read_text())
    frames = list(raw["frames"])
    if max_frames > 0:
        frames = frames[:max_frames]
    # per-frame lists of dicts: cam, bbox, feat, fr_idx, fr
    by_fr: list[list[dict]] = []
    feats_all: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        row = raw["data"][str(fr)]
        obs_fr: list[dict] = []
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
                if crop is None:
                    continue
                feat = jersey_feature_variant(crop, FEAT_CFG)
                if feat is None:
                    continue
                obs_fr.append(
                    {
                        "cam": cam,
                        "bbox": [float(v) for v in d["bbox"]],
                        "feat": feat,
                        "fr": int(fr),
                        "fr_i": i,
                    }
                )
                feats_all.append(feat)
        by_fr.append(obs_fr)
        if i % 40 == 0:
            print(f"  frame {fr} obs={sum(len(x) for x in by_fr)}", flush=True)
    return {
        "frames": frames,
        "cams": list(raw["cams"]),
        "by_fr": by_fr,
        "n_obs": int(sum(len(x) for x in by_fr)),
        "n_feat_ok": len(feats_all),
    }


def save_obs(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # flatten for npz
    cams, bboxes, frs, fr_is, feats = [], [], [], [], []
    for obs_fr in bundle["by_fr"]:
        for o in obs_fr:
            cams.append(o["cam"])
            bboxes.append(o["bbox"])
            frs.append(o["fr"])
            fr_is.append(o["fr_i"])
            feats.append(o["feat"])
    np.savez_compressed(
        path,
        cams=np.array(cams),
        bboxes=np.asarray(bboxes, dtype=np.float32),
        frs=np.asarray(frs, dtype=np.int32),
        fr_is=np.asarray(fr_is, dtype=np.int32),
        feats=np.stack(feats, axis=0).astype(np.float32),
        frames=np.asarray(bundle["frames"], dtype=np.int32),
    )
    meta = {
        "n_obs": bundle["n_obs"],
        "n_frames": len(bundle["frames"]),
        "feat": "hard_center_50",
        "cache": str(CACHE),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_obs(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    frames = [int(x) for x in z["frames"]]
    by_fr: list[list[dict]] = [[] for _ in frames]
    cams = z["cams"]
    for i in range(len(cams)):
        fr_i = int(z["fr_is"][i])
        by_fr[fr_i].append(
            {
                "cam": str(cams[i]),
                "bbox": [float(v) for v in z["bboxes"][i]],
                "feat": z["feats"][i].astype(np.float32),
                "fr": int(z["frs"][i]),
                "fr_i": fr_i,
            }
        )
    return {
        "frames": frames,
        "by_fr": by_fr,
        "n_obs": int(sum(len(x) for x in by_fr)),
    }


def _bbox_center(b) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in b]
    return x + 0.5 * w, y + 0.5 * h


def _link_score(a, b, iou_thr: float, max_center_px: float) -> float:
    """Higher is better; IoU preferred, else center proximity for stride-sparse cache."""
    ov = iou_xywh(a, b)
    if ov >= iou_thr:
        return 1.0 + ov
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    if dist <= max_center_px:
        return 0.5 * (1.0 - dist / max_center_px)
    return -1.0


def build_tracks(
    by_fr: list[list[dict]],
    iou_thr: float,
    max_center_px: float = 120.0,
) -> list[list[dict]]:
    """Greedy IoU / center tracks per camera across consecutive cache frames."""
    active: dict[str, list[dict]] = {}
    tracks: dict[int, list[dict]] = {}
    next_id = 1
    for obs_fr in by_fr:
        by_cam: dict[str, list[dict]] = defaultdict(list)
        for o in obs_fr:
            by_cam[o["cam"]].append(o)
        new_active: dict[str, list[dict]] = {}
        for cam, dets in by_cam.items():
            prev = active.get(cam) or []
            used_prev = set()
            used_det = set()
            pairs = []
            for pi, p in enumerate(prev):
                for di, d in enumerate(dets):
                    sc = _link_score(p["bbox"], d["bbox"], iou_thr, max_center_px)
                    if sc >= 0:
                        pairs.append((sc, pi, di))
            pairs.sort(reverse=True)
            cam_alive = []
            for sc, pi, di in pairs:
                if pi in used_prev or di in used_det:
                    continue
                used_prev.add(pi)
                used_det.add(di)
                tid = prev[pi]["tid"]
                d = dets[di]
                row = {**d, "tid": tid}
                tracks[tid].append(row)
                cam_alive.append({"tid": tid, "bbox": d["bbox"]})
            for di, d in enumerate(dets):
                if di in used_det:
                    continue
                tid = next_id
                next_id += 1
                row = {**d, "tid": tid}
                tracks[tid] = [row]
                cam_alive.append({"tid": tid, "bbox": d["bbox"]})
            new_active[cam] = cam_alive
        active = new_active
    return list(tracks.values())


def instant_label(feat: np.ndarray, cents: np.ndarray, radius: float) -> tuple[int, float]:
    return assign_feature(feat, cents, radius)


def label_track(track: list[dict], cents: np.ndarray, radius: float, cfg: dict) -> list[int]:
    kind = cfg["kind"]
    n = len(track)
    raw = [instant_label(o["feat"], cents, radius) for o in track]
    labs = [int(t) for t, _ in raw]
    confs = [float(c) for _, c in raw]
    out = [-1] * n

    if kind == "per_frame":
        return labs

    if kind == "sticky":
        thr = float(cfg["flip_conf"])
        cur = labs[0]
        out[0] = cur
        for i in range(1, n):
            if labs[i] < 0:
                out[i] = cur
                continue
            if cur < 0:
                cur = labs[i]
            elif labs[i] != cur and confs[i] >= thr:
                cur = labs[i]
            out[i] = cur if cur >= 0 else labs[i]
        return out

    if kind == "vote":
        k = int(cfg["k"])
        for i in range(n):
            window = [labs[j] for j in range(max(0, i - k + 1), i + 1) if labs[j] >= 0]
            if not window:
                out[i] = labs[i]
                continue
            out[i] = Counter(window).most_common(1)[0][0]
        return out

    if kind == "feat_median":
        k = int(cfg["k"])
        for i in range(n):
            lo = 0 if k <= 0 else max(0, i - k + 1)
            feats = [track[j]["feat"] for j in range(lo, i + 1)]
            med = tracklet_median_feature(feats)
            if med is None:
                out[i] = labs[i]
            else:
                out[i] = int(instant_label(med, cents, radius)[0])
        return out

    if kind == "feat_ema":
        alpha = float(cfg["alpha"])
        ema = track[0]["feat"].astype(np.float32).copy()
        out[0] = int(instant_label(ema, cents, radius)[0])
        for i in range(1, n):
            ema = (1.0 - alpha) * ema + alpha * track[i]["feat"].astype(np.float32)
            out[i] = int(instant_label(ema, cents, radius)[0])
        return out

    if kind == "vote_sticky":
        k = int(cfg["k"])
        thr = float(cfg["flip_conf"])
        cur = labs[0]
        out[0] = cur
        for i in range(1, n):
            window = [labs[j] for j in range(max(0, i - k + 1), i + 1) if labs[j] >= 0]
            voted = Counter(window).most_common(1)[0][0] if window else labs[i]
            if cur < 0:
                cur = voted
            elif voted != cur:
                # only flip if instant conf high AND vote agrees with new
                if confs[i] >= thr and voted == labs[i]:
                    cur = voted
            out[i] = cur if cur >= 0 else voted
        return out

    raise ValueError(kind)


def track_flips(labs: list[int]) -> tuple[int, int]:
    flips, edges = 0, 0
    for a, b in zip(labs, labs[1:]):
        if a < 0 or b < 0:
            continue
        edges += 1
        if a != b:
            flips += 1
    return flips, edges


def score_variant(
    name: str,
    cfg: dict,
    tracks: list[list[dict]],
    cents: np.ndarray,
    radius: float,
) -> dict:
    n0 = n1 = n_unsure = 0
    flips = edges = 0
    for tr in tracks:
        labs = label_track(tr, cents, radius, cfg)
        f, e = track_flips(labs)
        flips += f
        edges += e
        for t in labs:
            if t == 0:
                n0 += 1
            elif t == 1:
                n1 += 1
            else:
                n_unsure += 1
    n = n0 + n1
    if n <= 0:
        return {"variant": name, "error": "no_labels"}
    p0 = 100.0 * n0 / n
    p1 = 100.0 * n1 / n
    dev = abs(p0 - 50.0)
    bal = float(min(n0, n1) / max(n0, n1))
    flip_rate = float(flips / max(edges, 1))
    # closer to 50/50, fewer flips, more labels
    score = 100.0 * (
        0.50 * (1.0 - min(dev / 25.0, 1.0))
        + 0.30 * (1.0 - flip_rate)
        + 0.20 * bal
    )
    return {
        "variant": name,
        "n0": n0,
        "n1": n1,
        "n_unsure": n_unsure,
        "pct0": round(p0, 2),
        "pct1": round(p1, 2),
        "dev_from_50_pp": round(dev, 2),
        "balance": round(bal, 4),
        "flip_rate": round(flip_rate, 4),
        "n_tracks": len(tracks),
        "n_labeled": n,
        "score": round(score, 2),
    }


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.rebuild_obs or not OBS_PATH.is_file():
        bundle = extract_obs(args.max_frames)
        save_obs(bundle, OBS_PATH)
        print(f"saved obs n={bundle['n_obs']} → {OBS_PATH}", flush=True)
    else:
        bundle = load_obs(OBS_PATH)
        print(f"loaded obs n={bundle['n_obs']}", flush=True)

    tracks = build_tracks(bundle["by_fr"], args.iou)
    tracks = [t for t in tracks if len(t) >= 1]
    print(f"tracks={len(tracks)} mean_len={np.mean([len(t) for t in tracks]):.2f}", flush=True)

    # fit centroids from all observation features (hard_center_50)
    all_feats = [o["feat"] for fr in bundle["by_fr"] for o in fr]
    fit = fit_match_centroids(all_feats, min_crops=TEAM_MIN_CROPS)
    if fit is None:
        raise SystemExit("centroid fit failed")
    cents, radius = fit

    rows = []
    for name, cfg in VARIANTS:
        print(f"scoring {name}…", flush=True)
        rows.append(score_variant(name, cfg, tracks, cents, radius))
    rows_sorted = sorted(rows, key=lambda r: (r.get("dev_from_50_pp", 99), -r.get("score", 0)))
    # also rank by composite score
    by_score = sorted(rows, key=lambda r: -float(r.get("score", -1)))
    base = next(r for r in rows if r["variant"] == "per_frame")
    win = by_score[0]
    payload = {
        "n_obs": bundle["n_obs"],
        "n_tracks": len(tracks),
        "feat": "hard_center_50",
        "iou": args.iou,
        "ranking_by_50_50": rows_sorted,
        "ranking_by_score": by_score,
        "before_per_frame": base,
        "winner_by_score": win["variant"],
        "winner_by_50_50": rows_sorted[0]["variant"],
    }
    out_json = OUT / "ab_jersey_traj_variants.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    md = [
        "# Jersey trajectory / history A/B vs 50/50",
        "",
        f"n_obs={bundle['n_obs']} tracks={len(tracks)} feat=`hard_center_50` iou={args.iou}",
        "",
        "## Before (`per_frame`) vs best-by-score",
        "",
        f"| metric | before | after (`{win['variant']}`) |",
        "|---|---:|---:|",
        f"| Team0 / Team1 | {base['pct0']:.1f}% / {base['pct1']:.1f}% | {win['pct0']:.1f}% / {win['pct1']:.1f}% |",
        f"| Off 50/50 | {base['dev_from_50_pp']:.1f} pp | {win['dev_from_50_pp']:.1f} pp |",
        f"| Track flip rate | {base['flip_rate']:.3f} | {win['flip_rate']:.3f} |",
        f"| balance min/max | {base['balance']:.3f} | {win['balance']:.3f} |",
        "",
        "## Ranking (closest to 50/50 first)",
        "",
        "| rank | variant | pct0 | pct1 | off50 | flips | bal | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows_sorted, 1):
        md.append(
            f"| {i} | `{r['variant']}` | {r['pct0']:.1f} | {r['pct1']:.1f} | "
            f"{r['dev_from_50_pp']:.1f} | {r['flip_rate']:.3f} | {r['balance']:.3f} | {r['score']:.1f} |"
        )
    (OUT / "ab_jersey_traj_variants.md").write_text("\n".join(md) + "\n")
    print(json.dumps(by_score[:5], indent=2))
    print(f"closest_50={rows_sorted[0]['variant']} best_score={win['variant']} → {out_json}")


if __name__ == "__main__":
    main()
