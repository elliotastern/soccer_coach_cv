#!/usr/bin/env python3
"""Rebuild Match3 holdout det caches with product ball checkpoint; score proxy R.

Writes to pitchmap_gallery_holdout/det_cache_v13/ (does not overwrite v12 caches).
Compares clear_ball_proxy_R vs existing det_cache/ baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_4quad_multicam_survey import (  # noqa: E402
    DETECT_H,
    DETECT_W,
    SIZE,
    cache_dump_n,
    dets_to_rows,
    read_resized,
    source_path,
)
from score_match3_ball_m1 import score_cache  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

HOLDOUT = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/ab_v13_holdout_score.json"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
STRIDE = 2
HOLDOUT_GATE = 0.884


def product_ckpt() -> Path:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    rel = cfg["detection"]["ball_checkpoint"]
    return ROOT / rel


def detect_stem(model, src: Path, stem: str, out_cache: Path, stride: int) -> dict:
    first = source_path(src, stem, CAMS[0])
    cap0 = cv2.VideoCapture(str(first))
    n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    if n <= 0:
        raise RuntimeError(f"empty {first}")
    cfg = BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE)
    pre = BallPrelabeler(model, cfg, class_id=1)
    caps = {}
    for cam in CAMS:
        path = source_path(src, stem, cam)
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    out = {cam: [] for cam in CAMS}
    try:
        for i in range(n):
            for cam, cap in caps.items():
                frame = read_resized(cap)
                if frame is None:
                    n = i
                    break
                if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
                    frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
                if i % stride == 0:
                    out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
                else:
                    out[cam].append([])
            else:
                continue
            break
    finally:
        for cap in caps.values():
            cap.release()
    for cam in CAMS:
        out[cam] = out[cam][:n]
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_dump_n(out_cache, out, n)
    try:
        cache_s = str(out_cache.resolve().relative_to(ROOT))
    except ValueError:
        cache_s = str(out_cache)
    return {"stem": stem, "n_frames": n, "cache": cache_s}


def score_dir(cache_dir: Path) -> dict:
    rows = [score_cache(p) for p in sorted(cache_dir.glob("det_cache_*_thr010.json"))]
    if not rows:
        return {"error": f"no caches in {cache_dir}"}
    clear = sum(r.get("n_clear_proxy", 0) for r in rows)
    emit = sum(r.get("n_clear_emit", 0) for r in rows)
    # score_cache field names from score_match3_ball_m1
    tot_clear = sum(int(r.get("clear_proxy") or r.get("n_clear_proxy") or 0) for r in rows)
    # Fall back: use clear_ball_proxy_R weighted
    if "clear_ball_proxy_R" in rows[0]:
        # aggregate via totals if present
        pass
    return {
        "n_caches": len(rows),
        "rows": rows,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="override checkpoint (default: configs/default.yaml ball_checkpoint)",
    )
    args = p.parse_args()
    ckpt = args.ckpt or product_ckpt()
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")

    src = HOLDOUT / "source"
    base_dir = HOLDOUT / "det_cache"
    v13_dir = HOLDOUT / "det_cache_v13"
    stems = [e["stem"] for e in json.loads((HOLDOUT / "manifest.json").read_text())]

    built = []
    if not args.skip_detect:
        model = load_ball_model(str(ckpt))
        for stem in stems:
            out_cache = v13_dir / f"det_cache_{stem}_thr010.json"
            print(f"detect {stem} → {out_cache.name}", flush=True)
            built.append(detect_stem(model, src, stem, out_cache, args.stride))

    def pack_score(cache_dir: Path) -> dict:
        rows = []
        tot_clear = tot_emit = 0
        for path in sorted(cache_dir.glob("det_cache_*_thr010.json")):
            row = score_cache(path)
            rows.append({"cache": path.name, **{k: row[k] for k in row if k != "per_frame"}})
            # score_cache returns clear_ball_proxy_R and counts
            c = int(row.get("n_clear") or row.get("clear") or 0)
            e = int(row.get("n_clear_emit") or row.get("clear_emit") or 0)
            # Prefer explicit fields from score_match3_ball_m1.score_cache
            if "clear_ball_proxy_R" in row and "n_clear_proxy" not in row:
                # inspect keys once
                pass
            tot_clear += int(row.get("n_clear_proxy") or 0)
            tot_emit += int(row.get("n_clear_emit") or 0)
        # If n_clear_proxy missing, derive from proxy R * something
        if tot_clear == 0 and rows:
            # read raw score_cache keys from first
            sample = score_cache(sorted(cache_dir.glob("det_cache_*_thr010.json"))[0])
            return {"sample_keys": list(sample.keys()), "rows": rows}
        r = None if tot_clear == 0 else round(tot_emit / tot_clear, 3)
        return {
            "n_caches": len(rows),
            "n_clear_proxy": tot_clear,
            "n_clear_emit": tot_emit,
            "clear_ball_proxy_R": r,
            "rows": rows,
        }

    # Probe score_cache schema
    probe = score_cache(sorted(base_dir.glob("det_cache_*_thr010.json"))[0])

    def pack_score2(cache_dir: Path) -> dict:
        paths = sorted(cache_dir.glob("det_cache_*_thr010.json"))
        rows = [score_cache(p) for p in paths]
        clear = sum(int(r.get("clear_frames") or 0) for r in rows)
        emit = sum(int(r.get("clear_emit") or 0) for r in rows)
        return {
            "n_caches": len(rows),
            "clear_frames": clear,
            "clear_emit": emit,
            "clear_ball_proxy_R": None if clear == 0 else round(emit / clear, 3),
            "agree_among_emit": None
            if sum(int(r.get("emit") or 0) for r in rows) == 0
            else round(
                sum(int(r.get("agree") or 0) for r in rows)
                / sum(int(r.get("emit") or 0) for r in rows),
                3,
            ),
            "rows": rows,
        }

    baseline = pack_score2(base_dir)
    cand = pack_score2(v13_dir)
    base_r = baseline.get("clear_ball_proxy_R")
    cand_r = cand.get("clear_ball_proxy_R")
    payload = {
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "stride": args.stride,
        "holdout_gate": HOLDOUT_GATE,
        "baseline_v12_caches": baseline,
        "v13_caches": cand,
        "built": built,
        "d_clear_R": None
        if base_r is None or cand_r is None
        else round(float(cand_r) - float(base_r), 3),
        "passes_gate": cand_r is not None and float(cand_r) >= HOLDOUT_GATE - 0.001,
        "improved_vs_baseline": cand_r is not None
        and base_r is not None
        and float(cand_r) >= float(base_r) - 0.001,
        "probe_score_cache_keys": sorted(probe.keys()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"baseline_R={base_r} v13_R={cand_r} d={payload['d_clear_R']} "
        f"gate={payload['passes_gate']} improved={payload['improved_vs_baseline']}"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
