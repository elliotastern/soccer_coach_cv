#!/usr/bin/env python3
"""Rebuild strip (+ optional holdout) det caches for a checkpoint, then score vs promoted v13.

Product path: raw detect thr 0.20 (prelabel thr 0.10 cache) + foot undistort fuse.
Does not lower EMIT_CONF / widen agree.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

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
)
from score_match3_ball_m1 import score_proxy_packs, score_strip  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/ab_v14_residual_vs_v13.json"
CACHE_DIR = ROOT / "reports/eval_match3/improve_eng_loop/v14_residual_det_cache"
STRIPS = [
    ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json",
    ROOT / "data/processed/gold_sets/match3_quad_p8_87/labels.json",
]
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
P_EMIT_GATE = 0.80


def detect_strip_cache(labels_path: Path, model, out_cache: Path, stride: int) -> Path:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    focus = labels["focus_cam"]
    stem = labels["stem"]
    src_dir = ROOT / "reports/eval_match3/quad_pitchmap_gallery/source"
    n = int(labels["n_frames"])
    cfg = BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE)
    pre = BallPrelabeler(model, cfg, class_id=1)
    caps = {}
    for cam in CAMS:
        path = src_dir / f"{stem}_{cam}.mp4"
        if not path.is_file():
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    if focus not in caps:
        raise RuntimeError(f"missing focus video {focus} for {stem}")
    out = {cam: [] for cam in caps}
    try:
        for i in range(n):
            for cam, cap in caps.items():
                frame = read_resized(cap)
                if frame is None:
                    raise RuntimeError(f"short video {cam} @ {i}")
                if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
                    frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
                if i % stride == 0:
                    out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
                else:
                    out[cam].append([])
    finally:
        for cap in caps.values():
            cap.release()
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_dump_n(out_cache, out, n)
    return out_cache


def score_with_cache(labels_path: Path, cache_path: Path) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    tmp = labels_path.with_suffix(".ab_tmp.json")
    labels = dict(labels)
    try:
        rel = str(cache_path.relative_to(ROOT))
    except ValueError:
        rel = str(cache_path)
    labels["det_cache"] = rel
    tmp.write_text(json.dumps(labels), encoding="utf-8")
    try:
        return score_strip(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "models/v14_residual_snaps/post_train/checkpoint.pth",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=ROOT / "reports/eval_match3/improve_eng_loop/ab_v13_residual_vs_v12.json",
    )
    p.add_argument("--stride", type=int, default=None, help="override strip stride")
    args = p.parse_args()
    if not args.ckpt.is_file():
        raise SystemExit(f"missing {args.ckpt}")

    model = load_ball_model(str(args.ckpt))
    strips_v14 = {}
    for lab in STRIPS:
        if not lab.is_file():
            continue
        labels = json.loads(lab.read_text(encoding="utf-8"))
        stride = args.stride
        if stride is None:
            from score_match3_ball_m1 import infer_cache_stride
            from eval_match2_top_left_multicam_baseline import cache_load

            stride = infer_cache_stride(cache_load(ROOT / labels["det_cache"]))
        cache_path = CACHE_DIR / f"det_cache_{labels['stem']}_v14_thr010.json"
        print(f"detect {labels['pack']} stride={stride} → {cache_path.name}")
        detect_strip_cache(lab, model, cache_path, stride)
        strips_v14[labels["pack"]] = score_with_cache(lab, cache_path)

    baseline = {}
    if args.baseline_json.is_file():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    kills = []
    for name, row in strips_v14.items():
        pe = row.get("P_emit")
        if pe is not None and pe < P_EMIT_GATE:
            kills.append(f"{name} P_emit={pe}")

    base_strips = {}
    if isinstance(baseline, dict):
        base_strips = baseline.get("strips_v13") or baseline.get("strips") or {}
    comparison = {}
    for name, row in strips_v14.items():
        base = base_strips.get(name) or {}
        br, cr = base.get("clear_ball_R"), row.get("clear_ball_R")
        comparison[name] = {
            "v13_clear_R": br,
            "v14_clear_R": cr,
            "d_clear_R": None if br is None or cr is None else round(float(cr) - float(br), 3),
        }
        if br is not None and cr is not None and float(cr) + 0.001 < float(br):
            kills.append(f"{name} clear_R regress {br}->{cr}")
    payload = {
        "v14_ckpt": str(args.ckpt),
        "strips_v14": strips_v14,
        "baseline_strips_v13": base_strips,
        "comparison_vs_v13": comparison,
        "p_emit_kills": kills,
        "promote_candidate": len(kills) == 0,
        "decision": "promote" if not kills else "no_promote",
        "cache_dir": str(CACHE_DIR.relative_to(ROOT)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, row in strips_v14.items():
        print(
            f"{name}: P_emit={row.get('P_emit')} clear_R={row.get('clear_ball_R')} "
            f"pass_P={row.get('poc_pass_P_emit')} pass_R={row.get('poc_pass_clear_R')}"
        )
    print(f"promote_candidate={payload['promote_candidate']} wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
