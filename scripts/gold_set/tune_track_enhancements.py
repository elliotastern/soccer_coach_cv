#!/usr/bin/env python3
"""Tune Kalman/ByteTrack on Match2 train100, then score held-out gold.

Never trains the detector. Never uses gold for picking. Gold XML is source of truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import (
    det_tuple,
    load_gold_items,
    make_prelabeler,
    match_tp,
    pr,
    read_n_frames_from,
)
from src.perception.rfdetr_local import load_ball_model
from src.perception.tracker import Tracker
from src.state.types import Detection

CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
TRAIN_PACK = ROOT / "data/processed/gold_sets/match2_train_label100"
TRAIN_COCO = ROOT / "data/processed/gold_sets/match2_train_test/train/_annotations.coco.json"
GOLD_DIR = ROOT / "data/processed/gold_sets/match2_gold_frames"
OUT = ROOT / "reports/eval_match2_v10/track_tune.json"
WARMUP = 10
MIN_THR = 0.30
SCORE_THR = 0.30
PREC_FLOOR = 0.90


def load_train_items():
    names = {
        im["file_name"]
        for im in json.loads(TRAIN_COCO.read_text())["images"]
    }
    items = load_gold_items(TRAIN_PACK)
    out = [it for it in items if it["image"] in names and it["gt"]]
    if len(out) != 87:
        raise RuntimeError(f"expected 87 train items with GT, got {len(out)}")
    return out


def dets_to_dicts(dets):
    return [
        {
            "bbox": list(d.bbox),
            "confidence": float(d.confidence),
            "class_id": int(d.class_id),
            "class_name": d.class_name,
        }
        for d in dets
    ]


def dicts_to_dets(rows):
    return [
        Detection(
            class_id=int(r["class_id"]),
            confidence=float(r["confidence"]),
            bbox=tuple(r["bbox"]),
            class_name=r["class_name"],
        )
        for r in rows
    ]


def cache_item(model, item, use_kalman: bool):
    pre = make_prelabeler(model, MIN_THR, use_kalman)
    start = max(0, item["frame_idx"] - WARMUP)
    n_warm = item["frame_idx"] - start
    warm, _ = read_n_frames_from(item["video_path"], start, n_warm)
    stream = []
    for frame in warm:
        stream.append(dets_to_dicts(pre.detect_bgr(frame)))
    gold = cv2.imread(str(item["image_path"]))
    if gold is None:
        raise RuntimeError(f"missing jpeg {item['image_path']}")
    stream.append(dets_to_dicts(pre.detect_bgr(gold)))
    h, w = gold.shape[:2]
    return {"stream": stream, "hw": [h, w]}


def cache_split(model, items, use_kalman: bool, tag: str):
    rows = []
    for i, item in enumerate(items):
        cached = cache_item(model, item, use_kalman)
        rows.append(cached)
        n = len(cached["stream"][-1])
        top = max((d["confidence"] for d in cached["stream"][-1]), default=0.0)
        print(
            f"{tag} {i:02d}/{len(items)-1} {item['image']} "
            f"n={n} top={top:.3f} kalman={use_kalman}",
            flush=True,
        )
    return rows


def make_tracker(cfg: dict):
    return Tracker(
        track_thresh=cfg["track_thresh"],
        match_thresh=cfg["match_thresh"],
        emit_thresh=cfg["emit_thresh"],
        ema_alpha=cfg["ema_alpha"],
        apply_emit_gate=cfg["emit_gate"],
        match_px=cfg.get("match_px"),
        frame_rate=30,
    )


def replay_last(stream, hw, cfg: dict):
    if cfg["kind"] == "detector":
        dets = dicts_to_dets(stream[-1])
        preds = [det_tuple(d) for d in dets]
        preds.sort(key=lambda x: -x[1])
        return preds
    tracker = make_tracker(cfg)
    frame = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
    published = []
    for rows in stream:
        dets = dicts_to_dets(rows)
        published = tracker.update(dets, frame)
    preds = [det_tuple(o.detection) for o in published]
    preds.sort(key=lambda x: -x[1])
    return preds


def score_cached(items, cache, cfg: dict):
    tp = fp = fn = 0
    for item, row in zip(items, cache):
        preds = replay_last(row["stream"], row["hw"], cfg)
        keep = [p for p in preds if p[1] >= SCORE_THR]
        tpi, fpi, fni = match_tp(item["gt"], keep)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def configs():
    return [
        {
            "name": "detector_only",
            "kind": "detector",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.8,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "legacy_bt_px80_iou08",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.8,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": 80.0,
        },
        {
            "name": "bt_scaled_px_iou08",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.8,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "bt_scaled_px_iou03",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.3,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "bt_scaled_px_iou05",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.5,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "bt_iou03_track05",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.05,
            "match_thresh": 0.3,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "bt_iou03_emit80",
            "kind": "track",
            "kalman": False,
            "track_thresh": 0.10,
            "match_thresh": 0.3,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": True,
            "match_px": None,
        },
        {
            "name": "kalman_bt_iou03",
            "kind": "track",
            "kalman": True,
            "track_thresh": 0.10,
            "match_thresh": 0.3,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
        {
            "name": "kalman_only_as_detect",
            "kind": "detector",
            "kalman": True,
            "track_thresh": 0.10,
            "match_thresh": 0.8,
            "emit_thresh": 0.80,
            "ema_alpha": 0.3,
            "emit_gate": False,
            "match_px": None,
        },
    ]


def pick_winner(rows, detector_prec: float):
    floor = min(PREC_FLOOR, detector_prec - 0.02)
    eligible = [r for r in rows if r["train"]["precision"] >= floor]
    if not eligible:
        eligible = rows
    eligible = sorted(eligible, key=lambda r: (-r["train"]["recall"], -r["train"]["precision"]))
    return eligible[0]


def main() -> int:
    train_items = load_train_items()
    gold_items = load_gold_items(GOLD_DIR)
    print(f"train items: {len(train_items)} gold items: {len(gold_items)}", flush=True)
    model = load_ball_model(str(CKPT))
    train_raw = cache_split(model, train_items, False, "train_raw")
    train_kal = cache_split(model, train_items, True, "train_kal")
    gold_raw = cache_split(model, gold_items, False, "gold_raw")
    gold_kal = cache_split(model, gold_items, True, "gold_kal")

    train_rows = []
    for cfg in configs():
        tcache = train_kal if cfg["kalman"] else train_raw
        gcache = gold_kal if cfg["kalman"] else gold_raw
        train_m = score_cached(train_items, tcache, cfg)
        print(f"TRAIN {cfg['name']}: R={train_m['recall']:.3f} P={train_m['precision']:.3f} "
              f"tp/fp/fn={train_m['tp']}/{train_m['fp']}/{train_m['fn']}", flush=True)
        train_rows.append({"name": cfg["name"], "cfg": cfg, "train": train_m, "gold": None})

    detector = next(r for r in train_rows if r["name"] == "detector_only")
    winner = pick_winner(train_rows, detector["train"]["precision"])
    print(f"PICKED on train: {winner['name']}", flush=True)

    gold_rows = []
    for row in train_rows:
        cfg = row["cfg"]
        gcache = gold_kal if cfg["kalman"] else gold_raw
        gold_m = score_cached(gold_items, gcache, cfg)
        row["gold"] = gold_m
        gold_rows.append(row)
        print(f"GOLD {cfg['name']}: R={gold_m['recall']:.3f} P={gold_m['precision']:.3f} "
              f"tp/fp/fn={gold_m['tp']}/{gold_m['fp']}/{gold_m['fn']}", flush=True)

    report = {
        "checkpoint": "models/v10_snaps/post_train/checkpoint.pth",
        "tune_on": "match2_train_label100 train split (87), not gold",
        "score_thr": SCORE_THR,
        "warmup": WARMUP,
        "picked": winner["name"],
        "detector_train": detector["train"],
        "winner_train": winner["train"],
        "winner_gold": next(r["gold"] for r in gold_rows if r["name"] == winner["name"]),
        "detector_gold": next(r["gold"] for r in gold_rows if r["name"] == "detector_only"),
        "all": [
            {"name": r["name"], "cfg": r["cfg"], "train": r["train"], "gold": r["gold"]}
            for r in gold_rows
        ],
        "read": (
            "Pick used train recall @0.3 with precision floor. "
            "Gold numbers are locked after pick. "
            "Helps gold only if winner_gold.recall > detector_gold.recall "
            "without collapsing precision."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
