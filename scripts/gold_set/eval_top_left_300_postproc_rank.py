#!/usr/bin/env python3
"""Rank ~30 ball post-process stacks on Match2 Top Left 300 gold.

Ground truth: data/processed/gold_sets/match2_4quad_top_left/gold/annotations.xml
Frames: review/frames/000.jpg…299.jpg (1920×1080). Never trains.
Writes reports/eval_match2_v10/top_left_300_postproc_rank/.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_poc_ball_metrics import match_preds, pr  # noqa: E402
from eval_v10_postprocess_ablation import detect_hflip_tta  # noqa: E402
from run_ball_postprocessing_test import (  # noqa: E402
    CKPT,
    make_tracker,
    variant_specs as postproc_specs,
)
from run_ball_sahi_hurt_test import variant_specs as hurt_specs  # noqa: E402
from run_ball_sahi_next_test import (  # noqa: E402
    detect_variant as next_detect,
    variant_specs as next_specs,
)
from src.perception.ball_prelabel import BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

GOLD_DIR = ROOT / "data/processed/gold_sets/match2_4quad_top_left"
OUT_DEFAULT = ROOT / "reports/eval_match2_v10/top_left_300_postproc_rank"
IOU = 0.5
N_FRAMES = 300


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="Optional technique ids")
    return p.parse_args()


def load_top_left_gt(xml_path: Path) -> dict[int, list]:
    """Frame → list of xywh. Prefer manual boxes when both sources exist."""
    root = ET.parse(xml_path).getroot()
    raw = defaultdict(list)
    for track in root.findall("track"):
        if (track.get("label") or "").lower() != "ball":
            continue
        source = track.get("source") or "auto"
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            xywh = (xtl, ytl, xbr - xtl, ybr - ytl)
            raw[frame].append((xywh, source))
    out = {}
    for frame, rows in raw.items():
        manuals = [b for b, s in rows if s == "manual"]
        autos = [b for b, s in rows if s != "manual"]
        out[frame] = manuals if manuals else autos
    return out


def load_frames(gold_dir: Path) -> list[Path]:
    frames_dir = gold_dir / "review" / "frames"
    paths = [frames_dir / f"{i:03d}.jpg" for i in range(N_FRAMES)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} frames e.g. {missing[0]}")
    return paths


def technique_specs() -> list[dict]:
    """Exactly 30: postproc 10 + SAHI-hurt 10 + SAHI-next 10."""
    rows = []
    for spec in postproc_specs():
        rows.append({**spec, "family": "postproc", "engine": "prelabel"})
    for spec in hurt_specs():
        rows.append({**spec, "family": "sahi_hurt", "engine": "prelabel"})
    for spec in next_specs():
        rows.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "why": spec["why"],
                "family": "sahi_next",
                "engine": "next",
                "kind": spec["kind"],
            }
        )
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 techniques, got {len(rows)}")
    return rows


def dets_to_preds(dets) -> list:
    return [(list(d.bbox), float(d.confidence)) for d in dets]


def run_prelabel_frames(model, frame_paths, spec) -> list:
    pre = BallPrelabeler(model, spec["cfg"], class_id=1)
    tracker = None
    if spec["mode"] == "bytetrack":
        tracker = make_tracker(spec.get("emit_gate", True), spec.get("match_thresh", 0.8))
    per_frame = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"failed to read {path}")
        mode = spec["mode"]
        if mode == "tta":
            dets = detect_hflip_tta(pre, frame)
        elif mode == "emit80":
            dets = [d for d in pre.detect_bgr(frame) if d.confidence >= 0.80]
        elif mode == "bytetrack":
            raw = pre.detect_bgr(frame)
            tracked = tracker.update(raw, frame)
            dets = [o.detection for o in tracked if o.detection.class_name == "ball"]
        else:
            dets = pre.detect_bgr(frame)
        per_frame.append(dets_to_preds(dets))
        if i % 50 == 0:
            print(f"  {spec['id']} {i}/{len(frame_paths)-1} n={len(dets)}", flush=True)
    return per_frame


def run_next_frames(model, frame_paths, spec) -> list:
    state = {}
    per_frame = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"failed to read {path}")
        dets = next_detect(model, frame, spec["kind"], state)
        per_frame.append(dets_to_preds(dets))
        if i % 50 == 0:
            print(f"  {spec['id']} {i}/{len(frame_paths)-1} n={len(dets)}", flush=True)
    return per_frame


def score_at(gt_by_frame: dict, preds_by_frame: list, thr: float) -> dict:
    tp = fp = fn = 0
    n_gt = 0
    for i, preds in enumerate(preds_by_frame):
        gt = gt_by_frame.get(i, [])
        n_gt += len(gt)
        kept = [(b, c) for b, c in preds if c >= thr]
        tpi, fpi, fni = match_preds(gt, kept)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "precision": p,
        "recall": r,
        "f1": f1,
        "P_emit": p if (tp + fp) > 0 else None,
        "n_emitted": tp + fp,
    }


def rank_rows(rows: list) -> list:
    """Best first: F1@0.3, then recall@0.3, then precision@0.3."""
    return sorted(
        rows,
        key=lambda r: (
            -r["at_0_3"]["f1"],
            -r["at_0_3"]["recall"],
            -r["at_0_3"]["precision"],
            r["id"],
        ),
    )


def write_report(out: Path, payload: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ranking.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# {payload['title']}",
        "",
        f"Gold: `{payload['gold_xml']}` · {payload['n_frames']} frames · "
        f"{payload['n_gt_boxes']} GT boxes · IoU≥{IOU}",
        "",
        "Ranked by **F1 @ conf≥0.3** (then recall, then precision). "
        "Also report product emit @ conf≥0.8.",
        "",
        "| Rank | id | family | F1@0.3 | R@0.3 | P@0.3 | R@0.8 | P_emit@0.8 | tp/fp/fn@0.3 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(payload["ranked"], 1):
        a, b = row["at_0_3"], row["at_0_8"]
        p80 = "—" if b["P_emit"] is None else f"{b['P_emit']:.3f}"
        lines.append(
            f"| {i} | `{row['id']}` | {row['family']} | {a['f1']:.3f} | "
            f"{a['recall']:.3f} | {a['precision']:.3f} | {b['recall']:.3f} | "
            f"{p80} | {a['tp']}/{a['fp']}/{a['fn']} |"
        )
    lines += ["", "## Top 5", ""]
    for i, row in enumerate(payload["ranked"][:5], 1):
        lines.append(
            f"{i}. **{row['id']}** — {row['title']}  \n"
            f"   {row['why']}  \n"
            f"   F1@0.3={row['at_0_3']['f1']:.3f} R={row['at_0_3']['recall']:.3f} "
            f"P={row['at_0_3']['precision']:.3f}"
        )
    md_path = out / "ranking.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    args = parse_args()
    gold_dir = args.gold_dir
    xml_path = gold_dir / "gold" / "annotations.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"missing gold XML {xml_path}")
    if not args.ball_checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint {args.ball_checkpoint}")

    gt = load_top_left_gt(xml_path)
    frame_paths = load_frames(gold_dir)
    n_gt = sum(len(v) for v in gt.values())
    print(f"gt_frames={len(gt)} gt_boxes={n_gt} frames={len(frame_paths)}", flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "partial.json"
    done = {}
    if args.resume and partial_path.is_file():
        done = {r["id"]: r for r in json.loads(partial_path.read_text())["rows"]}
        print(f"resume with {len(done)} done", flush=True)

    specs = technique_specs()
    if args.only:
        want = set(args.only)
        specs = [s for s in specs if s["id"] in want]

    model = load_ball_model(str(args.ball_checkpoint))
    rows = list(done.values())
    for spec in specs:
        if spec["id"] in done:
            print(f"skip {spec['id']}", flush=True)
            continue
        print(f"run {spec['id']} ({spec['family']})", flush=True)
        if spec["engine"] == "next":
            preds = run_next_frames(model, frame_paths, spec)
        else:
            preds = run_prelabel_frames(model, frame_paths, spec)
        row = {
            "id": spec["id"],
            "title": spec["title"],
            "why": spec["why"],
            "family": spec["family"],
            "at_0_3": score_at(gt, preds, 0.30),
            "at_0_8": score_at(gt, preds, 0.80),
        }
        rows = [r for r in rows if r["id"] != spec["id"]] + [row]
        partial_path.write_text(
            json.dumps({"rows": rows}, indent=2), encoding="utf-8"
        )
        a = row["at_0_3"]
        print(
            f"DONE {spec['id']} F1={a['f1']:.3f} R={a['recall']:.3f} "
            f"P={a['precision']:.3f} tp/fp/fn={a['tp']}/{a['fp']}/{a['fn']}",
            flush=True,
        )

    ranked = rank_rows(rows)
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    payload = {
        "title": "Match2 Top Left 300 — post-process ranking",
        "gold_dir": str(gold_dir),
        "gold_xml": str(xml_path),
        "checkpoint": str(args.ball_checkpoint),
        "n_frames": len(frame_paths),
        "n_gt_boxes": n_gt,
        "n_gt_frames": len(gt),
        "iou": IOU,
        "rank_key": "f1_at_conf_0.3",
        "ranked": ranked,
    }
    md = write_report(out, payload)
    print(f"wrote {md}", flush=True)
    print("TOP 5:", flush=True)
    for row in ranked[:5]:
        print(
            f"  #{row['rank']} {row['id']} F1={row['at_0_3']['f1']:.3f} "
            f"R={row['at_0_3']['recall']:.3f} P={row['at_0_3']['precision']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
