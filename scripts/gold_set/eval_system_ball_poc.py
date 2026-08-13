#!/usr/bin/env python3
"""System ball PoC: SAHI + Kalman + ByteTrack emit gate + multi-cam selection.

Scores product-style publishes (emit EMA/instant >= emit_thresh) on Match1 20s
multicam pack. Labels are provisional auto-seed by default—report says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model
from src.perception.tracker import Tracker
from src.state.types import Detection

CAMS = ["cam8", "cam9", "cam11", "cam13"]
IOU_THR = 0.5
DEFAULT_PACK = ROOT / "data/processed/multicam_20s_match1"
DEFAULT_CKPT = ROOT / "models/v8_snaps/post_train/checkpoint.pth"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--ball-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--min-thr", type=float, default=0.30)
    p.add_argument("--emit-thresh", type=float, default=0.80)
    p.add_argument("--track-thresh", type=float, default=0.10)
    p.add_argument("--ema-alpha", type=float, default=0.3)
    p.add_argument("--use-sahi", action="store_true", default=True)
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--use-kalman", action="store_true", default=True)
    p.add_argument("--no-kalman", action="store_true")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth labeled timestamp")
    p.add_argument("--single-cam", type=str, default="cam11")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "system_poc_v8_multicam20s.json",
    )
    return p.parse_args()


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def as_xywh(b) -> list[float]:
    if isinstance(b, dict):
        if "bbox" in b:
            return [float(x) for x in b["bbox"]]
        return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]
    return [float(x) for x in b]


def gt_boxes(item: dict) -> list[list[float]] | None:
    if item.get("empty") is True:
        return []
    if item.get("gt_balls") is None:
        return None
    return [as_xywh(b) for b in item["gt_balls"]]


def match_tp(gt, preds):
    matched = set()
    tp = 0
    for box, _, _ in preds:
        best_j, best = -1, 0.0
        for j, g in enumerate(gt):
            if j in matched:
                continue
            v = iou_xywh(box, g)
            if v > best:
                best, best_j = v, j
        if best >= IOU_THR and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gt) - len(matched)
    return tp, fp, fn


def pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def det_tuple(d: Detection):
    box = list(d.bbox)
    conf = float(d.confidence)
    side = min(box[2], box[3])
    return (box, conf, side)


def filter_thr(preds, thr: float):
    return [p for p in preds if p[1] >= thr]


def make_prelabeler(model, min_thr: float, use_sahi: bool, use_kalman: bool):
    return BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=min_thr,
            use_sahi=use_sahi,
            sahi_fallback_only=True,
            sahi_recover_only=True,
            use_size_filter=True,
            topk=2,
            use_kalman=use_kalman,
            min_side=4,
            max_side=240,
        ),
        class_id=1,
    )


def resolve_image(pack_dir: Path, rel: str) -> Path:
    # labels store images/cam8/t000_00.00s.jpg relative to eval_pack
    p = pack_dir / "eval_pack" / rel
    if p.is_file():
        return p
    # fall back to frames/
    name = Path(rel).name
    cam = Path(rel).parts[-2] if len(Path(rel).parts) >= 2 else "cam8"
    alt = pack_dir / "frames" / cam / name
    return alt


def run_cam_stream(pre, tracker: Tracker, paths: list[Path]):
    """Return list of dicts raw_preds + emit_preds per path (xywh tuples)."""
    rows = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"missing image {path}")
        dets = pre.detect_bgr(frame)
        raw = [det_tuple(d) for d in dets]
        raw.sort(key=lambda x: -x[1])
        tracked = tracker.update(dets, frame)
        emits = []
        for obj in tracked:
            if obj.detection.class_name != "ball":
                continue
            emits.append(det_tuple(obj.detection))
        emits.sort(key=lambda x: -x[1])
        rows.append({"raw": raw, "emit": emits})
    return rows


def pick_selected(cam_preds: dict, thr: float, mode: str):
    """cam_preds: cam -> list[(box,conf,side)]. Return (cam, pred) or (None,None)."""
    cands = []
    for cam, preds in cam_preds.items():
        preds = filter_thr(preds, thr)
        if not preds:
            continue
        box, conf, side = preds[0]
        if mode == "max_conf":
            score = conf
        elif mode == "size_weighted":
            score = conf * side
        else:
            score = conf
        cands.append((score, cam, preds[0]))
    if not cands:
        return None, None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], cands[0][2]


def score_selection(timestamps, cam_preds_by_ts, thr: float, mode: str, key: str):
    """key is 'raw' or 'emit' — which pred list to use per cam."""
    tp = fp = fn = 0
    n_emitted = 0
    for i, ts in enumerate(timestamps):
        has_ball = any(len(gt_boxes(ts["cams"][c]) or []) > 0 for c in CAMS)
        pred_map = {c: cam_preds_by_ts[i][c][key] for c in CAMS}
        cam, pred = pick_selected(pred_map, thr, mode)
        if pred is None:
            if has_ball:
                fn += 1
            continue
        n_emitted += 1
        gt = gt_boxes(ts["cams"][cam]) or []
        tpi, fpi, _ = match_tp(gt, [pred])
        if tpi:
            tp += 1
        else:
            fp += 1
            if has_ball:
                fn += 1
    p, r = pr(tp, fp, fn)
    p_emit = p if n_emitted > 0 else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "P_emit": p_emit,
        "n_emitted": n_emitted,
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "hollow": bool(p_emit is not None and p_emit >= 0.80 and n_emitted < 5),
    }


def score_oracle(timestamps, cam_preds_by_ts, thr: float, key: str):
    tp = fp = fn = 0
    n_emitted = 0
    for i, ts in enumerate(timestamps):
        has_ball = any(len(gt_boxes(ts["cams"][c]) or []) > 0 for c in CAMS)
        any_tp = False
        any_pred = False
        for cam in CAMS:
            preds = filter_thr(cam_preds_by_ts[i][cam][key], thr)
            if preds:
                any_pred = True
                n_emitted += 1
            tpi, _, _ = match_tp(gt_boxes(ts["cams"][cam]) or [], preds)
            if tpi:
                any_tp = True
        if any_tp:
            tp += 1
        elif has_ball:
            fn += 1
        elif any_pred:
            fp += 1
    p, r = pr(tp, fp, fn)
    p_emit = p if (tp + fp) > 0 else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "P_emit": p_emit,
        "n_emitted": n_emitted,
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "hollow": bool(p_emit is not None and p_emit >= 0.80 and n_emitted < 5),
    }


def score_single_cam_stream(timestamps, cam_preds_by_ts, cam: str, thr: float, key: str):
    tp = fp = fn = 0
    n_emitted = 0
    confs = []
    for i, ts in enumerate(timestamps):
        gt = gt_boxes(ts["cams"][cam])
        if gt is None:
            continue
        preds = filter_thr(cam_preds_by_ts[i][cam][key], thr)
        if preds:
            n_emitted += 1
            confs.append(preds[0][1])
        tpi, fpi, fni = match_tp(gt, preds)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    p_emit = p if n_emitted > 0 else None
    return {
        "cam": cam,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "P_emit": p_emit,
        "n_emitted": n_emitted,
        "mean_emit_conf": (sum(confs) / len(confs)) if confs else None,
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "hollow": bool(p_emit is not None and p_emit >= 0.80 and n_emitted < 5),
    }


def write_md(path: Path, report: dict):
    lines = [
        "# System ball PoC — multicam 20s (v8)",
        "",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"Pack: `{report['pack']}`  ",
        f"Labels: **{report['label_note']}**  ",
        f"Stack: SAHI={report['use_sahi']} Kalman={report['use_kalman']} "
        f"detect_thr={report['min_thr']} emit_thresh={report['emit_thresh']} "
        f"stride={report['stride']}  ",
        f"Timestamps scored: {report['n_timestamps']}",
        "",
        "## Primary (track emit + multi-cam selection) @ conf ≥ emit_thresh",
        "",
        "| method | P_emit | n_emitted | R | tp/fp/fn | hollow? |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in report["primary"].items():
        pe = row["P_emit"]
        pe_s = f"{pe:.3f}" if pe is not None else "none"
        lines.append(
            f"| {name} | {pe_s} | {row['n_emitted']} | {row['recall']:.3f} | "
            f"{row['tp']}/{row['fp']}/{row['fn']} | {row['hollow']} |"
        )
    lines += [
        "",
        "## Ablations @ conf ≥ emit_thresh",
        "",
        "| stack | P_emit | n_emitted | R | tp/fp/fn | hollow? |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in report["ablations"].items():
        pe = row["P_emit"]
        pe_s = f"{pe:.3f}" if pe is not None else "none"
        lines.append(
            f"| {name} | {pe_s} | {row['n_emitted']} | {row['recall']:.3f} | "
            f"{row['tp']}/{row['fp']}/{row['fn']} | {row['hollow']} |"
        )
    sc = report.get("single_cam") or {}
    lines += [
        "",
        f"## Single-cam temporal ({sc.get('cam', '?')}) track emit @ emit_thresh",
        "",
        f"- n_emitted: **{sc.get('n_emitted')}**",
        f"- P_emit: {sc.get('P_emit')}",
        f"- mean conf when emit: {sc.get('mean_emit_conf')}",
        f"- R={sc.get('recall')} tp/fp/fn={sc.get('tp')}/{sc.get('fp')}/{sc.get('fn')}",
        "",
        "## Read",
        "",
        report.get(
            "read",
            "Compare n_emitted vs detector-only Gold PoC (often 0–1 at 0.8). "
            "Hollow = P_emit≥0.8 with n_emitted<5.",
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    use_sahi = bool(args.use_sahi) and not args.no_sahi
    use_kalman = bool(args.use_kalman) and not args.no_kalman
    labels_path = args.labels or (args.pack_dir / "eval_pack" / "labels.json")
    data = json.loads(labels_path.read_text())
    all_ts = data["timestamps"]
    # only timestamps with ALL cams labeled
    timestamps = []
    for ts in all_ts:
        if all(gt_boxes(ts["cams"][c]) is not None for c in CAMS):
            timestamps.append(ts)
    if args.stride > 1:
        timestamps = timestamps[:: args.stride]
    if not timestamps:
        raise SystemExit("no fully labeled timestamps in labels.json")

    print(f"checkpoint: {args.ball_checkpoint}")
    print(f"timestamps: {len(timestamps)} (stride={args.stride})")
    print(f"sahi={use_sahi} kalman={use_kalman} emit={args.emit_thresh}")
    model = load_ball_model(str(args.ball_checkpoint))

    # Stream each cam once (time order) so Kalman + tracker see continuous video.
    cam_preds_by_ts = [
        {c: {"raw": [], "emit": []} for c in CAMS} for _ in timestamps
    ]
    for cam in CAMS:
        pre = make_prelabeler(model, args.min_thr, use_sahi, use_kalman)
        tracker = Tracker(
            track_thresh=args.track_thresh,
            emit_thresh=args.emit_thresh,
            ema_alpha=args.ema_alpha,
            apply_emit_gate=True,
            frame_rate=30,
        )
        paths = [
            resolve_image(args.pack_dir, ts["cams"][cam]["image"]) for ts in timestamps
        ]
        print(f"{cam}: running {len(paths)} frames…", flush=True)
        rows = run_cam_stream(pre, tracker, paths)
        for i, row in enumerate(rows):
            cam_preds_by_ts[i][cam] = row

    thr = float(args.emit_thresh)
    primary = {
        "track_emit+max_conf": score_selection(
            timestamps, cam_preds_by_ts, thr, "max_conf", "emit"
        ),
        "track_emit+size_weighted": score_selection(
            timestamps, cam_preds_by_ts, thr, "size_weighted", "emit"
        ),
    }
    ablations = {
        "raw_max_conf": score_selection(
            timestamps, cam_preds_by_ts, thr, "max_conf", "raw"
        ),
        "raw_size_weighted": score_selection(
            timestamps, cam_preds_by_ts, thr, "size_weighted", "raw"
        ),
        "raw_oracle_any_tp": score_oracle(timestamps, cam_preds_by_ts, thr, "raw"),
        "track_emit_oracle_any_tp": score_oracle(
            timestamps, cam_preds_by_ts, thr, "emit"
        ),
        # per-cam emit without selection (score cam11 alone in selection terms)
        f"track_emit_single_{args.single_cam}": score_single_cam_stream(
            timestamps, cam_preds_by_ts, args.single_cam, thr, "emit"
        ),
    }
    single = score_single_cam_stream(
        timestamps, cam_preds_by_ts, args.single_cam, thr, "emit"
    )
    try:
        ckpt_rel = str(args.ball_checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        ckpt_rel = str(args.ball_checkpoint)

    best_n = max((r["n_emitted"] for r in primary.values()), default=0)
    read = (
        f"At emit_thresh={thr}, system track+selection n_emitted_max={best_n}. "
        "If still ~0–1, temporal stack did not unlock the gate; domain/confidence "
        "remains the bottleneck. Labels are provisional (auto-seeded prelabels) — "
        "not human multicam GT."
    )

    try:
        pack_rel = str(args.pack_dir.resolve().relative_to(ROOT))
    except ValueError:
        pack_rel = str(args.pack_dir)
    report = {
        "checkpoint": ckpt_rel,
        "pack": pack_rel,
        "labels": str(labels_path),
        "label_note": data.get("label_source", "provisional / unknown"),
        "use_sahi": use_sahi,
        "use_kalman": use_kalman,
        "min_thr": args.min_thr,
        "emit_thresh": thr,
        "track_thresh": args.track_thresh,
        "stride": args.stride,
        "n_timestamps": len(timestamps),
        "primary": primary,
        "ablations": ablations,
        "single_cam": single,
        "read": read,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    md_path = args.out.with_suffix(".md")
    write_md(md_path, report)
    print(json.dumps({"primary": primary, "single_cam": single}, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
