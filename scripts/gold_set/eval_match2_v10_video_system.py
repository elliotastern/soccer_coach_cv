#!/usr/bin/env python3
"""Match 2 video system: detect 0.3 → ByteTrack → emit 0.80 → best cam.

Cam5plus + Cam4plus consecutive strip (product n_emitted).
Held-out Match 2 gold 50 scored with video warmup (true P_emit / FPs).
Never trains. Gold XML/COCO is source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model
from src.perception.tracker import Tracker
from src.state.types import Detection

IOU_THR = 0.5
CLEAR_MIN_SIDE = 25.0
MASTER_CAMS = [
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
]
DEFAULT_CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
GOLD_DIR = ROOT / "data/processed/gold_sets/match2_gold_frames"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    p.add_argument("--start-sec", type=float, default=33.0)
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--min-thr", type=float, default=0.30)
    p.add_argument("--emit-thresh", type=float, default=0.80)
    p.add_argument("--track-thresh", type=float, default=0.10)
    p.add_argument("--ema-alpha", type=float, default=0.3)
    p.add_argument("--no-kalman", action="store_true")
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument("--skip-strip", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/eval_match2_v10/video_system.json",
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


def pick_selected(cam_preds: dict, mode: str):
    cands = []
    for cam, preds in cam_preds.items():
        if not preds:
            continue
        box, conf, side = preds[0]
        score = conf * side if mode == "size_weighted" else conf
        cands.append((score, cam, preds[0]))
    if not cands:
        return None, None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], cands[0][2]


def metrics_block(tp, fp, fn, n_emitted):
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


def make_prelabeler(model, min_thr: float, use_kalman: bool):
    return BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=min_thr,
            use_sahi=False,
            use_size_filter=True,
            topk=2,
            use_kalman=use_kalman,
            min_side=4,
            max_side=240,
        ),
        class_id=1,
    )


def make_tracker(args):
    return Tracker(
        track_thresh=args.track_thresh,
        emit_thresh=args.emit_thresh,
        ema_alpha=args.ema_alpha,
        apply_emit_gate=True,
        frame_rate=30,
    )


def raw_emit_from_dets(pre, tracker, frame):
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
    return {"raw": raw, "emit": emits}


def open_video(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")
    return cap


def read_n_frames_from(path: Path, start_frame: int, n: int):
    cap = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, start_frame)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def load_gold_items(gold_dir: Path):
    coco = json.loads((gold_dir / "gold" / "annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = {}
    for a in coco["annotations"]:
        if cats.get(a["category_id"]) != "ball":
            continue
        by_image.setdefault(a["image_id"], []).append(a["bbox"])
    id_by_name = {im["file_name"]: im for im in coco["images"]}
    man = json.loads((gold_dir / "manifest.json").read_text())
    items = []
    for row in man["frames"]:
        im = id_by_name.get(row["image"])
        if im is None:
            raise RuntimeError(f"gold image missing from coco: {row['image']}")
        path = gold_dir / "images" / row["image"]
        if not path.is_file():
            raise FileNotFoundError(f"missing gold jpeg: {path}")
        items.append({
            "strip_frame": row["strip_frame"],
            "camera": row["camera"],
            "video_path": Path(row["video_path"]),
            "frame_idx": int(row["frame_idx"]),
            "t_sec": float(row["t_sec"]),
            "image": row["image"],
            "image_path": path,
            "image_id": im["id"],
            "gt": by_image.get(im["id"], []),
        })
    return items


def score_rows_vs_gt(items, rows, key: str):
    tp = fp = fn = 0
    n_emitted = 0
    for item, row in zip(items, rows):
        preds = row[key]
        if preds:
            n_emitted += 1
        tpi, fpi, fni = match_tp(item["gt"], preds)
        tp += tpi
        fp += fpi
        fn += fni
    return metrics_block(tp, fp, fn, n_emitted)


def clear_recall(items, rows, key: str, clear_min_side: float):
    clear_tp = clear_fn = n_clear = 0
    for item, row in zip(items, rows):
        clear_gt = [g for g in item["gt"] if min(g[2], g[3]) >= clear_min_side]
        n_clear += len(clear_gt)
        if not clear_gt:
            continue
        ct, _, cf = match_tp(clear_gt, row[key])
        clear_tp += ct
        clear_fn += cf
    r = clear_tp / (clear_tp + clear_fn) if (clear_tp + clear_fn) else 0.0
    return {"n_gt": n_clear, "tp": clear_tp, "fn": clear_fn, "recall": r}


def run_gold_warmup(model, items, args):
    rows = []
    use_kalman = not args.no_kalman
    for i, item in enumerate(items):
        pre = make_prelabeler(model, args.min_thr, use_kalman)
        tracker = make_tracker(args)
        start = max(0, item["frame_idx"] - args.warmup)
        n_warm = item["frame_idx"] - start
        warm, _fps = read_n_frames_from(item["video_path"], start, n_warm)
        for frame in warm:
            raw_emit_from_dets(pre, tracker, frame)
        gold = cv2.imread(str(item["image_path"]))
        if gold is None:
            raise RuntimeError(f"failed to read gold jpeg {item['image_path']}")
        row = raw_emit_from_dets(pre, tracker, gold)
        top_raw = row["raw"][0][1] if row["raw"] else 0.0
        top_emit = row["emit"][0][1] if row["emit"] else 0.0
        print(
            f"gold {i:02d}/{len(items)-1} {item['image']} "
            f"raw={top_raw:.3f} emit={top_emit:.3f} n_emit={len(row['emit'])}",
            flush=True,
        )
        rows.append(row)
    return rows


def run_dual_cam_strip(model, args):
    use_kalman = not args.no_kalman
    cam_rows = {}
    fps = None
    start_frame = None
    for name, path in MASTER_CAMS:
        cap = open_video(path)
        cam_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps is None:
            fps = cam_fps
            start_frame = int(round(args.start_sec * cam_fps))
        cap.release()
        frames, _ = read_n_frames_from(path, start_frame, args.num_frames)
        if len(frames) < args.num_frames:
            raise RuntimeError(f"{name}: got {len(frames)}/{args.num_frames} frames")
        pre = make_prelabeler(model, args.min_thr, use_kalman)
        tracker = make_tracker(args)
        rows = []
        print(f"{name}: {len(frames)} frames from f={start_frame}", flush=True)
        for i, frame in enumerate(frames):
            row = raw_emit_from_dets(pre, tracker, frame)
            rows.append(row)
            top = row["emit"][0][1] if row["emit"] else 0.0
            print(f"  {name} f={i:03d} n_emit={len(row['emit'])} top={top:.3f}", flush=True)
        cam_rows[name] = {"frames": frames, "rows": rows, "fps": cam_fps}
    return cam_rows, start_frame, fps


def score_strip_selection(cam_rows, mode: str, key: str, n: int):
    n_emitted = 0
    confs = []
    per_cam = {name: 0 for name in cam_rows}
    for i in range(n):
        pred_map = {name: cam_rows[name]["rows"][i][key] for name in cam_rows}
        cam, pred = pick_selected(pred_map, mode)
        if pred is None:
            continue
        n_emitted += 1
        confs.append(pred[1])
        per_cam[cam] += 1
    return {
        "n_emitted": n_emitted,
        "n_frames": n,
        "emit_rate": n_emitted / n if n else 0.0,
        "mean_conf": (sum(confs) / len(confs)) if confs else None,
        "per_cam": per_cam,
        "note": "no GT on this strip — n_emitted/rate only, not P_emit",
    }


def gold_in_strip(items, start_frame: int, n: int, cam_names):
    hits = []
    for item in items:
        if item["camera"] not in cam_names:
            continue
        idx = item["frame_idx"]
        if start_frame <= idx < start_frame + n:
            hits.append({
                "image": item["image"],
                "camera": item["camera"],
                "frame_idx": idx,
                "strip_i": idx - start_frame,
            })
    return hits


def score_gold_on_strip(items, cam_rows, start_frame: int, n: int, mode: str):
    scored = []
    tp = fp = fn = 0
    n_emitted = 0
    cam_names = set(cam_rows)
    for item in items:
        if item["camera"] not in cam_names:
            continue
        strip_i = item["frame_idx"] - start_frame
        if strip_i < 0 or strip_i >= n:
            continue
        pred_map = {
            name: cam_rows[name]["rows"][strip_i]["emit"] for name in cam_rows
        }
        cam, pred = pick_selected(pred_map, mode)
        preds = [pred] if pred is not None else []
        if cam != item["camera"]:
            scored.append({
                "image": item["image"],
                "gold_cam": item["camera"],
                "selected_cam": cam,
                "skipped": "selected other cam; no GT on that view",
            })
            continue
        if preds:
            n_emitted += 1
        tpi, fpi, fni = match_tp(item["gt"], preds)
        tp += tpi
        fp += fpi
        fn += fni
        scored.append({
            "image": item["image"],
            "gold_cam": item["camera"],
            "selected_cam": cam,
            "tp": tpi,
            "fp": fpi,
            "fn": fni,
        })
    block = metrics_block(tp, fp, fn, n_emitted)
    block["frames"] = scored
    return block


def draw_pred(frame, pred, label: str):
    out = frame.copy()
    if pred is None:
        tag = f"{label}  no emit"
        cv2.rectangle(out, (8, 8), (720, 52), (0, 0, 0), -1)
        cv2.putText(out, tag, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
        return out
    box, conf, side = pred
    x, y, w, h = box
    color = (0, 255, 0) if conf >= 0.80 else (0, 220, 255)
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
    tag = f"{label}  {conf:.2f}  s={side:.0f}"
    cv2.rectangle(out, (8, 8), (720, 52), (0, 0, 0), -1)
    cv2.putText(out, tag, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out


def tile_contact(frames, cols=10, cell=320):
    n = len(frames)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (cell, cell))
        y0, x0 = r * cell, c * cell
        sheet[y0:y0 + cell, x0:x0 + cell] = thumb
    return sheet


def write_strip_visuals(out_dir: Path, cam_rows, n: int):
    annotated = []
    for i in range(n):
        pred_map = {name: cam_rows[name]["rows"][i]["emit"] for name in cam_rows}
        cam, pred = pick_selected(pred_map, "max_conf")
        if cam is None:
            frame = cam_rows["Cam5plus"]["frames"][i]
            annotated.append(draw_pred(frame, None, "none"))
            continue
        frame = cam_rows[cam]["frames"][i]
        annotated.append(draw_pred(frame, pred, cam))
    sheet = tile_contact(annotated)
    path = out_dir / "bestcam_contact_10x10.jpg"
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return path


def write_md(path: Path, report: dict):
    gold = report.get("gold_warmup_emit") or {}
    strip = report.get("strip_track_emit_max_conf") or {}
    pe = gold.get("P_emit")
    pe_s = f"{pe:.3f}" if pe is not None else "none"
    lines = [
        "# Match 2 v10 video system",
        "",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"Stack: detect={report['min_thr']} → ByteTrack → emit={report['emit_thresh']} "
        f"Kalman={report['use_kalman']} SAHI=False  ",
        f"Master cams: Cam5plus + Cam4plus  ",
        f"Strip: t={report['start_sec']}s, {report['num_frames']} frames  ",
        f"Gold warmup: {report['warmup']} frames before each held-out gold JPEG",
        "",
        "## Client primary — gold 50 with tracker warmup @ emit 0.80",
        "",
        f"- P_emit: **{pe_s}**",
        f"- n_emitted: **{gold.get('n_emitted')}**",
        f"- FP: **{gold.get('fp')}**  TP: {gold.get('tp')}  FN: {gold.get('fn')}",
        f"- hollow: {gold.get('hollow')}",
        f"- clear-ball R: {report.get('gold_clear_ball', {}).get('recall')}",
        "",
        "## Product strip — Cam5plus/Cam4plus best-cam (no GT on full strip)",
        "",
        f"- n_emitted: **{strip.get('n_emitted')}** / {strip.get('n_frames')} "
        f"(rate {strip.get('emit_rate')})",
        f"- mean emit conf: {strip.get('mean_conf')}",
        f"- per cam: {strip.get('per_cam')}",
        "",
        report.get("read", ""),
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    if not args.ball_checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {args.ball_checkpoint}")
    print(f"checkpoint: {args.ball_checkpoint}", flush=True)
    model = load_ball_model(str(args.ball_checkpoint))
    items = load_gold_items(args.gold_dir)
    print(f"gold items: {len(items)}", flush=True)

    gold_emit = gold_raw = gold_clear = None
    gold_rows = []
    if not args.skip_gold:
        gold_rows = run_gold_warmup(model, items, args)
        gold_emit = score_rows_vs_gt(items, gold_rows, "emit")
        gold_raw = score_rows_vs_gt(items, gold_rows, "raw")
        gold_clear = clear_recall(items, gold_rows, "emit", CLEAR_MIN_SIDE)
        print(f"gold emit: {json.dumps(gold_emit)}", flush=True)

    strip_sel = strip_raw = strip_gold = overlap = None
    start_frame = None
    if not args.skip_strip:
        cam_rows, start_frame, fps = run_dual_cam_strip(model, args)
        n = args.num_frames
        strip_sel = score_strip_selection(cam_rows, "max_conf", "emit", n)
        strip_raw = score_strip_selection(cam_rows, "max_conf", "raw", n)
        overlap = gold_in_strip(
            items, start_frame, n, [name for name, _ in MASTER_CAMS]
        )
        strip_gold = score_gold_on_strip(items, cam_rows, start_frame, n, "max_conf")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        vis = write_strip_visuals(args.out.parent, cam_rows, n)
        print(f"contact: {vis}", flush=True)
        print(f"strip emit: {json.dumps(strip_sel)}", flush=True)

    try:
        ckpt_rel = str(args.ball_checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        ckpt_rel = str(args.ball_checkpoint)

    read = (
        "P_emit/FPs come from held-out Match 2 gold 50 with video warmup + emit gate. "
        "The Cam5plus/Cam4plus strip reports product n_emitted (how often we publish). "
        "Strip has no full GT; overlapping gold frames are scored separately."
    )
    report = {
        "checkpoint": ckpt_rel,
        "min_thr": args.min_thr,
        "emit_thresh": args.emit_thresh,
        "use_kalman": not args.no_kalman,
        "use_sahi": False,
        "warmup": args.warmup,
        "start_sec": args.start_sec,
        "num_frames": args.num_frames,
        "strip_start_frame": start_frame,
        "gold_warmup_emit": gold_emit,
        "gold_warmup_raw": gold_raw,
        "gold_clear_ball": gold_clear,
        "strip_track_emit_max_conf": strip_sel,
        "strip_raw_max_conf": strip_raw,
        "strip_gold_overlap": overlap,
        "strip_gold_scored": strip_gold,
        "read": read,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    md_path = args.out.with_suffix(".md")
    write_md(md_path, report)
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
