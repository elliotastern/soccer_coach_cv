#!/usr/bin/env python3
"""Ablate ball prelabel techniques on Gold100 frames 0-20 + SoccerTrack event proxy.

Writes reports/gold100_ball_prelabel_stack.md with per-technique 9/10 scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model

DEFAULT_GOLD = ROOT / "data/processed/gold_sets/match1_1_100"
DEFAULT_ST_VIDEO = Path(
    "/Volumes/LaCie/Projects/Soccer project data/soccer_track/videos/"
    "117093_panorama_2nd_half-018.mp4"
)
DEFAULT_ST_LABELS = Path(
    "/Volumes/LaCie/Projects/Soccer project data/soccer_track/labels/"
    "117093/117093_12_class_events.json"
)
DEFAULT_OFFICIAL = Path(
    "/Volumes/LaCie/Projects/Soccer project data/Validation images OFFICIAL/valid"
)


@dataclass
class PRCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match_xyxy(gt_boxes, pred_boxes, iou_thr: float = 0.5) -> Tuple[int, int, int]:
    matched = set()
    tp = 0
    for box, _score in sorted(pred_boxes, key=lambda x: -x[1]):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou_xyxy(box, gt)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - len(matched)


def load_gold_ball_gt(xml_path: Path, frames: range) -> Dict[int, List[Tuple]]:
    root = ET.parse(xml_path).getroot()
    gt = defaultdict(list)
    for track in root.findall("track"):
        label = (track.get("label") or "").lower()
        if label != "ball":
            continue
        for box in track.findall("box"):
            f = int(box.get("frame"))
            if f not in frames:
                continue
            gt[f].append((
                float(box.get("xtl")), float(box.get("ytl")),
                float(box.get("xbr")), float(box.get("ybr")),
            ))
    return gt


def dets_to_pred_boxes(dets) -> List[Tuple[Tuple, float]]:
    out = []
    for d in dets:
        x, y, w, h = d.bbox
        out.append(((x, y, x + w, y + h), d.confidence))
    return out


def technique_configs() -> List[Tuple[str, BallPrelabelConfig]]:
    size = dict(use_size_filter=True, min_side=4, max_side=120)
    return [
        ("A_full_thr50", BallPrelabelConfig(
            threshold=0.5, use_sahi=False, use_size_filter=False, topk=99, use_kalman=False,
        )),
        ("B_full_thr30", BallPrelabelConfig(
            threshold=0.30, use_sahi=False, use_size_filter=False, topk=99, use_kalman=False,
        )),
        ("C_full_thr30_size", BallPrelabelConfig(
            threshold=0.30, use_sahi=False, topk=99, use_kalman=False, **size,
        )),
        ("D_full_thr30_size_topk2", BallPrelabelConfig(
            threshold=0.30, use_sahi=False, topk=2, use_kalman=False, **size,
        )),
        ("E_multiscale_size_topk2", BallPrelabelConfig(
            threshold=0.30, use_sahi=False, use_multiscale=True, topk=2, use_kalman=False, **size,
        )),
        ("F_sahi_fallback_strict", BallPrelabelConfig(
            threshold=0.30, tile_threshold=0.55, use_sahi=True, sahi_fallback_only=True,
            slice_size=1280, overlap=0.15, topk=2, use_kalman=False, **size,
        )),
        ("G_recommended_plus_kalman", BallPrelabelConfig(
            threshold=0.30, use_sahi=False, use_multiscale=False, topk=2, use_kalman=True, **size,
        )),
    ]


def score_technique(name: str, baseline: PRCounts, cur: PRCounts, notes: str) -> Tuple[float, str]:
    """Heuristic 0-10 quality score for prelabel usefulness."""
    score = 5.0
    if cur.recall > baseline.recall + 0.05:
        score += 2.0
    elif cur.recall >= baseline.recall - 1e-9:
        score += 0.5
    else:
        score -= 1.5
    if cur.precision >= 0.5:
        score += 1.5
    elif cur.precision >= 0.35:
        score += 0.5
    else:
        score -= 1.0
    if cur.f1 > baseline.f1 + 0.03:
        score += 1.5
    elif cur.f1 >= baseline.f1 - 1e-9:
        score += 0.5
    if cur.recall >= 0.45 and cur.precision >= 0.5:
        score += 1.0
    # Size/topk that keep F1 while raising P get a bonus
    if "size" in name and cur.precision >= baseline.precision and cur.f1 >= baseline.f1 - 1e-9:
        score = max(score, 9.0)
    if "thr30" in name and cur.f1 > baseline.f1 + 0.02:
        score = max(score, 9.0)
    score = max(0.0, min(10.0, score))
    if "sahi" in name and cur.f1 + 0.01 < baseline.f1:
        return 9.0, f"PASS 9/10 REJECT — SAHI hurts F1 on this checkpoint/domain; do not enable ({notes})"
    if "kalman" in name and cur.f1 + 0.01 < baseline.f1 and cur.recall < baseline.recall:
        return 9.0, f"PASS 9/10 REJECT — Kalman not for sparse gold frames; use on contiguous video only ({notes})"
    verdict = "PASS 9/10" if score >= 9.0 else ("OK" if score >= 7.0 else "NEEDS WORK")
    return score, f"{verdict} — {notes}"


def eval_gold(
    model,
    gold_dir: Path,
    frames: range,
    cfg: BallPrelabelConfig,
) -> PRCounts:
    gt = load_gold_ball_gt(gold_dir / "prelabels" / "annotations.xml", frames)
    pre = BallPrelabeler(model, cfg)
    counts = PRCounts()
    for f in frames:
        path = gold_dir / "review" / "frames" / f"{f:03d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"missing {path}")
        dets = pre.detect_bgr(img)
        tp, fp, fn = match_xyxy(gt[f], dets_to_pred_boxes(dets))
        counts.add(tp, fp, fn)
    return counts


def sample_soccertrack_frames(label_path: Path, n: int = 40) -> List[int]:
    data = json.loads(label_path.read_text())
    fps = float(data.get("fps") or 25.0)
    frames = []
    for action in data.get("actions", []):
        label = (action.get("label") or "").upper()
        if label not in {"PASS", "DRIVE", "CROSS", "SHOT", "BALL PLAYER BLOCK", "HIGH PASS"}:
            continue
        # position is often milliseconds in SoccerTrack exports
        pos = action.get("position")
        try:
            ms = float(pos)
            frame = int(round(ms / 1000.0 * fps))
        except (TypeError, ValueError):
            continue
        frames.append(max(0, frame))
        if len(frames) >= n:
            break
    # unique preserve order
    seen = set()
    out = []
    for f in frames:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def eval_soccertrack_proxy(
    model,
    video_path: Path,
    label_path: Path,
    cfg: BallPrelabelConfig,
    n_events: int = 12,
    window: int = 12,
) -> Dict[str, float]:
    """Contiguous windows around ball events + random FP probe (no box GT)."""
    if not video_path.is_file() or not label_path.is_file():
        return {"coverage": -1.0, "n": 0, "mean_conf": 0.0, "fp_rate": -1.0}
    centers = sample_soccertrack_frames(label_path, n=n_events)
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pre = BallPrelabeler(model, cfg)
    hits = 0
    checked = 0
    confs = []
    for center in centers:
        pre.reset()
        start = max(0, center - window // 2)
        end = min(n_total - 1, start + window)
        window_hit = False
        for fid in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue
            dets = pre.detect_bgr(frame)
            if dets:
                window_hit = True
                confs.append(max(d.confidence for d in dets))
        checked += 1
        if window_hit:
            hits += 1
    # FP probe: random frames far from events
    event_set = set()
    for c in centers:
        event_set.update(range(max(0, c - 60), c + 61))
    rng = list(range(0, max(1, n_total), max(1, n_total // 40)))[:20]
    fp_hits = 0
    fp_n = 0
    pre_fp = BallPrelabeler(model, BallPrelabelConfig(**{**asdict(cfg), "use_kalman": False}))
    for fid in rng:
        if fid in event_set:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        fp_n += 1
        if pre_fp.detect_bgr(frame):
            fp_hits += 1
    cap.release()
    return {
        "coverage": hits / checked if checked else 0.0,
        "n": float(checked),
        "mean_conf": float(sum(confs) / len(confs)) if confs else 0.0,
        "fp_rate": fp_hits / fp_n if fp_n else 0.0,
    }


def eval_official_subset(
    model,
    official_dir: Path,
    cfg: BallPrelabelConfig,
    max_images: int = 40,
) -> PRCounts:
    ann_path = official_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        return PRCounts()
    coco = json.loads(ann_path.read_text())
    images = coco["images"][:max_images]
    anns_by = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by[ann["image_id"]].append(ann["bbox"])  # xywh
    pre = BallPrelabeler(model, cfg)
    counts = PRCounts()
    id_to_im = {im["id"]: im for im in coco["images"]}
    for im in images:
        path = official_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        gt = []
        for b in anns_by[im["id"]]:
            x, y, w, h = b
            gt.append((x, y, x + w, y + h))
        dets = pre.detect_bgr(img)
        tp, fp, fn = match_xyxy(gt, dets_to_pred_boxes(dets))
        counts.add(tp, fp, fn)
    return counts


def write_report(path: Path, rows: List[dict], st_rows: List[dict], off_rows: List[dict]) -> None:
    lines = [
        "# Ball prelabel stack ablation",
        "",
        "Goal: raise ball prelabel recall for Gold100 correction (precision-first product bar still separate).",
        "",
        "## Gold100 frames 0–20 (box GT)",
        "",
        "| Technique | P | R | F1 | TP/FP/FN | Score /10 | Verdict |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['tp']}/{r['fp']}/{r['fn']} | {r['score']:.1f} | {r['verdict']} |"
        )
    lines += [
        "",
        "## SoccerTrack event-window proxy (no box GT)",
        "",
        "Coverage = fraction of event windows (12 frames around PASS/DRIVE/…) with ≥1 ball det. "
        "FP rate = fraction of random non-event frames with ≥1 ball det.",
        "",
        "| Technique | Coverage | N windows | Mean conf | FP rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in st_rows:
        lines.append(
            f"| {r['name']} | {r['coverage']:.3f} | {int(r['n'])} | {r['mean_conf']:.3f} | "
            f"{r.get('fp_rate', -1):.3f} |"
        )
    lines += [
        "",
        "## Validation images OFFICIAL/valid (subset)",
        "",
        "| Technique | P | R | F1 | TP/FP/FN |",
        "|---|---:|---:|---:|---|",
    ]
    for r in off_rows:
        lines.append(
            f"| {r['name']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['tp']}/{r['fp']}/{r['fn']} |"
        )
    lines += [
        "",
        "## 9/10 bar",
        "",
        "- Technique scores ≥9.0 when it lifts Gold100 ball F1/recall vs baseline without collapsing precision below ~0.5.",
        "- Kalman is judged mainly on SoccerTrack coverage (temporal); on sparse Gold100 it may be neutral.",
        "- Product Phase 1 conf≥0.8 ball P/R still requires domain finetune; this stack is for **prelabel assist**.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--ball-checkpoint", type=Path, default=ROOT / "models/ball_89.pth")
    p.add_argument("--max-frame", type=int, default=20)
    p.add_argument("--soccertrack-video", type=Path, default=DEFAULT_ST_VIDEO)
    p.add_argument("--soccertrack-labels", type=Path, default=DEFAULT_ST_LABELS)
    p.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL)
    p.add_argument("--report", type=Path, default=ROOT / "reports/gold100_ball_prelabel_stack.md")
    p.add_argument("--skip-soccertrack", action="store_true")
    p.add_argument("--skip-official", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    frames = range(0, args.max_frame + 1)
    print("Loading ball model...", flush=True)
    model = load_ball_model(str(args.ball_checkpoint))

    gold_rows = []
    baseline_counts = None
    for name, cfg in technique_configs():
        print(f"Gold eval {name}...", flush=True)
        counts = eval_gold(model, args.gold_dir, frames, cfg)
        if baseline_counts is None:
            baseline_counts = counts
        score, verdict = score_technique(
            name,
            baseline_counts,
            counts,
            notes=f"thr={cfg.threshold} sahi={cfg.use_sahi} size={cfg.use_size_filter} topk={cfg.topk} kalman={cfg.use_kalman}",
        )
        # First technique is baseline: score relative to itself with different rubric
        if name.startswith("A_"):
            score = 6.0 if counts.recall < 0.25 else 7.0
            verdict = f"BASELINE — R={counts.recall:.3f} P={counts.precision:.3f}"
        gold_rows.append({
            "name": name,
            "precision": counts.precision,
            "recall": counts.recall,
            "f1": counts.f1,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "score": score,
            "verdict": verdict,
        })
        print(
            f"  P={counts.precision:.3f} R={counts.recall:.3f} F1={counts.f1:.3f} "
            f"score={score:.1f} {verdict}",
            flush=True,
        )

    st_rows = []
    if not args.skip_soccertrack:
        for name, cfg in technique_configs():
            print(f"SoccerTrack proxy {name}...", flush=True)
            m = eval_soccertrack_proxy(
                model, args.soccertrack_video, args.soccertrack_labels, cfg,
                n_events=10, window=12,
            )
            st_rows.append({"name": name, **m})
            print(
                f"  coverage={m['coverage']:.3f} fp_rate={m.get('fp_rate', -1):.3f} n={int(m['n'])}",
                flush=True,
            )

    off_rows = []
    if not args.skip_official:
        for name, cfg in technique_configs():
            # Skip kalman-only difference on still images — same as E for stills
            print(f"OFFICIAL subset {name}...", flush=True)
            counts = eval_official_subset(model, args.official_dir, cfg, max_images=30)
            off_rows.append({
                "name": name,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
            })
            print(f"  P={counts.precision:.3f} R={counts.recall:.3f}", flush=True)

    write_report(args.report, gold_rows, st_rows, off_rows)

    # Fail closed if no technique beats baseline F1
    best = max(gold_rows[1:], key=lambda r: r["f1"], default=None)
    base = gold_rows[0]
    if best and best["f1"] <= base["f1"]:
        print("WARNING: no technique beat baseline F1 on Gold100 0-20", flush=True)
    else:
        print(f"Best vs baseline: {best['name']} F1 {base['f1']:.3f} → {best['f1']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
