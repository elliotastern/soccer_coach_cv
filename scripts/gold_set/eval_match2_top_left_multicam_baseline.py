#!/usr/bin/env python3
"""Match2 Top Left 6 P-cam baseline + soft consensus (no SAHI, no calib).

Detect once at low thr on synced 4quad Top Left clips (downscaled to 1920×1080),
then score:
  A) baseline max_conf @0.30
  B) baseline emit @0.80
  C) soft ≥2-cam co-occurrence @0.15

System R/P (dual gold): frames where selected cam is P10 or P7, scored vs that
cam's Top Left 300 gold. Goal R≥0.8 P≥0.9 on that covered set.
Never trains.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import det_tuple, pick_selected  # noqa: E402
from eval_poc_ball_metrics import match_preds, pr  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

P_CAMS = ["P1", "P6", "P7", "P8", "P10", "P12"]
GOLD_CAMS = ["P7", "P10"]
N_FRAMES = 300
DETECT_W, DETECT_H = 1920, 1080
CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
GOLD_DIR_P10 = ROOT / "data/processed/gold_sets/match2_4quad_top_left"
GOLD_DIR_P7 = ROOT / "data/processed/gold_sets/match2_4quad_top_left_p7"
SOURCE_DIR = ROOT / "reports/eval_match2_v10/4quad_test/source"
OUT_BASE = ROOT / "reports/eval_match2_v10/top_left_multicam_baseline"
OUT_CONS = ROOT / "reports/eval_match2_v10/top_left_multicam_consensus"
SIZE = dict(use_size_filter=True, min_side=4, max_side=240, use_kalman=False)
GOAL_R, GOAL_P = 0.80, 0.90


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--gold-dir-p10", type=Path, default=GOLD_DIR_P10)
    p.add_argument("--gold-dir-p7", type=Path, default=GOLD_DIR_P7)
    p.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--out-baseline", type=Path, default=OUT_BASE)
    p.add_argument("--out-consensus", type=Path, default=OUT_CONS)
    p.add_argument("--detect-thr", type=float, default=0.10)
    p.add_argument("--baseline-thr", type=float, default=0.30)
    p.add_argument("--emit-thr", type=float, default=0.80)
    p.add_argument("--consensus-thr", type=float, default=0.15)
    p.add_argument("--min-cams", type=int, default=2)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--skip-detect", action="store_true")
    return p.parse_args()


def source_path(source_dir: Path, cam: str) -> Path:
    return source_dir / f"quad_top_left_t00026.0s_{cam}.mp4"


def load_top_left_gt(xml_path: Path) -> dict[int, list]:
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
            raw[frame].append(((xtl, ytl, xbr - xtl, ybr - ytl), source))
    out = {}
    for frame, rows in raw.items():
        manuals = [b for b, s in rows if s == "manual"]
        autos = [b for b, s in rows if s != "manual"]
        out[frame] = manuals if manuals else autos
    return out


def open_cams(source_dir: Path):
    caps = {}
    for cam in P_CAMS:
        path = source_path(source_dir, cam)
        if not path.is_file():
            raise FileNotFoundError(f"missing source {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    return caps


def read_resized(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
        frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
    return frame


def dets_to_rows(dets) -> list:
    rows = [det_tuple(d) for d in dets]
    rows.sort(key=lambda x: -x[1])
    return rows


def filter_rows(rows: list, thr: float, topk: int = 2) -> list:
    return [r for r in rows if r[1] >= thr][:topk]


def run_detect(model, source_dir: Path, thr: float) -> dict:
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(threshold=thr, use_sahi=False, topk=5, **SIZE),
        class_id=1,
    )
    caps = open_cams(source_dir)
    out = {cam: [] for cam in P_CAMS}
    try:
        for i in range(N_FRAMES):
            for cam in P_CAMS:
                frame = read_resized(caps[cam])
                if frame is None:
                    raise RuntimeError(f"{cam} ended early at frame {i}")
                out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
            if i % 50 == 0:
                n = sum(1 for c in P_CAMS if out[c][i])
                print(f"detect {i}/{N_FRAMES - 1} cams_with_det={n}", flush=True)
    finally:
        for cap in caps.values():
            cap.release()
    return out


def cache_dump(path: Path, dets: dict):
    serial = {
        cam: [[[list(box), conf, side] for box, conf, side in rows] for rows in frames]
        for cam, frames in dets.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_frames": N_FRAMES, "cams": serial}), encoding="utf-8")


def cache_load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for cam, frames in payload["cams"].items():
        out[cam] = [
            [(tuple(box), float(conf), float(side)) for box, conf, side in rows]
            for rows in frames
        ]
    return out


def select_frame(cam_rows: dict, min_cams: int = 1):
    active = {c: r for c, r in cam_rows.items() if r}
    if len(active) < min_cams:
        return None, None
    return pick_selected(active, "max_conf")


def score_proxy(gt: dict, frames: list, preds_by_frame: list, note: str) -> dict:
    tp = fp = fn = 0
    n_gt = 0
    for i, preds in zip(frames, preds_by_frame):
        g = gt.get(i, [])
        n_gt += len(g)
        tpi, fpi, fni = match_preds(g, preds)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "n_frames_scored": len(frames),
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "hit_goal_r": r >= GOAL_R,
        "hit_goal_p": p >= GOAL_P,
        "hit_goal": (r >= GOAL_R and p >= GOAL_P),
        "note": note,
    }


def merge_proxy_scores(*parts: dict) -> dict:
    tp = sum(p["tp"] for p in parts)
    fp = sum(p["fp"] for p in parts)
    fn = sum(p["fn"] for p in parts)
    n_frames = sum(p["n_frames_scored"] for p in parts)
    n_gt = sum(p["n_gt"] for p in parts)
    p, r = pr(tp, fp, fn)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "n_frames_scored": n_frames,
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "hit_goal_r": r >= GOAL_R,
        "hit_goal_p": p >= GOAL_P,
        "hit_goal": (r >= GOAL_R and p >= GOAL_P),
        "note": "selected cam in {P7,P10}; GT from that cam's gold",
    }


def score_all_frames(gt: dict, preds_per_frame: list, thr: float) -> dict:
    tp = fp = fn = 0
    n_gt = sum(len(gt.get(i, [])) for i in range(N_FRAMES))
    for i in range(N_FRAMES):
        preds = [(b, c) for b, c in preds_per_frame[i] if c >= thr]
        tpi, fpi, fni = match_preds(gt.get(i, []), preds)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "hit_goal_r": r >= GOAL_R,
        "hit_goal_p": p >= GOAL_P,
        "hit_goal": (r >= GOAL_R and p >= GOAL_P),
    }


def score_max_conf(
    dets,
    gt_by_cam: dict,
    thr: float,
    min_cams: int,
    emit_thr: float | None = None,
) -> dict:
    """gt_by_cam maps cam name -> frame->boxes gold (e.g. P7, P10)."""
    select_counts = Counter()
    n_selected = 0
    n_empty = 0
    n_support = []
    by_cam_frames = {cam: [] for cam in gt_by_cam}
    by_cam_preds = {cam: [] for cam in gt_by_cam}
    single_preds = {cam: [] for cam in gt_by_cam}
    score_thr = emit_thr if emit_thr is not None else thr
    for i in range(N_FRAMES):
        cam_rows = {cam: filter_rows(dets[cam][i], thr) for cam in P_CAMS}
        n_support.append(sum(1 for r in cam_rows.values() if r))
        for gcam in gt_by_cam:
            single_preds[gcam].append(
                [(b, c) for b, c, _ in cam_rows.get(gcam, []) if c >= score_thr]
            )
        cam, pred = select_frame(cam_rows, min_cams=min_cams)
        if cam is None or pred is None:
            n_empty += 1
            select_counts["none"] += 1
            continue
        if emit_thr is not None and pred[1] < emit_thr:
            n_empty += 1
            select_counts["below_emit"] += 1
            continue
        n_selected += 1
        select_counts[cam] += 1
        if cam in gt_by_cam:
            by_cam_frames[cam].append(i)
            by_cam_preds[cam].append([(list(pred[0]), float(pred[1]))])
    proxy_by_cam = {
        cam: score_proxy(
            gt_by_cam[cam],
            by_cam_frames[cam],
            by_cam_preds[cam],
            note=f"only timestamps where selected cam is {cam}",
        )
        for cam in gt_by_cam
    }
    parts = [proxy_by_cam[c] for c in GOLD_CAMS if c in proxy_by_cam]
    combined = merge_proxy_scores(*parts) if parts else merge_proxy_scores(
        score_proxy({}, [], [], note="empty")
    )
    single = {
        cam: score_all_frames(gt_by_cam[cam], single_preds[cam], score_thr)
        for cam in gt_by_cam
    }
    n_covered = sum(proxy_by_cam[c]["n_frames_scored"] for c in gt_by_cam)
    return {
        "thr": thr,
        "emit_thr": emit_thr,
        "min_cams": min_cams,
        "n_frames": N_FRAMES,
        "n_selected": n_selected,
        "n_empty_or_below_emit": n_empty,
        "selection_share": {k: v / N_FRAMES for k, v in sorted(select_counts.items())},
        "selection_counts": dict(select_counts),
        "mean_cams_with_det": sum(n_support) / max(len(n_support), 1),
        "proxy_by_cam": proxy_by_cam,
        "proxy_p7_or_p10": combined,
        "proxy_p10_selected": proxy_by_cam.get("P10", combined),
        "proxy_p7_selected": proxy_by_cam.get("P7", combined),
        "single_cam": single,
        "p10_single_cam": single.get("P10", {}),
        "p7_single_cam": single.get("P7", {}),
        "n_proxy_frames": n_covered,
        "coverage_share": n_covered / N_FRAMES,
    }


def _fmt_pr(label: str, s: dict) -> str:
    return (
        f"- {label} P={s['precision']:.3f} R={s['recall']:.3f} "
        f"n_frames={s.get('n_frames_scored', '—')} "
        f"tp/fp/fn={s['tp']}/{s['fp']}/{s['fn']} "
        f"goal={'HIT' if s.get('hit_goal') else 'MISS'}"
    )


def write_baseline_report(out: Path, payload: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    a, b = payload["baseline_a"], payload["baseline_b"]
    cov = a["proxy_p7_or_p10"]
    lines = [
        "# Match2 Top Left — 6 P-cam multicam baseline (dual gold)",
        "",
        f"Window 0:26–0:31 · cams `{', '.join(P_CAMS)}` · v10 · no SAHI · "
        f"detect@{DETECT_W}x{DETECT_H}",
        "",
        "System score: when selected cam is **P7 or P10**, compare to that cam's gold.",
        f"Goal: **R≥{GOAL_R} P≥{GOAL_P}** on the covered (P7∪P10-selected) set.",
        "",
        "## Baseline A — max_conf @0.30",
        "",
        f"- Selected frames: {a['n_selected']}/{a['n_frames']}",
        f"- Gold-covered share: {a['coverage_share']:.1%} "
        f"({a['n_proxy_frames']} frames)",
        f"- Selection share: `{json.dumps(a['selection_share'])}`",
        f"- Mean cams with det: {a['mean_cams_with_det']:.2f}",
        _fmt_pr("**P7∪P10 system**", cov),
        _fmt_pr("P10-selected only", a["proxy_p10_selected"]),
        _fmt_pr("P7-selected only", a["proxy_p7_selected"]),
        (
            f"- P10 single-cam @0.30 P={a['p10_single_cam']['precision']:.3f} "
            f"R={a['p10_single_cam']['recall']:.3f}"
        ),
        (
            f"- P7 single-cam @0.30 P={a['p7_single_cam']['precision']:.3f} "
            f"R={a['p7_single_cam']['recall']:.3f}"
        ),
        "",
        "## Baseline B — emit ≥0.80 after max_conf",
        "",
        f"- Emitted: {b['n_selected']}/{b['n_frames']}",
        _fmt_pr("**P7∪P10 system**", b["proxy_p7_or_p10"]),
        _fmt_pr("P10-selected", b["proxy_p10_selected"]),
        _fmt_pr("P7-selected", b["proxy_p7_selected"]),
        "",
        "## Verdict vs goal",
        "",
        (
            f"**{'HIT' if cov['hit_goal'] else 'MISS'}** live @0.30 on covered frames: "
            f"R={cov['recall']:.3f} (need ≥{GOAL_R}), "
            f"P={cov['precision']:.3f} (need ≥{GOAL_P})."
        ),
        "",
    ]
    path = out / "baseline.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_consensus_report(out: Path, payload: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "consensus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    c, a = payload["soft_consensus"], payload["baseline_a"]
    ac, cc = a["proxy_p7_or_p10"], c["proxy_p7_or_p10"]
    d_r = cc["recall"] - ac["recall"]
    d_p = cc["precision"] - ac["precision"]
    if cc["hit_goal"]:
        gate = (
            "GO on covered P7∪P10 @ consensus → next 5090 latency for live path. "
            "Still unlabeled cams (P1/P6/P8/P12) before full-match claim."
        )
    elif ac["hit_goal"] and not cc["hit_goal"]:
        gate = "Baseline A hits goal; consensus does not — keep max_conf @0.30."
    elif ac["hit_goal_p"] and not ac["hit_goal_r"]:
        gate = (
            "NO-GO on recall for covered system → improve detect/selection before "
            "latency or epipolar. Dense SAHI stays offline-only."
        )
    elif ac["hit_goal_r"] and not ac["hit_goal_p"]:
        gate = "NO-GO on precision → raise thr / tighten selection before adding cams."
    else:
        gate = "NO-GO on both R and P for covered system — diagnose P7 vs P10 split."
    lines = [
        "# Match2 Top Left — soft 2-cam consensus vs baseline (dual gold)",
        "",
        f"Consensus: thr={c['thr']}, min_cams={c['min_cams']}, max_conf among supporters, no SAHI.",
        "",
        "| stack | system P | system R | ΔP | ΔR | n_covered | mean cams w/ det | goal |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| baseline_a @0.30 | {ac['precision']:.3f} | {ac['recall']:.3f} | — | — | "
            f"{ac['n_frames_scored']} | {a['mean_cams_with_det']:.2f} | "
            f"{'HIT' if ac['hit_goal'] else 'MISS'} |"
        ),
        (
            f"| soft_consensus | {cc['precision']:.3f} | {cc['recall']:.3f} | "
            f"{d_p:+.3f} | {d_r:+.3f} | {cc['n_frames_scored']} | "
            f"{c['mean_cams_with_det']:.2f} | {'HIT' if cc['hit_goal'] else 'MISS'} |"
        ),
        "",
        f"Selection share (consensus): `{json.dumps(c['selection_share'])}`",
        "",
        "## Gate",
        "",
        gate,
        "",
        "Epipolar still blocked (no Match 2 extrinsics). Dense SAHI stays out of the live path.",
        "",
    ]
    path = out / "consensus.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    xml_p10 = args.gold_dir_p10 / "gold" / "annotations.xml"
    xml_p7 = args.gold_dir_p7 / "gold" / "annotations.xml"
    for xml_path in (xml_p10, xml_p7):
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
    if not args.ball_checkpoint.is_file():
        raise FileNotFoundError(args.ball_checkpoint)
    gt_by_cam = {
        "P10": load_top_left_gt(xml_p10),
        "P7": load_top_left_gt(xml_p7),
    }
    for cam, gt in gt_by_cam.items():
        print(
            f"{cam} gt_frames={len(gt)} gt_boxes={sum(len(v) for v in gt.values())}",
            flush=True,
        )

    cache = args.cache or (args.out_baseline / "det_cache_thr010.json")
    if args.skip_detect and cache.is_file():
        dets = cache_load(cache)
        print(f"loaded cache {cache}", flush=True)
    else:
        model = load_ball_model(str(args.ball_checkpoint))
        print("detecting…", flush=True)
        dets = run_detect(model, args.source_dir, args.detect_thr)
        cache_dump(cache, dets)
        print(f"wrote cache {cache}", flush=True)

    baseline_a = score_max_conf(dets, gt_by_cam, thr=args.baseline_thr, min_cams=1)
    baseline_b = score_max_conf(
        dets, gt_by_cam, thr=args.baseline_thr, min_cams=1, emit_thr=args.emit_thr
    )
    soft = score_max_conf(
        dets, gt_by_cam, thr=args.consensus_thr, min_cams=args.min_cams
    )

    base_payload = {
        "title": "top_left_multicam_baseline_dual_gold",
        "cams": P_CAMS,
        "gold_cams": GOLD_CAMS,
        "checkpoint": str(args.ball_checkpoint),
        "detect_thr_cache": args.detect_thr,
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
    }
    md_a = write_baseline_report(args.out_baseline, base_payload)
    print(f"wrote {md_a}", flush=True)

    cons_payload = {
        "title": "top_left_multicam_consensus_dual_gold",
        "baseline_a": baseline_a,
        "soft_consensus": soft,
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
    }
    md_c = write_consensus_report(args.out_consensus, cons_payload)
    print(f"wrote {md_c}", flush=True)
    cov = baseline_a["proxy_p7_or_p10"]
    print(
        f"A system P/R={cov['precision']:.3f}/{cov['recall']:.3f} "
        f"goal={'HIT' if cov['hit_goal'] else 'MISS'} | "
        f"P10={baseline_a['proxy_p10_selected']['precision']:.3f}/"
        f"{baseline_a['proxy_p10_selected']['recall']:.3f} | "
        f"P7={baseline_a['proxy_p7_selected']['precision']:.3f}/"
        f"{baseline_a['proxy_p7_selected']['recall']:.3f} | "
        f"soft P/R={soft['proxy_p7_or_p10']['precision']:.3f}/"
        f"{soft['proxy_p7_or_p10']['recall']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())