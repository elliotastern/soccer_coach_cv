#!/usr/bin/env python3
"""Match 2 gold 100 frames in a row test — ball size + detect smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLEAR_MIN_SIDE = 25.0
DETECT_THR = 0.30
REVIEW_MAX_W = 1920
MATCH2_CAMS = [
    ("P1", ROOT / "data/raw/Match 2/Cam 3-P1.mp4"),
    ("P6", ROOT / "data/raw/Match 2/Cam 6-P6-002.mp4"),
    ("P7", ROOT / "data/raw/Match 2/Cam 11-P7-003.mp4"),
    ("P8", ROOT / "data/raw/Match 2/Cam 14-P8-001.mp4"),
    ("P10", ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"),
    ("P12", ROOT / "data/raw/Match 2/Cam 10-P12-001.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
]
MATCH1_DIR = ROOT / "data/processed/multicam_20s_match1/clips_osd"
MATCH1_CAMS = [
    ("cam8", MATCH1_DIR / "cam8_20s.mp4"),
    ("cam9", MATCH1_DIR / "cam9_20s.mp4"),
    ("cam11", MATCH1_DIR / "cam11_20s.mp4"),
    ("cam13", MATCH1_DIR / "cam13_20s.mp4"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-sec", type=float, default=33.0)
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument(
        "--ball-checkpoint",
        default=str(ROOT / "models/v8_snaps/post_train/checkpoint.pth"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/match2_gold_100_frames_in_a_row",
    )
    p.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "data/processed/match2_gold_100_in_a_row",
    )
    p.add_argument("--match1-start-frame", type=int, default=0)
    p.add_argument("--skip-match1", action="store_true")
    return p.parse_args()


def load_model(checkpoint: str):
    from src.perception.rfdetr_local import load_ball_model
    return load_ball_model(checkpoint)


def detect_balls(model, frame_bgr, thr: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raw = model.predict(Image.fromarray(rgb), threshold=thr)
    balls = []
    if not hasattr(raw, "class_id"):
        return balls
    for i in range(len(raw.class_id)):
        x1, y1, x2, y2 = map(float, raw.xyxy[i])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        side = min(bw, bh)
        if side < 8 or side > 240:
            continue
        aspect = bw / bh if bh else 999.0
        if aspect < 0.35 or aspect > 2.8:
            continue
        balls.append({
            "bbox": [x1, y1, bw, bh],
            "confidence": float(raw.confidence[i]),
            "side": side,
        })
    balls.sort(key=lambda b: -b["confidence"])
    return balls[:2]


def conf_color(c: float):
    if c >= 0.8:
        return (0, 255, 0)
    if c >= 0.5:
        return (0, 220, 255)
    return (0, 140, 255)


def draw_balls(frame, balls, label: str):
    out = frame.copy()
    top = balls[0]["confidence"] if balls else 0.0
    side = balls[0]["side"] if balls else 0.0
    text = f"{label}  n={len(balls)}  top={top:.2f}  side={side:.0f}"
    cv2.rectangle(out, (8, 8), (720, 52), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    for b in balls:
        x, y, w, h = b["bbox"]
        color = conf_color(b["confidence"])
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(out, p1, p2, color, 3)
        tag = f"ball {b['confidence']:.2f} s={b['side']:.0f}"
        cv2.putText(out, tag, (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")
    return cap


def seek_sec(cap, start_sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)


def seek_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))


def read_n_frames(cap, n: int):
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def score_frames(model, frames, thr: float):
    rows = []
    for i, frame in enumerate(frames):
        balls = detect_balls(model, frame, thr)
        rows.append({"frame": i, "balls": balls})
        top = balls[0]["confidence"] if balls else 0.0
        print(f"  f={i:03d} n={len(balls)} top={top:.3f}")
    return rows


def summarize_rows(rows, clear_side: float = CLEAR_MIN_SIDE):
    n = len(rows)
    sides = [b["side"] for r in rows for b in r["balls"]]
    tops = [r["balls"][0]["confidence"] for r in rows if r["balls"]]
    det05 = sum(1 for r in rows if any(b["confidence"] >= 0.5 for b in r["balls"]))
    det08 = sum(1 for r in rows if any(b["confidence"] >= 0.8 for b in r["balls"]))
    clear_hits = sum(
        1 for r in rows
        if any(b["confidence"] >= 0.5 and b["side"] >= clear_side for b in r["balls"])
    )
    with_ball = sum(1 for r in rows if r["balls"])
    max_conf_series = [
        (r["balls"][0]["confidence"] if r["balls"] else 0.0) for r in rows
    ]
    return {
        "n_frames": n,
        "frames_with_det_ge_0_5": det05,
        "frames_with_det_ge_0_8": det08,
        "frames_with_any_det": with_ball,
        "pct_det_ge_0_5": det05 / n if n else 0.0,
        "pct_det_ge_0_8": det08 / n if n else 0.0,
        "max_conf": float(max(tops)) if tops else None,
        "mean_side": float(np.mean(sides)) if sides else None,
        "median_side": float(np.median(sides)) if sides else None,
        "p10_side": float(np.percentile(sides, 10)) if sides else None,
        "frames_side_ge_clear_and_det05": clear_hits,
        "pct_clear_proxy_of_frames": clear_hits / n if n else 0.0,
        "clear_min_side": clear_side,
        "max_conf_series": max_conf_series,
    }


def tile_contact(annotated, cols: int = 10, cell: int = 320):
    n = len(annotated)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, frame in enumerate(annotated):
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (cell, cell))
        y0, x0 = r * cell, c * cell
        sheet[y0:y0 + cell, x0:x0 + cell] = thumb
    return sheet


def build_grid(thumbs: list, cols: int = 4):
    if not thumbs:
        return None
    h = min(t.shape[0] for t in thumbs)
    w = min(t.shape[1] for t in thumbs)
    resized = [cv2.resize(t, (w, h)) for t in thumbs]
    while len(resized) % cols:
        resized.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = []
    for i in range(0, len(resized), cols):
        rows.append(np.hstack(resized[i:i + cols]))
    return np.vstack(rows)


def write_overlay_mp4(path: Path, frames, fps: float):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )
    for f in frames:
        writer.write(f)
    writer.release()


def resize_review(frame, max_w: int = REVIEW_MAX_W):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    scale = max_w / w
    out = cv2.resize(frame, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def scale_balls(balls, scale: float):
    out = []
    for b in balls:
        x, y, w, h = b["bbox"]
        out.append({
            "bbox": [x * scale, y * scale, w * scale, h * scale],
            "confidence": b["confidence"],
            "side": b["side"],
            "side_fullres": b["side"],
        })
    return out


def write_review_cam(review_dir: Path, name: str, frames, rows, stats, scale_note):
    cam_dir = review_dir / "cams" / name
    frames_dir = cam_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    review_h = review_w = None
    scale = 1.0
    for i, (frame, row) in enumerate(zip(frames, rows)):
        thumb, scale = resize_review(frame)
        review_h, review_w = thumb.shape[:2]
        cv2.imwrite(
            str(frames_dir / f"{i:03d}.jpg"),
            thumb,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        review_rows.append({
            "frame": i,
            "balls": scale_balls(row["balls"], scale),
        })
    (cam_dir / "preds.json").write_text(json.dumps(review_rows, indent=2))
    (cam_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return {
        "label": name,
        "n_frames": len(frames),
        "review_width": review_w,
        "review_height": review_h,
        "scale_from_fullres": scale,
        "frames_glob": f"cams/{name}/frames/%03d.jpg",
        "preds": f"cams/{name}/preds.json",
        "stats": f"cams/{name}/stats.json",
        **scale_note,
    }


def process_cam(model, name: str, path: Path, seek_mode: str, seek_val, n: int):
    print(f"\n=== {name}: {path.name} ===")
    if not path.is_file():
        raise FileNotFoundError(f"missing video: {path}")
    cap = open_video(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if seek_mode == "sec":
        seek_sec(cap, float(seek_val))
    else:
        seek_frame(cap, int(seek_val))
    frames = read_n_frames(cap, n)
    cap.release()
    if len(frames) < n:
        print(f"  warning: got {len(frames)}/{n} frames")
    rows = score_frames(model, frames, DETECT_THR)
    annotated = [draw_balls(f, r["balls"], name) for f, r in zip(frames, rows)]
    stats = summarize_rows(rows)
    stats["cam"] = name
    stats["source"] = str(path)
    stats["fps"] = fps
    return stats, rows, annotated, fps, frames


def save_cam_outputs(out_dir: Path, name: str, annotated, rows, fps: float):
    cam_dir = out_dir / name
    cam_dir.mkdir(parents=True, exist_ok=True)
    sheet = tile_contact(annotated)
    sheet_path = cam_dir / "contact_10x10.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    write_overlay_mp4(cam_dir / "overlay.mp4", annotated, fps)
    (cam_dir / "preds.json").write_text(json.dumps(rows, indent=2))
    return sheet_path


def markdown_table(stats_list: list) -> str:
    header = (
        "| cam | n | @0.5 | @0.8 | mean side | median side | p10 side | "
        "clear≥25 @0.5 | max conf |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for s in stats_list:
        lines.append(
            f"| {s['cam']} | {s['n_frames']} | "
            f"{s['frames_with_det_ge_0_5']} ({s['pct_det_ge_0_5']:.0%}) | "
            f"{s['frames_with_det_ge_0_8']} ({s['pct_det_ge_0_8']:.0%}) | "
            f"{_fmt(s['mean_side'])} | {_fmt(s['median_side'])} | "
            f"{_fmt(s['p10_side'])} | "
            f"{s['frames_side_ge_clear_and_det05']} "
            f"({s['pct_clear_proxy_of_frames']:.0%}) | "
            f"{_fmt(s['max_conf'], 3)} |\n"
        )
    return "".join(lines)


def _fmt(v, digits=1):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def write_summary(out_dir: Path, meta: dict, m2_stats: list, m1_stats: list):
    slim_m2 = [{k: v for k, v in s.items() if k != "max_conf_series"} for s in m2_stats]
    slim_m1 = [{k: v for k, v in s.items() if k != "max_conf_series"} for s in m1_stats]
    payload = {
        "name": "match 2 gold 100 frames in a row test",
        "meta": meta,
        "match2": slim_m2,
        "match1_baseline": slim_m1,
        "notes": [
            "No GT — rates are presence/emit proxies, not P_emit.",
            "Clear proxy: min(bbox side) >= 25 px and conf >= 0.5.",
            "Match 1 baseline is multicam_20s pack (old settings), different play.",
        ],
        "dashboard": "http://127.0.0.1:8080/match2-100row",
        "path_to_80": "PATH_TO_80.md",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md = [
        "# Match 2 gold 100 frames in a row test\n\n",
        f"Window: Match 2 t={meta['start_sec']}s, "
        f"{meta['num_frames']} consecutive frames. "
        f"Checkpoint: `{meta['checkpoint']}`.\n\n",
        "Detect floor 0.30. Green boxes = conf ≥ 0.8. "
        "Clear-ball size proxy = min side ≥ 25 px on full-res.\n\n",
        "**Dashboard:** run `python3 serve_viewer.py` then open "
        "[http://127.0.0.1:8080/match2-100row](http://127.0.0.1:8080/match2-100row)\n\n",
        "See also [PATH_TO_80.md](PATH_TO_80.md).\n\n",
        "## Match 2 (new settings, 8 cams)\n\n",
        markdown_table(m2_stats),
        "\n## Match 1 baseline (old settings, multicam_20s)\n\n",
        markdown_table(m1_stats) if m1_stats else "_skipped_\n",
        "\n## Artifacts\n\n",
        "- Per-cam: `<cam>/contact_10x10.jpg`, `<cam>/overlay.mp4`\n",
        "- Mosaic: `mosaic_contact.jpg`\n",
        "- Review pack: `data/processed/match2_gold_100_in_a_row/`\n",
        "- Raw: `summary.json`\n\n",
        "## Claim limit\n\n",
        "Visual size + raw detect/emit rates only. "
        "Not stratified Match Gold100; not true P_emit.\n",
    ]
    (out_dir / "summary.md").write_text("".join(md))


def mid_frame_thumb(annotated):
    if not annotated:
        return None
    return annotated[len(annotated) // 2]


def main():
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    review_dir = args.review_dir
    review_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ball_checkpoint)

    m2_stats = []
    m2_mids = []
    cam_manifest = []
    for name, path in MATCH2_CAMS:
        stats, rows, annotated, fps, frames = process_cam(
            model, name, path, "sec", args.start_sec, args.num_frames,
        )
        save_cam_outputs(out, name, annotated, rows, fps)
        entry = write_review_cam(
            review_dir, name, frames, rows, stats,
            {"source": str(path), "fps": fps},
        )
        cam_manifest.append(entry)
        m2_stats.append(stats)
        mid = mid_frame_thumb(annotated)
        if mid is not None:
            m2_mids.append(cv2.resize(mid, (640, 360)))

    mosaic = build_grid(m2_mids, cols=4)
    if mosaic is not None:
        cv2.imwrite(
            str(out / "mosaic_contact.jpg"),
            mosaic,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

    m1_stats = []
    if not args.skip_match1:
        for name, path in MATCH1_CAMS:
            if not path.is_file():
                print(f"skip missing Match1 clip: {path}")
                continue
            stats, rows, annotated, fps, _frames = process_cam(
                model, f"match1_{name}", path, "frame",
                args.match1_start_frame, args.num_frames,
            )
            save_cam_outputs(out, f"match1_{name}", annotated, rows, fps)
            m1_stats.append(stats)

    meta = {
        "start_sec": args.start_sec,
        "num_frames": args.num_frames,
        "checkpoint": args.ball_checkpoint,
        "detect_thr": DETECT_THR,
        "clear_min_side": CLEAR_MIN_SIDE,
        "match1_start_frame": args.match1_start_frame,
        "review_dir": str(review_dir),
    }
    write_summary(out, meta, m2_stats, m1_stats)

    slim_stats = [{k: v for k, v in s.items() if k != "max_conf_series"} for s in m2_stats]
    manifest = {
        "name": "match 2 gold 100 frames in a row",
        "start_sec": args.start_sec,
        "num_frames": args.num_frames,
        "checkpoint": args.ball_checkpoint,
        "detect_thr": DETECT_THR,
        "clear_min_side": CLEAR_MIN_SIDE,
        "cams": cam_manifest,
        "match2_stats": slim_stats,
    }
    (review_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (review_dir / "summary.json").write_text(json.dumps({
        "match2": slim_stats,
        "meta": meta,
    }, indent=2))
    print(f"\nWrote {out / 'summary.md'}")
    print(f"Review pack: {review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
