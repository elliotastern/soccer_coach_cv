#!/usr/bin/env python3
"""Extract synced Match-1 20s multi-cam pack and score per-cam vs oracle ball detect."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MATCH_DIR = ROOT / "data/raw/Match 1/Match 1 -1"
# Shared filename clock (/1e7 = seconds). Window = cam11 dedicated 20s clip.
WINDOW_START_S = 2040.0
WINDOW_DUR_S = 20.0
CAMS = {
    "cam8": {
        "video": MATCH_DIR / "cam 8/0000000019400000000.mp4",
        "file_start_s": 1940.0,
    },
    "cam9": {
        "video": MATCH_DIR / "cam 9/0000000019100000000.mp4",
        "file_start_s": 1910.0,
    },
    "cam11": {
        "video": MATCH_DIR / "cam 11/0000000020400000000.mp4",
        "file_start_s": 2040.0,
    },
    "cam13": {
        "video": MATCH_DIR / "cam 13/0000000019200000000.mp4",
        "file_start_s": 1920.0,
    },
}
DEFAULT_OUT = ROOT / "data/processed/multicam_20s_match1"
DEFAULT_CKPT = ROOT / "models/v6_snaps/epoch_110/checkpoint_best_regular.pth"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ball-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--fps-sample", type=float, default=2.0, help="synced sample rate")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8])
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def offset_s(cam: str) -> float:
    return WINDOW_START_S - CAMS[cam]["file_start_s"]


def extract_clip(cam: str, clips_dir: Path) -> Path:
    out = clips_dir / f"{cam}_20s.mp4"
    off = offset_s(cam)
    # -ss after -i for more accurate cut; reencode for sync-friendly playback
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(CAMS[cam]["video"]),
        "-ss", f"{off:.3f}", "-t", f"{WINDOW_DUR_S:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-an", str(out),
    ])
    return out


def sample_times(fps_sample: float) -> list[float]:
    n = int(round(WINDOW_DUR_S * fps_sample))
    return [i / fps_sample for i in range(n)]


def extract_frame_at(video: Path, t_s: float, out_path: Path) -> bool:
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(out_path),
    ])
    return out_path.is_file()


def extract_all(out_dir: Path, fps_sample: float) -> list[float]:
    clips_dir = out_dir / "clips"
    frames_dir = out_dir / "frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    times = sample_times(fps_sample)
    for cam in CAMS:
        clip = extract_clip(cam, clips_dir)
        cam_dir = frames_dir / cam
        cam_dir.mkdir(exist_ok=True)
        # Seek inside the cut clip so t=0 is shared window start
        for i, t in enumerate(times):
            extract_frame_at(clip, t, cam_dir / f"t{i:03d}_{t:05.2f}s.jpg")
        print(f"{cam}: wrote {len(times)} frames")
    return times


def load_detector(ckpt: Path, thr: float):
    from src.perception.rfdetr_local import load_ball_model
    from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler

    model = load_ball_model(str(ckpt))
    return BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=thr,
            use_sahi=False,
            use_size_filter=True,
            topk=2,
            use_kalman=False,
            min_side=4,
            max_side=120,
        ),
    )


def frame_path(out_dir: Path, cam: str, i: int, t: float) -> Path:
    return out_dir / "frames" / cam / f"t{i:03d}_{t:05.2f}s.jpg"


def infer_all(out_dir: Path, times: list[float], ckpt: Path, min_thr: float) -> dict:
    pre = load_detector(ckpt, min_thr)
    preds = {}
    for cam in CAMS:
        preds[cam] = {}
        for i, t in enumerate(times):
            path = frame_path(out_dir, cam, i, t)
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"missing frame {path}")
            dets = pre.detect_bgr(img)
            balls = [
                {
                    "bbox": [float(x) for x in d.bbox],
                    "confidence": float(d.confidence),
                    "side": float(min(d.bbox[2], d.bbox[3])),
                }
                for d in dets
            ]
            preds[cam][f"{i:03d}"] = balls
            top = max((b["confidence"] for b in balls), default=0.0)
            print(f"{cam} t={t:05.2f}s n={len(balls)} top={top:.3f}")
    return preds


def best_ball(balls: list[dict], thr: float) -> dict | None:
    ok = [b for b in balls if b["confidence"] >= thr]
    if not ok:
        return None
    return max(ok, key=lambda b: b["confidence"])


def score_presence(preds: dict, times: list[float], thr: float) -> dict:
    """Presence proxy (no GT): did we fire a ball box at this timestamp?"""
    per_cam = {cam: {"hit": 0, "n": len(times)} for cam in CAMS}
    oracle_hit = 0
    maxconf_hit = 0
    nearest_hit = 0  # largest ball side among cams with a det
    rows = []
    for i, t in enumerate(times):
        key = f"{i:03d}"
        chosen = {}
        for cam in CAMS:
            b = best_ball(preds[cam][key], thr)
            chosen[cam] = b
            if b is not None:
                per_cam[cam]["hit"] += 1
        any_hit = any(chosen[c] is not None for c in CAMS)
        if any_hit:
            oracle_hit += 1
        # max-conf merge
        cands = [(c, chosen[c]) for c in CAMS if chosen[c] is not None]
        if cands:
            maxconf_hit += 1
            near = max(cands, key=lambda x: x[1]["side"])
            nearest_hit += 1
            pick_max = max(cands, key=lambda x: x[1]["confidence"])
        else:
            near = None
            pick_max = None
        rows.append({
            "i": i,
            "t": t,
            "cams": {
                c: None if chosen[c] is None else {
                    "confidence": chosen[c]["confidence"],
                    "side": chosen[c]["side"],
                }
                for c in CAMS
            },
            "oracle": any_hit,
            "max_conf_cam": None if pick_max is None else pick_max[0],
            "nearest_cam": None if near is None else near[0],
        })
    n = len(times)
    return {
        "threshold": thr,
        "n_timestamps": n,
        "per_cam_recall_proxy": {
            cam: per_cam[cam]["hit"] / n for cam in CAMS
        },
        "per_cam_hits": {cam: per_cam[cam]["hit"] for cam in CAMS},
        "oracle_union_recall_proxy": oracle_hit / n,
        "oracle_hits": oracle_hit,
        "max_conf_hits": maxconf_hit,
        "nearest_side_hits": nearest_hit,
        "rows": rows,
    }


def write_contact_sheet(out_dir: Path, times: list[float], preds: dict, thr: float) -> Path:
    """4-up JPEG every 2s for sync + near-cam visual check."""
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    cam_order = list(CAMS.keys())
    thumb_w = 480
    for i, t in enumerate(times):
        if abs(t % 2.0) > 1e-6 and t != 0:
            continue
        tiles = []
        for cam in cam_order:
            img = cv2.imread(str(frame_path(out_dir, cam, i, t)))
            h, w = img.shape[:2]
            scale = thumb_w / w
            thumb = cv2.resize(img, (thumb_w, int(h * scale)))
            b = best_ball(preds[cam][f"{i:03d}"], thr)
            label = f"{cam}"
            if b:
                x, y, bw, bh = b["bbox"]
                cv2.rectangle(
                    thumb,
                    (int(x * scale), int(y * scale)),
                    (int((x + bw) * scale), int((y + bh) * scale)),
                    (0, 255, 0),
                    2,
                )
                label += f" {b['confidence']:.2f} s={b['side']:.0f}"
            else:
                label += " —"
            cv2.putText(
                thumb, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )
            tiles.append(thumb)
        row1 = np.hstack(tiles[:2])
        row2 = np.hstack(tiles[2:])
        sheet = np.vstack([row1, row2])
        path = sheet_dir / f"t{i:03d}_{t:05.2f}s_thr{thr:.1f}.jpg"
        cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return sheet_dir


def write_html(out_dir: Path, metrics: list[dict], sheet_dir: Path) -> Path:
    rows = []
    for m in metrics:
        cams = " | ".join(
            f"{c}={m['per_cam_hits'][c]}/{m['n_timestamps']}"
            for c in CAMS
        )
        rows.append(
            f"<tr><td>{m['threshold']}</td><td>{cams}</td>"
            f"<td><b>{m['oracle_hits']}/{m['n_timestamps']}</b> "
            f"({m['oracle_union_recall_proxy']:.3f})</td></tr>"
        )
    sheets = sorted(sheet_dir.glob("*.jpg"))
    imgs = "\n".join(
        f'<div><h3>{p.name}</h3><img src="contact_sheets/{p.name}" '
        f'style="max-width:100%"></div>'
        for p in sheets
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Multi-cam 20s ball probe</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#111;color:#eee}}
table{{border-collapse:collapse}} td,th{{border:1px solid #444;padding:8px}}
</style></head><body>
<h1>Match 1 — 20s × 4 cams (v6_epoch_110)</h1>
<p>Presence proxy only (no GT yet). Oracle = union of cams with a ball det.</p>
<table><tr><th>conf</th><th>per-cam hits</th><th>oracle union</th></tr>
{''.join(rows)}
</table>
<h2>Contact sheets (sync check)</h2>
{imgs}
</body></html>"""
    path = out_dir / "index.html"
    path.write_text(html)
    return path


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    times_path = out_dir / "timestamps.json"

    if args.skip_extract and times_path.is_file():
        times = json.loads(times_path.read_text())
    else:
        times = extract_all(out_dir, args.fps_sample)
        times_path.write_text(json.dumps(times, indent=2))

    manifest = {
        "window_start_s": WINDOW_START_S,
        "window_dur_s": WINDOW_DUR_S,
        "fps_sample": args.fps_sample,
        "cams": {
            cam: {
                "video": str(CAMS[cam]["video"].relative_to(ROOT)),
                "file_start_s": CAMS[cam]["file_start_s"],
                "offset_s": offset_s(cam),
                "clip": f"clips/{cam}_20s.mp4",
            }
            for cam in CAMS
        },
        "ball_checkpoint": str(args.ball_checkpoint.relative_to(ROOT)),
        "note": "Aligned by shared filename clock; cam9 is 30fps, others 60fps.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    preds_path = out_dir / "preds_ball.json"
    min_thr = min(args.thresholds)
    if args.skip_infer and preds_path.is_file():
        preds = json.loads(preds_path.read_text())
    else:
        preds = infer_all(out_dir, times, args.ball_checkpoint, min_thr)
        preds_path.write_text(json.dumps(preds, indent=2))

    metrics = [score_presence(preds, times, thr) for thr in args.thresholds]
    (out_dir / "metrics_presence.json").write_text(json.dumps(metrics, indent=2))
    sheet_dir = write_contact_sheet(out_dir, times, preds, thr=0.5)
    html = write_html(out_dir, metrics, sheet_dir)

    print("\n=== Multi-cam 20s presence proxy ===")
    for m in metrics:
        print(f"\nconf>={m['threshold']}")
        for cam in CAMS:
            print(
                f"  {cam}: {m['per_cam_hits'][cam]}/{m['n_timestamps']} "
                f"({m['per_cam_recall_proxy'][cam]:.3f})"
            )
        print(
            f"  ORACLE union: {m['oracle_hits']}/{m['n_timestamps']} "
            f"({m['oracle_union_recall_proxy']:.3f})"
        )
    print(f"\nViewer: {html}")
    print("NOTE: presence proxy ≠ AP50. Label this pack next for real oracle AP50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
