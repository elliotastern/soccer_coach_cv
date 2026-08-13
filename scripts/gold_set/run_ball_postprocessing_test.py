#!/usr/bin/env python3
"""P10 Top Left post-process gallery: 10 curated stacks on one Match 2 clip.

Window 0:26–0:31, camera P10 only. Ranking from prior train/gold studies
(not this clip). Writes reports/eval_match2_v10/ball_postprocessing_test/.
Never trains. Real Match 2 footage only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import det_tuple  # noqa: E402
from eval_v10_postprocess_ablation import detect_hflip_tta  # noqa: E402
from run_5x5_ball_clips import encode_browser_mp4, paint_view  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402
from src.perception.tracker import Tracker  # noqa: E402

CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
P10_VIDEO = ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"
QUAD_SRC = (
    ROOT
    / "reports/eval_match2_v10/4quad_test/source/quad_top_left_t00026.0s_P10.mp4"
)
OUT_DEFAULT = ROOT / "reports/eval_match2_v10/ball_postprocessing_test"
START_CLOCK = "0:26"
END_CLOCK = "0:31"
OVERLAY_WIDTH = 1280
SIZE = dict(use_size_filter=True, min_side=4, max_side=240)


def parse_clock(text: str) -> float:
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"expected m:ss, got {text}")
    return int(parts[0]) * 60 + float(parts[1])


def variant_specs() -> list[dict]:
    return [
        {
            "id": "baseline_topk2",
            "title": "1. Baseline topk=2",
            "why": "Locked winner on train/gold (thr 0.3 + size + NMS + topk=2).",
            "mode": "detect",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, use_kalman=False, **SIZE),
        },
        {
            "id": "topk3",
            "title": "2. Topk=3",
            "why": "Same stack; gold peek found +1 TP at same FPs.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=3, use_kalman=False, **SIZE),
        },
        {
            "id": "hflip_tta",
            "title": "3. HFlip TTA + NMS",
            "why": "Train raised recall with more FPs; gold @0.8 published more boxes.",
            "mode": "tta",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=99, use_kalman=False, **SIZE),
        },
        {
            "id": "multiscale_1p5",
            "title": "4. Multiscale 1.5×",
            "why": "Train +1 TP +1 FP; gold no gain — included as recover candidate.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(
                threshold=0.30,
                use_sahi=False,
                use_multiscale=True,
                topk=2,
                use_kalman=False,
                **SIZE,
            ),
        },
        {
            "id": "sahi_fallback",
            "title": "5. SAHI fallback-only",
            "why": "Tiles only if fullframe empty; identical to baseline when a det exists.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(
                threshold=0.30,
                use_sahi=True,
                sahi_fallback_only=True,
                sahi_recover_only=True,
                topk=2,
                use_kalman=False,
                **SIZE,
            ),
        },
        {
            "id": "thr50_topk2",
            "title": "6. Thr 0.5 topk=2",
            "why": "Stricter detect floor — fewer weak RAWs, may miss hard balls.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(threshold=0.50, use_sahi=False, topk=2, use_kalman=False, **SIZE),
        },
        {
            "id": "emit80_pass",
            "title": "7. Emit ≥ 0.80 only",
            "why": "Publish gate on baseline detect (precision-only; not more balls).",
            "mode": "emit80",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, use_kalman=False, **SIZE),
        },
        {
            "id": "bytetrack_iou08",
            "title": "8. ByteTrack IoU 0.8",
            "why": "Least-harmful ByteTrack on train/gold; still drops some balls vs detector.",
            "mode": "bytetrack",
            "emit_gate": False,
            "match_thresh": 0.8,
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, use_kalman=False, **SIZE),
        },
        {
            "id": "bytetrack_emit80",
            "title": "9. ByteTrack + emit 0.80",
            "why": "Current product publish path (high precision, low recall).",
            "mode": "bytetrack",
            "emit_gate": True,
            "match_thresh": 0.8,
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, use_kalman=False, **SIZE),
        },
        {
            "id": "kalman_detect",
            "title": "10. Kalman as detect",
            "why": "Known weaker on gold (R~0.71) — contrast against detector-only.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, use_kalman=True, **SIZE),
        },
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--overlay-width", type=int, default=OVERLAY_WIDTH)
    p.add_argument("--skip-extract", action="store_true")
    return p.parse_args()


def ensure_source(out: Path, skip_extract: bool) -> Path:
    dest = out / "source" / "top_left_p10.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_extract and dest.is_file() and dest.stat().st_size > 1000:
        return dest
    if QUAD_SRC.is_file() and QUAD_SRC.stat().st_size > 1000:
        shutil.copy2(QUAD_SRC, dest)
        return dest
    start = parse_clock(START_CLOCK)
    end = parse_clock(END_CLOCK)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(P10_VIDEO),
        "-t",
        f"{end - start:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    if not dest.is_file():
        raise RuntimeError(f"ffmpeg did not write {dest}")
    return dest


def top_pred(dets) -> tuple | None:
    if not dets:
        return None
    rows = [det_tuple(d) for d in dets]
    rows.sort(key=lambda x: -x[1])
    return rows[0]


def detect_frame(pre: BallPrelabeler, frame, mode: str):
    if mode == "tta":
        return detect_hflip_tta(pre, frame)
    return pre.detect_bgr(frame)


def make_tracker(emit_gate: bool, match_thresh: float) -> Tracker:
    return Tracker(
        track_thresh=0.10,
        emit_thresh=0.80,
        ema_alpha=0.3,
        match_thresh=match_thresh,
        apply_emit_gate=emit_gate,
        frame_rate=30,
    )


def run_variant(model, src: Path, ov_path: Path, spec: dict, stride: int, width: int) -> dict:
    pre = BallPrelabeler(model, spec["cfg"], class_id=1)
    tracker = None
    if spec["mode"] == "bytetrack":
        tracker = make_tracker(spec.get("emit_gate", True), spec.get("match_thresh", 0.8))
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = None
    n_frames = 0
    n_raw = 0
    n_emit = 0
    confs = []
    last_raw = None
    last_emit = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        if (n_frames - 1) % max(1, stride) == 0:
            dets = detect_frame(pre, frame, spec["mode"] if spec["mode"] != "emit80" else "detect")
            raw = top_pred(dets)
            emit = None
            if spec["mode"] == "emit80":
                emit = raw if raw is not None and raw[1] >= 0.80 else None
                raw = emit
            elif tracker is not None:
                tracked = tracker.update(dets, frame)
                emit_rows = [det_tuple(o.detection) for o in tracked if o.detection.class_name == "ball"]
                emit_rows.sort(key=lambda x: -x[1])
                emit = emit_rows[0] if emit_rows else None
                if not spec.get("emit_gate", True):
                    # no gate: show tracked as the drawn box (treat as emit for green)
                    pass
            else:
                emit = raw if raw is not None and raw[1] >= 0.80 else None
            last_raw = raw
            last_emit = emit
        if last_raw is not None:
            n_raw += 1
        if last_emit is not None:
            n_emit += 1
            confs.append(float(last_emit[1]))
        draw_raw = last_raw
        draw_emit = last_emit
        if spec["mode"] == "emit80":
            draw_raw = None
            draw_emit = last_emit
        elif tracker is not None and not spec.get("emit_gate", True):
            draw_raw = last_raw
            draw_emit = last_emit  # tracked box green when present
        vis = paint_view(frame, draw_raw, draw_emit, "P10", width)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(ov_path), fourcc, fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        top = last_emit[1] if last_emit is not None else (last_raw[1] if last_raw else 0.0)
        print(
            f"  {spec['id']} f={n_frames:03d} raw={last_raw is not None} "
            f"emit={last_emit is not None} top={top:.3f}",
            flush=True,
        )
    cap.release()
    if writer is not None:
        writer.release()
        encode_browser_mp4(ov_path)
    return {
        "id": spec["id"],
        "title": spec["title"],
        "why": spec["why"],
        "mode": spec["mode"],
        "n_frames": n_frames,
        "n_raw_hits": n_raw,
        "n_emit_hold": n_emit,
        "emit_rate": (n_emit / n_frames) if n_frames else 0.0,
        "raw_rate": (n_raw / n_frames) if n_frames else 0.0,
        "mean_emit_conf": (sum(confs) / len(confs)) if confs else None,
        "overlay": f"overlay/{ov_path.name}",
    }


def html_esc(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_mean(value):
    return f"{value:.3f}" if value is not None else "—"


def write_html(out: Path, payload: dict) -> Path:
    cards = []
    for clip in payload["variants"]:
        cards.append(
            f"""
<section class="clip" id="{html_esc(clip['id'])}">
<h2>{html_esc(clip['title'])}</h2>
<p class="meta">{html_esc(payload['clock'])} · P10 · raw {clip['n_raw_hits']}/{clip['n_frames']}
({clip['raw_rate']:.1%}) · emit hold {clip['n_emit_hold']}/{clip['n_frames']}
({clip['emit_rate']:.1%}) · mean emit conf {html_esc(fmt_mean(clip.get('mean_emit_conf')))}</p>
<p class="why">{html_esc(clip['why'])}</p>
<p class="box-legend"><span class="emit">green EMIT</span> = published / tracked ·
<span class="raw">orange RAW</span> = detect below gate · marker on the ball</p>
<video controls playsinline type="video/mp4" src="{html_esc(clip['overlay'])}"></video>
</section>
"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ball_postprocessing_test</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #0b1220; color: #e8eefc; margin: 0; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
.sub, .why, .meta, .box-legend {{ color: #9db0d0; font-size: 14px; margin: 0 0 10px; }}
.why {{ color: #c5d3ea; }}
.box-legend .emit {{ color: #4ade80; }}
.box-legend .raw {{ color: #fb923c; }}
.clip {{ background: #151d2e; border-radius: 16px; padding: 18px; margin: 18px 0; }}
video {{ width: 100%; border-radius: 10px; background: #000; }}
.note {{ color: #9db0d0; font-size: 13px; margin-bottom: 20px; max-width: 72ch; }}
</style>
</head>
<body>
<main>
<h1>ball_postprocessing_test</h1>
<p class="sub">Top Left {html_esc(payload['clock'])} · camera P10 · detect floor from each variant ·
checkpoint {html_esc(payload['checkpoint'])}</p>
<p class="note">Ranking is from prior Match 2 train/gold postprocess + track studies, not this 5s window.
ByteTrack/Kalman are shown for contrast — they did not beat detector-only on gold.</p>
{"".join(cards)}
</main>
</body>
</html>
"""
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_summary(out: Path, payload: dict) -> Path:
    path = out / "summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# ball_postprocessing_test",
        "",
        f"Top Left `{payload['clock']}` · P10 · ranking from prior train/gold studies.",
        "",
        "| # | id | raw rate | emit hold | mean emit |",
        "|---|---|---:|---:|---:|",
    ]
    for i, row in enumerate(payload["variants"], start=1):
        lines.append(
            f"| {i} | `{row['id']}` | {row['raw_rate']:.1%} | {row['emit_rate']:.1%} | "
            f"{fmt_mean(row.get('mean_emit_conf'))} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    src = ensure_source(out, args.skip_extract)
    print(f"source {src}", flush=True)
    if not args.ball_checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint {args.ball_checkpoint}")
    model = load_ball_model(str(args.ball_checkpoint))
    specs = variant_specs()
    if len(specs) != 10:
        raise RuntimeError(f"expected 10 variants, got {len(specs)}")
    variants = []
    for spec in specs:
        ov = out / "overlay" / f"{spec['id']}.mp4"
        print(f"variant {spec['id']} → {ov}", flush=True)
        variants.append(
            run_variant(model, src, ov, spec, args.stride, args.overlay_width)
        )
    payload = {
        "title": "ball_postprocessing_test",
        "clock": f"{START_CLOCK}–{END_CLOCK}",
        "start_sec": parse_clock(START_CLOCK),
        "end_sec": parse_clock(END_CLOCK),
        "camera": "P10",
        "video_path": str(P10_VIDEO),
        "source": str(src),
        "checkpoint": str(args.ball_checkpoint),
        "stride": args.stride,
        "ranking_note": "from postprocess_ablation.md + track_tune.md, not this clip",
        "variants": variants,
    }
    summary = write_summary(out, payload)
    write_html(out, payload)
    print(f"wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
