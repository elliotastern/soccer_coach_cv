#!/usr/bin/env python3
"""Build product-fuse Pitch 1 timeline for phase1_check window → events score.

Same wiring as render_phase1_check_mosaic: defish tiles, apply_undistort=False.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.events.events import EventDetector  # noqa: E402
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import mosaic_quads_coach, match3_videos  # noqa: E402
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402
from src.state.types import Ball, FrameData, Player  # noqa: E402

OUT_DEFAULT = (
    ROOT / "data/processed/gold_sets/match3_events_v1/clips/check25_human"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2390)
    p.add_argument("--match-sec", type=float, default=25.0)
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    return p.parse_args()


def run_events_on_timeline(timeline: dict) -> list[dict]:
    det = EventDetector()
    frames = []
    for row in timeline["frames"]:
        players = [
            Player(int(p[0]), int(p[3]) if len(p) > 3 else 0, float(p[1]), float(p[2]),
                   (0, 0, 10, 10), int(row["frame_id"]), float(row["t"]))
            for p in (row.get("players") or [])
        ]
        ball = None
        if row.get("ball") is not None:
            bx, by = row["ball"]
            ball = Ball(float(bx), float(by), (0, 0, 4, 4),
                        int(row["frame_id"]), float(row["t"]))
        frames.append(FrameData(int(row["frame_id"]), float(row["t"]), players, ball))
    emits = []
    prev = None
    for fr in frames:
        for ev in det.detect_events(fr, prev):
            emits.append(
                {
                    "type": ev.type.value,
                    "frame_end": ev.end_frame,
                    "t_start": round(ev.timestamp_start, 4),
                    "t_end": round(ev.timestamp_end, 4),
                    "confidence": round(ev.confidence, 4),
                    "players": list(ev.involved_players),
                    "start_xy": [ev.start_location.x, ev.start_location.y],
                    "end_xy": [ev.end_location.x, ev.end_location.y],
                }
            )
        prev = fr
    return emits


def score_windows(gold_events: list, emits: list) -> dict:
    """Window match: emit t_end inside gold [t_start, t_end] same type."""
    used_g = set()
    tp = 0
    for em in emits:
        hit = False
        for i, g in enumerate(gold_events):
            if i in used_g or g["type"] != em["type"]:
                continue
            te = float(em["t_end"])
            if float(g["t_start"]) - 0.25 <= te <= float(g["t_end"]) + 0.25:
                used_g.add(i)
                hit = True
                break
        if hit:
            tp += 1
    fp = len(emits) - tp
    fn = len(gold_events) - len(used_g)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1) if gold_events else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "p_emit": round(p, 4),
        "recall": round(r, 4),
        "n_emit": len(emits),
        "n_gold": len(gold_events),
        "matched_gold_idx": sorted(used_g),
    }


def main() -> int:
    args = parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    n_match = int(round(args.match_sec * args.src_fps))
    end = args.start + n_match - 1
    frame_ids = list(range(args.start, end + 1, args.stride))
    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    sess = TeamSession()
    apply_defish = True
    apply_undistort = not apply_defish
    rows = []
    print(f"timeline n={len(frame_ids)} stride={args.stride}", flush=True)
    for i, fr in enumerate(frame_ids):
        bag = {}
        mosaic_quads_coach(
            vids,
            fr,
            tile_w=480,
            tile_h=270,
            dets_by_cam=bag,
            detect_fn=detect_fn,
            apply_defish=apply_defish,
        )
        live = fuse_live_dets_for_pitch(
            bag, apply_undistort=apply_undistort, team_session=sess
        )
        t_s = (fr - args.start) / args.src_fps
        players = []
        for p in live["players"]:
            x, y = float(p[0]), float(p[1])
            team = int(p[2]) if len(p) > 2 else -1
            pid = int(p[3]) if len(p) > 3 else 0
            players.append([pid, x, y, team])
        ball = live["ball_xy"]
        rows.append(
            {
                "frame_id": fr,
                "t": round(t_s, 4),
                "ball": [round(ball[0], 4), round(ball[1], 4)] if ball else None,
                "players": players,
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i + 1}/{len(frame_ids)} fr={fr} ball={ball is not None}", flush=True)

    timeline = {
        "start_frame": args.start,
        "src_fps": args.src_fps,
        "stride": args.stride,
        "match_sec": args.match_sec,
        "apply_defish": apply_defish,
        "apply_undistort": apply_undistort,
        "note": "Product fuse timeline for event eng-loop (real Match 3)",
        "frames": rows,
    }
    tl_path = out / "timeline.json"
    tl_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print("WROTE", tl_path, flush=True)

    emits = run_events_on_timeline(timeline)
    (out / "emits.json").write_text(
        json.dumps({"emits": emits}, indent=2), encoding="utf-8"
    )
    labels_path = out / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {"events": []}
    score = score_windows(labels.get("events") or [], emits)
    score["path"] = str(tl_path.relative_to(ROOT))
    (out / "score_real.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    print("SCORE", json.dumps(score), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
