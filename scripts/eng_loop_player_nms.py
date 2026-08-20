#!/usr/bin/env python3
"""Eng-loop: verify coach UX = one player box after NMS on real Match 3 frames.

Gate: score ≥ 9.0/10. Writes before/after overlays + score JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import (  # noqa: E402
    LocalRFDETRDetector,
    _is_duplicate_player,
    nms_by_class,
)
from src.state.types import Detection  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_nms"
VIDEO = ROOT / "data/raw/Match 3/P10-002.mp4"
PLAYER_CKPT = ROOT / "models/people_after_100_epochs.pth"
BALL_CKPT = ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"
PASS = 9.0
# Frames around soft-open / mid sample where duplicates showed up in review
FRAME_IDS = [900, 1200, 1800, 2100, 2400, 3000, 3600, 4200, 4800, 5400]
THR = 0.15  # same low thr as Streamlit verify (worst case for duplicates)


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def count_overlap_pairs(players: list, thr: float) -> int:
    n = 0
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            if _is_duplicate_player(players[i].bbox, players[j].bbox, thr):
                n += 1
    return n


def draw_players(frame, dets, color, title: str):
    vis = frame.copy()
    for d in dets:
        x, y, w, h = [int(v) for v in d.bbox]
        cv2.rectangle(vis, (x, y), (x + max(1, w), y + max(1, h)), color, 3)
        lab = f"P {float(d.confidence):.2f}"
        cv2.putText(vis, lab, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis


def score_synthetic(overlap_thr: float) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    tall = Detection(0, 0.74, (100, 80, 50, 160), "player")
    short = Detection(0, 0.52, (105, 90, 45, 90), "player")
    far = Detection(0, 0.60, (400, 100, 50, 150), "player")
    body = Detection(0, 0.56, (1214.6, 94.2, 64.9, 139.8), "player")
    head = Detection(0, 0.55, (1214.6, -1.6, 51.4, 145.8), "player")
    left = Detection(0, 0.70, (200, 100, 40, 120), "player")
    right = Detection(0, 0.68, (230, 105, 40, 120), "player")
    out = nms_by_class([tall, short, far], player_iou=overlap_thr)
    if len(out) != 2:
        score -= 5.0
        notes.append(f"nested keep={len(out)} want 2")
    if any(abs(d.confidence - 0.52) < 1e-6 for d in out):
        score -= 4.0
        notes.append("kept weak nested box")
    stack = nms_by_class([body, head], player_iou=overlap_thr)
    if len(stack) != 1:
        score -= 5.0
        notes.append(f"vertical stack keep={len(stack)} want 1")
    adj = nms_by_class([left, right], player_iou=overlap_thr)
    if (not _is_duplicate_player(left.bbox, right.bbox, overlap_thr)) and len(adj) != 2:
        score -= 4.0
        notes.append("merged adjacent players")
    if not notes:
        notes.append("synthetic ok")
    return clamp(score), notes


def score_frames(rows: list, overlap_thr: float) -> tuple[float, list[str], dict]:
    notes = []
    with_players = [r for r in rows if r["n_raw"] > 0]
    if not with_players:
        return 0.0, ["no player dets"], {}
    clean = sum(1 for r in with_players if r["n_overlap_after"] == 0)
    frac = clean / len(with_players)
    # Also require NMS removed something when raw had overlaps
    raw_bad = [r for r in with_players if r["n_overlap_before"] > 0]
    fixed = sum(1 for r in raw_bad if r["n_overlap_after"] == 0)
    fix_frac = (fixed / len(raw_bad)) if raw_bad else 1.0
    # Preserve at least one player when raw had any
    emptied = sum(1 for r in with_players if r["n_after"] == 0)
    score = 10.0 * frac
    if fix_frac < 1.0 and raw_bad:
        score = min(score, 10.0 * fix_frac)
    if emptied:
        score -= min(3.0, 1.0 * emptied)
        notes.append(f"emptied {emptied} frames")
    if frac < 1.0:
        bad = [r["frame"] for r in with_players if r["n_overlap_after"] > 0]
        notes.append(f"still overlapping frames={bad}")
    else:
        notes.append(f"all {len(with_players)} frames clean after NMS")
    notes.append(f"raw_overlap_frames={len(raw_bad)} fixed={fixed}")
    meta = {
        "frames_with_players": len(with_players),
        "clean_after": clean,
        "frac_clean": round(frac, 3),
        "raw_overlap_frames": len(raw_bad),
        "fixed_overlap_frames": fixed,
        "fix_frac": round(fix_frac, 3),
        "overlap_thr": overlap_thr,
    }
    return clamp(score), notes, meta


def read_frame(video: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_id}")
    return frame


def run_once(overlap_thr: float) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)
    det = LocalRFDETRDetector(
        player_checkpoint=str(PLAYER_CKPT),
        ball_checkpoint=str(BALL_CKPT),
        confidence_threshold=THR,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=overlap_thr,
        ball_nms_iou=0.4,
    )
    rows = []
    for fid in FRAME_IDS:
        frame = read_frame(VIDEO, fid)
        # Bypass detector NMS: raw people only, then apply thr under test
        from src.perception.rfdetr_local import _frame_to_pil, _parse_rfdetr_detections

        raw = det.people_model.predict(_frame_to_pil(frame), threshold=THR)
        players_raw = _parse_rfdetr_detections(raw, 0, "player")
        after = nms_by_class(players_raw, player_iou=overlap_thr)
        players_after = [d for d in after if d.class_name == "player"]
        n_before = count_overlap_pairs(players_raw, overlap_thr)
        n_after = count_overlap_pairs(players_after, overlap_thr)
        row = {
            "frame": fid,
            "n_raw": len(players_raw),
            "n_after": len(players_after),
            "n_overlap_before": n_before,
            "n_overlap_after": n_after,
        }
        rows.append(row)
        before_img = draw_players(frame, players_raw, (0, 0, 255), f"BEFORE n={len(players_raw)} ov={n_before}")
        after_img = draw_players(
            frame, players_after, (0, 220, 0), f"AFTER n={len(players_after)} ov={n_after}"
        )
        pair = np.hstack([before_img, after_img])
        scale = 1280 / max(1, pair.shape[1])
        if scale < 1:
            pair = cv2.resize(pair, None, fx=scale, fy=scale)
        cv2.imwrite(str(OUT / "overlays" / f"f{fid:05d}_before_after.jpg"), pair)
        print(
            f"frame={fid} raw={len(players_raw)} after={len(players_after)} "
            f"ov {n_before}->{n_after}"
        )

    syn_s, syn_n = score_synthetic(overlap_thr)
    fr_s, fr_n, meta = score_frames(rows, overlap_thr)
    overall = clamp(min(syn_s, fr_s))
    summary = {
        "score": overall,
        "pass": overall >= PASS,
        "gate": PASS,
        "synthetic_score": syn_s,
        "synthetic_notes": syn_n,
        "frame_score": fr_s,
        "frame_notes": fr_n,
        "meta": meta,
        "rows": rows,
        "thr_detect": THR,
        "overlap_thr": overlap_thr,
        "overlays": str(OUT / "overlays"),
    }
    (OUT / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PLAYER_NMS_SCORE {overall}/10 gate={'PASS' if overall >= PASS else 'FAIL'}")
    print(json.dumps({"syn": syn_s, "frames": fr_s, "meta": meta}, indent=2))
    return summary


def main():
    # Eng-loop: tighten overlap thr until gate passes (or floor)
    floors = [0.35, 0.30, 0.25]
    last = None
    thr = floors[0]
    for thr in floors:
        print(f"\n=== eng-loop trial overlap_thr={thr} ===")
        last = run_once(thr)
        if last["pass"]:
            break
    if last and last["pass"]:
        cfg = ROOT / "configs/default.yaml"
        text = cfg.read_text(encoding="utf-8")
        import re

        text2, n = re.subn(
            r"player_nms_iou:\s*[0-9.]+",
            f"player_nms_iou: {thr}",
            text,
            count=1,
        )
        if n:
            cfg.write_text(text2, encoding="utf-8")
            print(f"locked {cfg} player_nms_iou={thr}")
        (OUT / "PASS").write_text(f"thr={thr} score={last['score']}\n", encoding="utf-8")
    else:
        (OUT / "FAIL").write_text(json.dumps(last, indent=2), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
