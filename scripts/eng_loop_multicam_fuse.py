#!/usr/bin/env python3
"""Eng-loop: multi-cam Pitch 1 fuse stability (gate ≥ 9.0)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.multicam_fuse import (  # noqa: E402
    PLAYER_MERGE_M,
    discover_cam_frame_csvs,
    fuse_ball_at_frame,
    fuse_frame_for_pitch,
    fuse_players_at_frame,
    load_cam_tables,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/multicam_fuse"
PASS = 9.0
OUTPUT_ROOT = ROOT / "data/output/full_match_2min"


def clamp(s: float) -> float:
    return round(max(0.0, min(10.0, s)), 1)


def score_discover() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    cams = discover_cam_frame_csvs(OUTPUT_ROOT)
    if len(cams) < 1:
        return 0.0, [f"no frame_data under {OUTPUT_ROOT}"]
    if len(cams) < 2:
        score -= 4.0
        notes.append(f"only {len(cams)} cam(s)={list(cams)} — need ≥2 for fuse")
    else:
        notes.append(f"cams={sorted(cams)}")
    return clamp(score), notes


def score_unit_merge() -> tuple[float, list[str]]:
    """Synthetic two-cam same player must collapse to 1."""
    notes = []
    score = 10.0
    rows_a = pd.DataFrame(
        [
            {"frame_id": 100, "Player_ID": 1, "Team_ID": 0, "Location_X": 0.0, "Location_Y": 0.0, "confidence": 0.9},
            {"frame_id": 100, "Player_ID": -1, "Team_ID": -1, "Location_X": 5.0, "Location_Y": 1.0, "confidence": 0.85},
        ]
    )
    rows_b = pd.DataFrame(
        [
            {"frame_id": 100, "Player_ID": 99, "Team_ID": 0, "Location_X": 0.4, "Location_Y": 0.2, "confidence": 0.8},
            {"frame_id": 100, "Player_ID": 7, "Team_ID": 1, "Location_X": -10.0, "Location_Y": 8.0, "confidence": 0.7},
            {"frame_id": 100, "Player_ID": -1, "Team_ID": -1, "Location_X": 5.2, "Location_Y": 1.1, "confidence": 0.82},
        ]
    )
    tables = {"P10": rows_a, "P1": rows_b}
    players = fuse_players_at_frame(tables, 100, merge_m=PLAYER_MERGE_M)
    ball = fuse_ball_at_frame(tables, 100)
    if len(players) != 2:
        score -= 5.0
        notes.append(f"player fuse count={len(players)} want 2 (near dup + far)")
    if ball is None:
        score -= 4.0
        notes.append("ball fuse None")
    else:
        if abs(ball[0] - 5.1) > 1.0:
            score -= 2.0
            notes.append(f"ball xy unexpected {ball}")
    # far players must not merge
    far = [(p[0], p[1]) for p in players]
    if any(_near(far[i], far[j]) for i in range(len(far)) for j in range(i + 1, len(far))):
        # if only 2 and one is near origin one far - ok if not near each other
        if len(players) == 2 and _near(far[0], far[1]):
            score -= 4.0
            notes.append("merged distant players")
    if not notes:
        notes.append("unit merge ok")
    return clamp(score), notes


def _near(a, b, thr=PLAYER_MERGE_M):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= thr


def score_live_frames() -> tuple[float, list[str], dict]:
    notes = []
    cams = discover_cam_frame_csvs(OUTPUT_ROOT)
    if len(cams) < 2:
        return 5.0, ["skip live: <2 cams"], {"n_cams": len(cams)}
    tables = load_cam_tables(cams)
    # frames present in ≥2 cams
    common = None
    for df in tables.values():
        ids = set(int(x) for x in df["frame_id"].unique())
        common = ids if common is None else (common & ids)
    if not common:
        return 4.0, ["no overlapping frame_ids"], {"n_cams": len(cams)}
    sample = sorted(common)
    # take up to 15 mid frames
    mid = sample[len(sample) // 3 : len(sample) // 3 + 15]
    if not mid:
        mid = sample[:15]
    lifts = []
    balls = 0
    for fid in mid:
        fused = fuse_frame_for_pitch(OUTPUT_ROOT, fid)
        # single-cam counts
        singles = []
        for cam, df in tables.items():
            sub = df[(df["frame_id"] == fid) & (df["Player_ID"] != -1)]
            singles.append(len(sub))
        n_single_max = max(singles) if singles else 0
        n_fused = len(fused["players"])
        lifts.append(n_fused - n_single_max)
        if fused["ball_xy"] is not None:
            balls += 1
        # fused should be ≥ max single (coverage) and not explode > sum
        n_sum = sum(singles)
        if n_fused > n_sum:
            notes.append(f"frame {fid} fused>{n_sum}")
        if n_fused < n_single_max:
            notes.append(f"frame {fid} fused<{n_single_max}")
    mean_lift = sum(lifts) / max(1, len(lifts))
    ball_frac = balls / max(1, len(mid))
    score = 10.0
    if mean_lift < 0:
        score -= 4.0
        notes.append(f"mean_lift={mean_lift:.2f} (lost players)")
    elif mean_lift < 0.5:
        score -= 1.5
        notes.append(f"mean_lift={mean_lift:.2f} (little multi-cam gain)")
    else:
        notes.append(f"mean_lift={mean_lift:.2f}")
    if ball_frac < 0.3:
        score -= 3.0
        notes.append(f"ball_frac={ball_frac:.2f}")
    else:
        notes.append(f"ball_frac={ball_frac:.2f}")
    if any("fused>" in n for n in notes):
        score -= 2.0
    meta = {
        "n_cams": len(cams),
        "frames": mid,
        "mean_lift": round(mean_lift, 3),
        "ball_frac": round(ball_frac, 3),
        "cams": sorted(cams),
    }
    return clamp(score), notes, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores, notes = {}, {}
    scores["discover"], notes["discover"] = score_discover()
    scores["unit"], notes["unit"] = score_unit_merge()
    scores["live"], notes["live"], meta = score_live_frames()
    overall = clamp(min(scores.values()))
    summary = {
        "score": overall,
        "pass": overall >= PASS,
        "gate": PASS,
        "scores": scores,
        "notes": notes,
        "meta": meta,
    }
    (OUT / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"MULTICAM_FUSE_SCORE {overall}/10 gate={'PASS' if overall >= PASS else 'FAIL'}")
    if overall < PASS:
        sys.exit(1)


if __name__ == "__main__":
    main()
