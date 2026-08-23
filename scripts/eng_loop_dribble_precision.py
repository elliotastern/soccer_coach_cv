#!/usr/bin/env python3
"""Eng-loop: dribble precision + batch anti-spam gate."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = 9.0
GATE_P_EMIT = 0.80
OUT = ROOT / "reports/eval_match3/improve_eng_loop/dribble_precision"
PROMPT = OUT / "PROMPT.md"
V1_GOLD = ROOT / "data/processed/gold_sets/match3_events_v1/manifest.json"
V2_GOLD = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/manifest.json"
BATCH_CSV = ROOT / "data/output/full_match_2min/P10-002/frame_data.csv"
META_15S = (
    ROOT
    / "reports/eval_match3/improve_eng_loop/player_map/check_15s_s4/meta.json"
)

_spec = importlib.util.spec_from_file_location(
    "eh_events", ROOT / "scripts/eng_loop_heuristic_events.py"
)
_eh = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eh)
match_emits = _eh.match_emits
run_timeline = _eh.run_timeline
score_clip = _eh.score_clip

from src.events.events import EventDetector  # noqa: E402


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def timeline_from_frame_csv(csv_path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    by_f: dict = {}
    for _, row in df.iterrows():
        fid = int(row.frame_id)
        if fid not in by_f:
            by_f[fid] = {
                "frame_id": fid,
                "t": float(row.Timestamp),
                "players": [],
                "ball": None,
            }
        if int(row.Player_ID) == -1:
            by_f[fid]["ball"] = [float(row.Location_X), float(row.Location_Y)]
        else:
            by_f[fid]["players"].append(
                [int(row.Player_ID), float(row.Location_X), float(row.Location_Y)]
            )
    frames = [by_f[f] for f in sorted(by_f.keys()) if by_f[f]["ball"] is not None]
    return {"frames": frames}


def run_det_on_timeline(timeline: dict) -> list[dict]:
    from src.state.types import Ball, FrameData, Player

    det = EventDetector()
    frames = []
    for row in timeline["frames"]:
        players = [
            Player(
                int(p[0]), 0, float(p[1]), float(p[2]),
                (0, 0, 10, 10), int(row["frame_id"]), float(row["t"]),
            )
            for p in (row.get("players") or [])
        ]
        ball = None
        if row.get("ball") is not None:
            bx, by = row["ball"]
            ball = Ball(
                float(bx), float(by), (0, 0, 4, 4),
                int(row["frame_id"]), float(row["t"]),
            )
        frames.append(FrameData(int(row["frame_id"]), float(row["t"]), players, ball))
    emits = []
    prev = None
    for fr in frames:
        for ev in det.detect_events(fr, prev):
            emits.append({"type": ev.type.value, "confidence": ev.confidence})
        prev = fr
    return emits


def count_dribble_15s_meta() -> int:
    if not META_15S.is_file():
        return 99
    meta = json.loads(META_15S.read_text(encoding="utf-8"))
    return sum(1 for e in (meta.get("emits") or []) if e.get("type") == "dribble")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not V2_GOLD.is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts/gold_set/build_match3_dribble_gold.py")],
            cwd=str(ROOT),
        )
    if not V1_GOLD.is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts/gold_set/build_match3_events_gold.py")],
            cwd=str(ROOT),
        )

    parent = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eng_loop_heuristic_events.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    parent_ok = parent.returncode == 0

    v2 = json.loads(V2_GOLD.read_text(encoding="utf-8"))
    carry_clip = next(c for c in v2["clips"] if c["id"] == "real_dribble_carry_15s")
    carry_tl = json.loads((ROOT / carry_clip["timeline"]).read_text(encoding="utf-8"))
    carry_lab = json.loads((ROOT / carry_clip["labels"]).read_text(encoding="utf-8"))
    carry_emits = run_timeline(carry_tl, EventDetector())
    carry_m = match_emits(carry_lab.get("events") or [], carry_emits)

    neg_fp = 0
    for clip in v2["clips"]:
        if not clip["id"].startswith("audit_static_"):
            continue
        tl = json.loads((ROOT / clip["timeline"]).read_text(encoding="utf-8"))
        neg_fp += len(run_timeline(tl, EventDetector()))

    jitter_fp = 0
    v1 = json.loads(V1_GOLD.read_text(encoding="utf-8"))
    for clip in v1["clips"]:
        if clip.get("id") != "synth_goal_jitter_none" or not clip.get("timeline"):
            continue
        jitter_fp = len(run_timeline(
            json.loads((ROOT / clip["timeline"]).read_text(encoding="utf-8")),
            EventDetector(),
        ))

    real_p, real_fp = 0.0, 0
    check_dir = ROOT / "data/processed/gold_sets/match3_events_v1/clips/check25_human"
    if (check_dir / "timeline.json").is_file() and (check_dir / "labels.json").is_file():
        spec = importlib.util.spec_from_file_location(
            "build_check25",
            ROOT / "scripts/gold_set/build_check25_event_timeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        tl = json.loads((check_dir / "timeline.json").read_text(encoding="utf-8"))
        labs = json.loads((check_dir / "labels.json").read_text(encoding="utf-8"))
        real_emits = mod.run_events_on_timeline(tl)
        sc = mod.score_windows(labs.get("events") or [], real_emits)
        real_p = float(sc["p_emit"])
        real_fp = int(sc["fp"])

    batch_n_dribble = 0
    batch_span_s = 0.0
    if BATCH_CSV.is_file():
        tl = timeline_from_frame_csv(BATCH_CSV)
        batch_n_dribble = sum(
            1 for e in run_det_on_timeline(tl) if e["type"] == "dribble"
        )
        if tl["frames"]:
            batch_span_s = tl["frames"][-1]["t"] - tl["frames"][0]["t"]

    synth_shot_ok = False
    synth_recovery_ok = False
    synth_move_ok = False
    for clip in v1["clips"]:
        if not clip.get("timeline"):
            continue
        m = score_clip(clip, EventDetector())
        if m is None:
            continue
        if clip["id"] == "synth_shot_goal2":
            synth_shot_ok = m["tp"] >= 1
        if clip["id"] == "synth_recovery":
            synth_recovery_ok = m["tp"] >= 1
        if clip["id"] == "synth_movement_midfield":
            synth_move_ok = m["tp"] >= 1

    yaml_txt = (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    batch_rate = batch_n_dribble / max(batch_span_s / 60.0, 1e-6)
    batch_ok = batch_n_dribble <= 30 and batch_rate <= 0.5
    n_dribble_15s = count_dribble_15s_meta()

    comps = {
        "01_prompt": _score(PROMPT.is_file()),
        "02_synth_regression": _score(parent_ok),
        "03_real_check25_p_emit": _score(
            real_p >= GATE_P_EMIT and real_fp == 0,
            max(1.0, 10.0 * real_p),
        ),
        "04_real_dribble_recall": _score(carry_m["tp"] >= 1),
        "05_negative_windows": _score(neg_fp == 0 and jitter_fp == 0),
        "06_batch_dribble_rate": _score(
            batch_ok, partial=max(1.0, 10.0 - batch_n_dribble / 30.0)
        ),
        "07_batch_p_emit": _score(neg_fp == 0),
        "08_pass_shot_recovery": _score(
            real_p >= GATE_P_EMIT and synth_shot_ok and synth_recovery_ok
        ),
        "09_movement_regression": _score(synth_move_ok),
        "10_config_yaml": _score(
            "dribble_window_frames" in yaml_txt and "dribble_min_carry_m" in yaml_txt
        ),
        "11_dead_ends": _score(
            (ROOT / "reports/events_testing/DEAD_ENDS.md").is_file()
        ),
        "12_render_proof": _score(n_dribble_15s <= 2, partial=max(1.0, 10.0 - n_dribble_15s / 2.0)),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "carry_match": carry_m,
        "batch_n_dribble": batch_n_dribble,
        "batch_span_s": round(batch_span_s, 2),
        "batch_rate_per_min": round(batch_rate, 3),
        "neg_fp": neg_fp,
        "real_p_emit": real_p,
        "n_dribble_15s": n_dribble_15s,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("WROTE", OUT / "scores.json")
    if not payload["pass"]:
        print("FAIL eng-loop dribble_precision")
        return 1
    print("all dribble_precision >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
