#!/usr/bin/env python3
"""Eng-loop for Phase 1 heuristic events — P_emit ≥ 0.80 on gold slice."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events.events import EMIT_CONF, PITCH1_HALF_LENGTH_M, EventDetector  # noqa: E402
from src.state.types import Ball, FrameData, Player  # noqa: E402


def _frame_from_row(row: dict) -> FrameData:
    players = []
    for p in row.get("players") or []:
        pid, x, y = int(p[0]), float(p[1]), float(p[2])
        players.append(
            Player(pid, 0, x, y, (0, 0, 10, 10), int(row["frame_id"]), float(row["t"]))
        )
    ball = None
    if row.get("ball") is not None:
        bx, by = row["ball"]
        ball = Ball(
            float(bx), float(by), (0, 0, 4, 4), int(row["frame_id"]), float(row["t"])
        )
    return FrameData(int(row["frame_id"]), float(row["t"]), players, ball)


def run_timeline(timeline: dict, det: EventDetector | None = None) -> list[dict]:
    det = det or EventDetector()
    frames = [_frame_from_row(r) for r in timeline["frames"]]
    emits = []
    prev = None
    for fr in frames:
        for ev in det.detect_events(fr, prev):
            emits.append(
                {
                    "type": ev.type.value,
                    "frame_end": ev.end_frame,
                    "t_start": ev.timestamp_start,
                    "t_end": ev.timestamp_end,
                    "confidence": ev.confidence,
                    "players": list(ev.involved_players),
                }
            )
        prev = fr
    return emits


PASS = 9.0
GATE_P_EMIT = 0.80
OUT = ROOT / "reports/eval_match3/improve_eng_loop/heuristic_events"
GOLD = ROOT / "data/processed/gold_sets/match3_events_v1/manifest.json"
MATCH_TOL_S = 0.55


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def match_emits(gold_events: list, emits: list) -> dict:
    """Precision-first: TP = emit matched to gold same type within tol."""
    used = set()
    tp = 0
    for em in emits:
        matched = False
        for i, g in enumerate(gold_events):
            if i in used:
                continue
            if g["type"] != em["type"]:
                continue
            # Match on end time
            gt = float(g.get("t_end", g.get("t_start", 0.0)))
            et = float(em.get("t_end", em.get("t_start", 0.0)))
            if abs(gt - et) <= MATCH_TOL_S:
                used.add(i)
                matched = True
                break
        if matched:
            tp += 1
    fp = len(emits) - tp
    fn = len(gold_events) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1) if gold_events else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "p_emit": p, "recall": r, "n_emit": len(emits)}


def score_clip(clip: dict, det: EventDetector) -> dict | None:
    if clip.get("score_offline") is False or not clip.get("timeline"):
        return None
    # Fresh detector per clip — cooldown/teleport state must not leak.
    det = EventDetector(
        pass_velocity_threshold=det.pass_velocity_threshold,
        dribble_distance_threshold=det.dribble_distance_threshold,
        shot_velocity_threshold=det.shot_velocity_threshold,
        recovery_proximity=det.recovery_proximity,
        emit_conf=det.emit_conf,
        half_length_m=det.half_length_m,
        shot_goal_band_m=det.shot_goal_band_m,
        max_ball_speed_m_s=det.max_ball_speed_m_s,
        min_emit_gap_s=det.min_emit_gap_s,
        enable_dribble=det.enable_dribble,
        enable_movement=det.enable_movement,
        movement_velocity_min=det.movement_velocity_min,
        movement_proximity=det.movement_proximity,
        co_move_min_player_m=det.co_move_min_player_m,
        co_move_min_cos=det.co_move_min_cos,
    )
    tl_path = ROOT / clip["timeline"]
    lab_path = ROOT / clip["labels"]
    timeline = json.loads(tl_path.read_text(encoding="utf-8"))
    labels = json.loads(lab_path.read_text(encoding="utf-8"))
    emits = run_timeline(timeline, det)
    (tl_path.parent / "emits.json").write_text(
        json.dumps({"emits": emits}, indent=2), encoding="utf-8"
    )
    m = match_emits(labels.get("events") or [], emits)
    m["id"] = clip["id"]
    m["emits"] = emits
    return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not GOLD.exists():
        import subprocess

        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts/gold_set/build_match3_events_gold.py")],
            cwd=str(ROOT),
        )
    manifest = json.loads(GOLD.read_text(encoding="utf-8"))
    det = EventDetector()
    per = []
    for clip in manifest["clips"]:
        m = score_clip(clip, det)
        if m is not None:
            per.append(m)

    tot_tp = sum(x["tp"] for x in per)
    tot_fp = sum(x["fp"] for x in per)
    tot_fn = sum(x["fn"] for x in per)
    p_emit = tot_tp / max(tot_tp + tot_fp, 1)
    recall = tot_tp / max(tot_tp + tot_fn, 1)

    # Real Match 3 product-fuse pack (check25) — primary product gate.
    real_dir = ROOT / "data/processed/gold_sets/match3_events_v1/clips/check25_human"
    real_score = None
    if (real_dir / "timeline.json").exists() and (real_dir / "labels.json").exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_check25",
            ROOT / "scripts/gold_set/build_check25_event_timeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        tl = json.loads((real_dir / "timeline.json").read_text(encoding="utf-8"))
        labs = json.loads((real_dir / "labels.json").read_text(encoding="utf-8"))
        real_emits = mod.run_events_on_timeline(tl)
        (real_dir / "emits.json").write_text(
            json.dumps({"emits": real_emits}, indent=2), encoding="utf-8"
        )
        real_score = mod.score_windows(labs.get("events") or [], real_emits)
        (real_dir / "score_real.json").write_text(
            json.dumps(real_score, indent=2), encoding="utf-8"
        )

    real_p = float(real_score["p_emit"]) if real_score else 0.0
    real_r = float(real_score["recall"]) if real_score else 0.0

    # Component scores (≥9 each for PASS)
    comps = {
        "01_prompt_wire": _score((OUT / "PROMPT.md").exists()),
        "02_pitch1_half": _score(abs(PITCH1_HALF_LENGTH_M - 26.95) < 1e-6),
        "03_no_fifa_52_5": _score("52.5" not in (ROOT / "src/events/events.py").read_text()),
        "04_emit_conf_const": _score(EMIT_CONF >= 0.80 and det.emit_conf >= 0.80),
        "05_priority_exclusive": _score(
            next((x for x in per if x["id"] == "synth_shot_goal2"), {"tp": 0})["tp"]
            >= 1
        ),
        "06_gold_manifest": _score(GOLD.exists() and len(per) >= 4),
        "07_offline_runner": _score(
            all(
                (ROOT / c["timeline"]).parent.joinpath("emits.json").exists()
                for c in manifest["clips"]
                if c.get("timeline")
            )
        ),
        "08_p_emit_gate": _score(p_emit >= GATE_P_EMIT, partial=max(1.0, 10.0 * p_emit)),
        "08b_real_p_emit": _score(real_p >= GATE_P_EMIT, partial=max(1.0, 10.0 * real_p)),
        "09_weak_none_clean": _score(
            next((x for x in per if x["id"] == "synth_pass_weak_none"), {"fp": 1})["fp"]
            == 0
        ),
        "10_shot_goal2": _score(
            next((x for x in per if x["id"] == "synth_shot_goal2"), {"tp": 0})["tp"]
            >= 1
        ),
        "11_pass_strong": _score(
            next((x for x in per if x["id"] == "synth_pass_strong"), {"tp": 0})["tp"]
            >= 1
        ),
        "12_recovery": _score(
            next((x for x in per if x["id"] == "synth_recovery"), {"tp": 0})["tp"] >= 1
        ),
        "13_midfield_not_shot": _score(
            all(
                e["type"] != "shot"
                for x in per
                if x["id"] == "synth_midfield_fast_pass"
                for e in x.get("emits") or []
            )
        ),
        "14_recall_secondary": _score(
            real_r >= 0.5 if real_score else recall >= 0.5,
            partial=max(1.0, 10.0 * (real_r if real_score else recall)),
        ),
        "15_dead_ends_doc": _score(
            (ROOT / "reports/events_testing/DEAD_ENDS.md").exists()
        ),
        "16_front_doc": _score(
            (ROOT / "reports/events_testing/EVENTS_FRONT.md").exists()
        ),
        "17_improve_plan": _score(
            (ROOT / "docs/product/HEURISTIC_EVENTS_IMPROVE_PLAN.md").exists()
        ),
        "18_unit_e0": _score(
            (ROOT / "scripts/test_heuristic_events_e0.py").exists()
        ),
        "19_config_emit": _score(
            "emit_conf" in (ROOT / "configs/default.yaml").read_text()
        ),
        "20_product_ready": 0.0,
    }
    core = [
        comps["08b_real_p_emit"],
        comps["08_p_emit_gate"],
        comps["02_pitch1_half"],
        comps["04_emit_conf_const"],
        comps["10_shot_goal2"],
        comps["11_pass_strong"],
    ]
    comps["20_product_ready"] = sum(core) / len(core)

    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "p_emit_synth": round(p_emit, 4),
        "recall_synth": round(recall, 4),
        "p_emit_real": round(real_p, 4),
        "recall_real": round(real_r, 4),
        "real_score": real_score,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "gate_p_emit": GATE_P_EMIT,
        "pass_bar": PASS,
        "per_clip": [{k: v for k, v in x.items() if k != "emits"} for x in per],
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0 and real_p >= GATE_P_EMIT and p_emit >= GATE_P_EMIT,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("p_emit_real", "recall_real", "p_emit_synth", "pass", "failed")
            },
            indent=2,
        )
    )
    print("WROTE", OUT / "scores.json")
    if not payload["pass"]:
        print("FAIL eng-loop heuristic events")
        return 1
    print("all heuristic_events >= 9/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
