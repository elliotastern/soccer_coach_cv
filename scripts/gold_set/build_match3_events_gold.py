#!/usr/bin/env python3
"""Build tiny Match 3 / synthetic event gold (pass, shot, recovery)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/gold_sets/match3_events_v1"


def _write(clip_id: str, timeline: dict, labels: dict, note: str):
    d = OUT / "clips" / clip_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    (d / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (d / "note.txt").write_text(note + "\n", encoding="utf-8")
    return {
        "id": clip_id,
        "timeline": str((d / "timeline.json").relative_to(ROOT)),
        "labels": str((d / "labels.json").relative_to(ROOT)),
        "note": note,
    }


def _frames_multi(steps: list, dt: float = 0.33):
    """steps: list of (ball_xy, players_list) per frame."""
    frames = []
    for i, (ball, players) in enumerate(steps):
        frames.append(
            {
                "frame_id": i,
                "t": round(i * dt, 4),
                "ball": list(ball),
                "players": players,
            }
        )
    return {"fps": 1.0 / dt, "dt": dt, "frames": frames}


def _frames_two(ball0, ball1, players0, players1, dt=1.0):
    return {
        "fps": 1.0 / dt,
        "dt": dt,
        "frames": [
            {
                "frame_id": 0,
                "t": 0.0,
                "ball": list(ball0),
                "players": players0,
            },
            {
                "frame_id": 1,
                "t": dt,
                "ball": list(ball1),
                "players": players1,
            },
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []

    # Strong midfield pass (18 m in 1 s → conf 0.9)
    clips.append(
        _write(
            "synth_pass_strong",
            _frames_two(
                (0.0, 0.0),
                (18.0, 0.0),
                [[1, 0.0, 0.0]],
                [[1, 0.0, 0.0]],
                dt=1.0,
            ),
            {
                "events": [
                    {"type": "pass", "t_start": 0.0, "t_end": 1.0, "frame_end": 1}
                ]
            },
            "Synthetic strong pass — expect emit pass",
        )
    )

    # Weak pass should NOT emit (gold has no event)
    clips.append(
        _write(
            "synth_pass_weak_none",
            _frames_two(
                (0.0, 0.0),
                (5.5, 0.0),
                [[1, 0.0, 0.0]],
                [[1, 0.0, 0.0]],
                dt=1.0,
            ),
            {"events": []},
            "Weak pass below emit conf — expect no emit",
        )
    )

    # Shot toward Goal2 (Pitch 1 half 26.95) — realistic ≤40 m/s
    clips.append(
        _write(
            "synth_shot_goal2",
            _frames_two(
                (20.0, 0.0),
                (25.0, 0.0),
                [[1, 20.0, 0.0]],
                [[1, 20.0, 0.0]],
                dt=0.2,
            ),
            {
                "events": [
                    {"type": "shot", "t_start": 0.0, "t_end": 0.2, "frame_end": 1}
                ]
            },
            "Synthetic shot into Goal2 band — expect emit shot",
        )
    )

    # Fast midfield motion is pass-not-shot (outside goal band) — strong vel but midfield
    clips.append(
        _write(
            "synth_midfield_fast_pass",
            _frames_two(
                (0.0, 0.0),
                (18.0, 0.0),
                [[1, 0.0, 0.0]],
                [[1, 0.0, 0.0]],
                dt=1.0,
            ),
            {
                "events": [
                    {"type": "pass", "t_start": 0.0, "t_end": 1.0, "frame_end": 1}
                ]
            },
            "Fast midfield = pass not shot",
        )
    )

    # Recovery: ball arrives at player
    clips.append(
        _write(
            "synth_recovery",
            _frames_two(
                (3.0, 0.0),
                (0.2, 0.0),
                [[1, 0.0, 0.0]],
                [[1, 0.0, 0.0]],
                dt=1.0,
            ),
            {
                "events": [
                    {
                        "type": "recovery",
                        "t_start": 0.0,
                        "t_end": 1.0,
                        "frame_end": 1,
                    }
                ]
            },
            "Ball enters player proximity — expect recovery",
        )
    )

    # Dribble: player and ball co-move over 3 steps (temporal window)
    clips.append(
        _write(
            "synth_dribble_midfield",
            _frames_multi(
                [
                    ((0.0, 0.0), [[1, 0.0, 0.0]]),
                    ((0.25, 0.0), [[1, 0.18, 0.0]]),
                    ((0.55, 0.0), [[1, 0.36, 0.0]]),
                    ((0.95, 0.0), [[1, 0.54, 0.0]]),
                ],
                dt=0.33,
            ),
            {
                "events": [
                    {"type": "dribble", "t_start": 0.0, "t_end": 0.99, "frame_end": 3}
                ]
            },
            "Synthetic midfield dribble — expect emit dribble after window",
        )
    )

    # Movement: slow co-moving carry
    clips.append(
        _write(
            "synth_movement_midfield",
            _frames_two(
                (0.0, 0.0),
                (2.5, 0.0),
                [[1, 0.0, 0.0]],
                [[1, 0.5, 0.0]],
                dt=1.0,
            ),
            {
                "events": [
                    {"type": "movement", "t_start": 0.0, "t_end": 1.0, "frame_end": 1}
                ]
            },
            "Synthetic slow co-move — expect emit movement",
        )
    )

    # Goal-band jitter: static player, ball wiggle — must NOT emit
    clips.append(
        _write(
            "synth_goal_jitter_none",
            _frames_two(
                (25.0, 2.5),
                (25.4, 2.7),
                [[1, 24.0, 2.0]],
                [[1, 24.0, 2.0]],
                dt=0.25,
            ),
            {"events": []},
            "Goal-band fusion jitter — expect no dribble/movement emit",
        )
    )

    # Check-window anchors — product gate (continuous fuse xy; passes only).
    check_labels = {
        "source": "reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4",
        "start_frame": 2390,
        "src_fps": 60.0,
        "timeline": "timeline.json",
        "label_note": "Continuous product-fuse xy only. Goal-band high-speed moves that leave the goal are pass (not shot).",
        "events": [
            {
                "type": "pass",
                "t_start": 2.0,
                "t_end": 2.5,
                "note": "High-speed continuous move leaving Goal2 band",
            },
            {
                "type": "pass",
                "t_start": 5.75,
                "t_end": 6.25,
                "note": "Continuous ~29 m/s leave Goal2 toward midfield",
            },
            {
                "type": "pass",
                "t_start": 15.25,
                "t_end": 15.75,
                "note": "High-speed continuous leave Goal2 band",
            },
        ],
    }
    check_dir = OUT / "clips" / "check25_human"
    check_dir.mkdir(parents=True, exist_ok=True)
    (check_dir / "labels.json").write_text(
        json.dumps(check_labels, indent=2), encoding="utf-8"
    )
    (check_dir / "note.txt").write_text(
        "Human labels on phase1_check 25s — scored when timeline exists; "
        "otherwise review/V1 only.\n",
        encoding="utf-8",
    )
    clips.append(
        {
            "id": "check25_human",
            "labels": str((check_dir / "labels.json").relative_to(ROOT)),
            "timeline": None,
            "note": "Human check-window labels (V1 / review)",
            "score_offline": False,
        }
    )

    manifest = {
        "id": "match3_events_v1",
        "slice": ["pass", "shot", "recovery", "dribble", "movement"],
        "emit_conf": 0.80,
        "pitch": "Pitch 1 (53.90 m)",
        "clips": clips,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE", OUT / "manifest.json", "n_clips=", len(clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
