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

    # Check-window anchors (human labels from phase1_check 25s @ fr 2390)
    # Times are match-relative seconds from clip start (t+ in mosaic footer).
    check_labels = {
        "source": "reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4",
        "start_frame": 2390,
        "src_fps": 60.0,
        "events": [
            {
                "type": "shot",
                "t_start": 4.0,
                "t_end": 5.5,
                "note": "Ball on Goal2 line / P8 — shot-like",
            },
            {
                "type": "pass",
                "t_start": 10.5,
                "t_end": 12.0,
                "note": "Ball relocates midfield→north with pace",
            },
            {
                "type": "recovery",
                "t_start": 22.0,
                "t_end": 23.5,
                "note": "Clear ball then player contact in P7",
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
        "slice": ["pass", "shot", "recovery"],
        "emit_conf": 0.80,
        "pitch": "Pitch 1 (53.90 m)",
        "clips": clips,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE", OUT / "manifest.json", "n_clips=", len(clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
