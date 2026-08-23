"""Fuse 15s shot/recovery gold — negatives on goal-band passes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"


def main() -> int:
    CLIP.mkdir(parents=True, exist_ok=True)
    labels_path = CLIP / "labels.json"
    base = {}
    if labels_path.is_file():
        base = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {
        **base,
        "shot_recovery_note": (
            "No true shot/recovery in this 15s window. Goal-band high-speed exits are pass."
        ),
        "shot_negatives": [
            {
                "t_start": 1.5,
                "t_end": 2.0,
                "note": "Goal2 band exit — pass not shot",
            },
            {
                "t_start": 4.8,
                "t_end": 5.5,
                "note": "Fast leave Goal2 — pass not shot",
            },
        ],
        "recovery_negatives": [
            {
                "t_start": 0.0,
                "t_end": 15.0,
                "note": "No labeled recovery in fuse 15s",
            },
        ],
    }
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (CLIP / "shot_recovery_note.txt").write_text(
        "Fuse 15s: score shot/recovery FP only; synth clips gate TP.\n",
        encoding="utf-8",
    )
    print("WROTE", labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
