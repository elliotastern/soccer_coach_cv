#!/usr/bin/env python3
"""V1: stamp human event labels onto phase1_check stills for spot-check."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "reports/eval_match3/improve_eng_loop/phase1_check"
LABELS = (
    ROOT
    / "data/processed/gold_sets/match3_events_v1/clips/check25_human/labels.json"
)
OUT = ROOT / "reports/eval_match3/improve_eng_loop/heuristic_events"


def main() -> int:
    labs = json.loads(LABELS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    # Use mid still; annotate label list
    src = CHECK / "still_mid.jpg"
    if not src.exists():
        src = CHECK / "still_first.jpg"
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"missing still {src}")
    y = 40
    cv2.putText(
        img,
        "V1 human event windows (check25)",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 28
    for ev in labs.get("events") or []:
        line = f"{ev['type']}  t={ev['t_start']:.1f}-{ev['t_end']:.1f}s"
        cv2.putText(
            img,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    out = OUT / "v1_check25_event_labels.jpg"
    cv2.imwrite(str(out), img)
    meta = {
        "still": str(src.relative_to(ROOT)),
        "labels": str(LABELS.relative_to(ROOT)),
        "events": labs.get("events"),
        "note": "Human windows for review; synth gold is eng-loop gate",
    }
    (OUT / "v1_check25_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
