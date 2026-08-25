#!/usr/bin/env python3
"""Score mosaic emits_render against provisional holdout labels (report-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gold_set"))
from build_check25_event_timeline import score_windows  # noqa: E402

MOSAIC_EMITS = (
    ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json"
)
CLIPS = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips"
# match_sec windows keyed by absolute mosaic start (frame/60 for start_frame)
HOLDOUTS = {
    "real_fuse_eval_20s": 1200 / 60.0,  # 20s
    "real_fuse_eval_49s": 2940 / 60.0,  # 49s
    "real_fuse_eval_69s": 4226 / 60.0,  # ~70.4s
}


def mosaic_emits_in_window(emits: list, abs_start: float, match_sec: float) -> list:
    out = []
    for e in emits:
        te = float(e.get("t_end", 0))
        if abs_start <= te <= abs_start + match_sec:
            row = dict(e)
            row["t_end"] = te - abs_start
            row["t_start"] = float(e.get("t_start", te)) - abs_start
            out.append(row)
    return out


def main() -> int:
    emits = json.loads(MOSAIC_EMITS.read_text(encoding="utf-8")).get("emits") or []
    rows = []
    for clip_id, abs_start in HOLDOUTS.items():
        labels_path = CLIPS / clip_id / "labels.json"
        if not labels_path.is_file():
            continue
        labs = json.loads(labels_path.read_text(encoding="utf-8"))
        match_sec = float(labs.get("match_sec", 15.0))
        window_emits = mosaic_emits_in_window(emits, abs_start, match_sec)
        sc = score_windows(labs.get("events") or [], window_emits)
        rows.append(
            {
                "clip": clip_id,
                "abs_start_s": abs_start,
                "n_mosaic_emits": len(window_emits),
                "mosaic_types": [e["type"] for e in window_emits],
                **sc,
            }
        )
        print(
            f"{clip_id}: p_emit={sc['p_emit']} tp={sc['tp']} fp={sc['fp']} "
            f"fn={sc['fn']} mosaic_n={len(window_emits)} { [e['type'] for e in window_emits] }"
        )
    out = ROOT / "reports/events_testing/MOSAIC_HOLDOUT_SCORE.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
