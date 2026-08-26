#!/usr/bin/env python3
"""Eng-loop: Match 4 team balance gates from mosaic meta.json (PROMPT team_match4_balance)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/eval_match3/improve_eng_loop/team_match4_balance"
BASELINE = OUT / "team_match4_baseline.json"
DEFAULT_META = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/meta.json"
GATE = 9.0

# Product gates (Match 4 white-kit collapse fix)
MEAN_BLUE_LO, MEAN_BLUE_HI = 0.35, 0.65
MAX_COLLAPSE_FRAC = 0.15
MIN_BOTH3_FRAC = 0.50
MAX_MEAN_PLAYERS = 14.5


def _score_bool(ok: bool, partial: float = 3.0) -> float:
    return 10.0 if ok else partial


def _score_range(val: float, lo: float, hi: float) -> float:
    if lo <= val <= hi:
        return 10.0
    if val < lo:
        return max(0.0, 10.0 * val / lo) if lo > 0 else 0.0
    return max(0.0, 10.0 * hi / val) if val > 0 else 0.0


def analyze_stats(stats: list[dict]) -> dict:
    if not stats:
        return {}
    blue_shares = []
    collapse = both3 = balanced = 0
    players_sum = 0
    for s in stats:
        n0, n1 = int(s.get("n0", 0)), int(s.get("n1", 0))
        n = n0 + n1
        players_sum += int(s.get("n", n0 + n1 + int(s.get("gray", 0))))
        if n > 0:
            blue_shares.append(n0 / n)
            if 0.25 <= n0 / n <= 0.75:
                balanced += 1
        if n1 <= 1 and n0 >= 5:
            collapse += 1
        if n0 >= 3 and n1 >= 3:
            both3 += 1
    n_frames = len(stats)
    return {
        "n_frames": n_frames,
        "mean_blue_share": float(np.mean(blue_shares)) if blue_shares else 0.0,
        "collapse_frac": collapse / n_frames,
        "both3_frac": both3 / n_frames,
        "balanced_frac": balanced / n_frames,
        "mean_players": players_sum / n_frames,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    meta_path = Path(args.meta)
    if not meta_path.is_file():
        scores["01_meta_exists"] = 0.0
        notes["01_meta_exists"] = f"missing {meta_path}"
        (OUT / "scores.json").write_text(json.dumps({"pass": False, "scores": scores}, indent=2))
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats") or []
    m = analyze_stats(stats)

    scores["01_meta_exists"] = 10.0
    notes["01_meta_exists"] = str(meta_path)

    scores["02_n_frames"] = _score_bool(m.get("n_frames", 0) >= 100)
    notes["02_n_frames"] = f"n={m.get('n_frames', 0)}"

    mean_blue = m.get("mean_blue_share", 0.0)
    scores["03_mean_blue_share"] = _score_range(mean_blue, MEAN_BLUE_LO, MEAN_BLUE_HI)
    notes["03_mean_blue_share"] = f"mean={mean_blue:.3f} gate=[{MEAN_BLUE_LO},{MEAN_BLUE_HI}]"

    collapse = m.get("collapse_frac", 1.0)
    scores["04_collapse_frac"] = 10.0 if collapse <= MAX_COLLAPSE_FRAC else max(
        0.0, 10.0 * (1.0 - collapse) / (1.0 - MAX_COLLAPSE_FRAC)
    )
    notes["04_collapse_frac"] = f"collapse={collapse:.3f} max={MAX_COLLAPSE_FRAC}"

    both3 = m.get("both3_frac", 0.0)
    scores["05_both3_frac"] = 10.0 if both3 >= MIN_BOTH3_FRAC else max(
        0.0, 10.0 * both3 / MIN_BOTH3_FRAC
    )
    notes["05_both3_frac"] = f"both3={both3:.3f} min={MIN_BOTH3_FRAC}"

    mean_n = m.get("mean_players", 99.0)
    scores["06_mean_players"] = 10.0 if mean_n <= MAX_MEAN_PLAYERS else max(
        0.0, 10.0 * MAX_MEAN_PLAYERS / mean_n
    )
    notes["06_mean_players"] = f"mean_n={mean_n:.2f} max={MAX_MEAN_PLAYERS}"

    scores["07_balanced_frac"] = _score_bool(m.get("balanced_frac", 0.0) >= 0.25)
    notes["07_balanced_frac"] = f"balanced={m.get('balanced_frac', 0):.3f}"

    from src.perception.team_core import KIT_MODE_AUTO, KIT_MODE_MATCH3  # noqa: E402

    scores["08_kit_mode_auto"] = _score_bool(KIT_MODE_AUTO == "auto")
    notes["08_kit_mode_auto"] = f"modes auto={KIT_MODE_AUTO} match3={KIT_MODE_MATCH3}"

    from src.review.multicam_fuse import FUSE_MAX_PLAYERS, PLAYER_MERGE_M_LIVE  # noqa: E402

    scores["09_fuse_cap"] = _score_bool(FUSE_MAX_PLAYERS == 14)
    notes["09_fuse_cap"] = f"max={FUSE_MAX_PLAYERS} merge_live={PLAYER_MERGE_M_LIVE}"

    src = (ROOT / "src/perception/team_core.py").read_text(encoding="utf-8")
    scores["10_no_match3_hard_in_auto"] = _score_bool(
        "if kit_mode == KIT_MODE_MATCH3:" in src and "_lock_labels_auto" in src
    )
    notes["10_no_match3_hard_in_auto"] = "auto skips hue hard rules"

    hard = [
        scores["03_mean_blue_share"],
        scores["04_collapse_frac"],
        scores["05_both3_frac"],
        scores["06_mean_players"],
    ]
    scores["20_product_ready"] = float(np.mean(hard))
    notes["20_product_ready"] = f"mean_hard={scores['20_product_ready']:.1f}"

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "meta": str(meta_path),
        "metrics": m,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not BASELINE.is_file():
        BASELINE.write_text(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "note": "Pre kit_mode=auto baseline (Match 4 collapse)",
                    "metrics": m,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
