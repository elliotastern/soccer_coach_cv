#!/usr/bin/env python3
"""Apply team ID grid winner to configs/default.yaml."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_strategy import STRATEGIES  # noqa: E402

RANKING = ROOT / "reports/eval_match3/team_id_strategy_grid/ranking.json"
CFG = ROOT / "configs/default.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking", type=Path, default=RANKING)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.ranking.is_file():
        print(f"missing {args.ranking}", file=sys.stderr)
        return 1
    payload = json.loads(args.ranking.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    if not results:
        print("empty ranking", file=sys.stderr)
        return 1
    winner = max(results, key=lambda r: r["scores"]["composite"])
    strat_name = winner["strategy_name"]
    if strat_name not in STRATEGIES:
        print(f"unknown strategy {strat_name}", file=sys.stderr)
        return 1
    strat = STRATEGIES[strat_name]
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}
    ta = cfg.setdefault("team_assignment", {})
    ta["kit_mode"] = strat.kit_mode
    ta["strategy"] = strat_name
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    print(f"winner: {winner['strategy_id']} {strat_name} sahi={winner['sahi']} composite={winner['scores']['composite']}")
    if args.dry_run:
        print(text)
        return 0
    CFG.write_text(text, encoding="utf-8")
    print(f"updated {CFG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
