#!/usr/bin/env python3
"""Scan batch output folders and write Phase 1 delivery manifest for sign-off."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    ROOT / "data/output/full_match",
    ROOT / "data/output/full_match_2min",
    ROOT / "data/processed/full_match_2min",
]
CSV_COLS = [
    "Timestamp",
    "Team_ID",
    "Player_ID",
    "Event",
    "Location_X",
    "Location_Y",
    "frame_id",
    "confidence",
]
EVENT_TYPES = ("pass", "dribble", "movement", "recovery", "shot")
ACCEPT_MATCHES = ("P10-002", "P1-006")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/eval_match3/improve_eng_loop/delivery_manifest.json",
    )
    p.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Output root to scan (repeatable). Default: full_match + 2min samples.",
    )
    return p.parse_args()


def scan_run(run_dir: Path) -> dict:
    issues = []
    frame_csv = run_dir / "frame_data.csv"
    events_json = run_dir / "events.json"
    events_csv = run_dir / "events.csv"
    ckpt_dir = run_dir / "checkpoints"

    row = {
        "run": run_dir.name,
        "path": str(run_dir.relative_to(ROOT)),
        "has_frame_data": frame_csv.is_file(),
        "has_events_json": events_json.is_file(),
        "has_events_csv": events_csv.is_file(),
        "has_checkpoints": ckpt_dir.is_dir(),
        "n_checkpoints": 0,
        "n_frame_rows": 0,
        "n_events": 0,
        "event_counts": {t: 0 for t in EVENT_TYPES},
        "csv_columns_ok": False,
        "pass": False,
        "issues": issues,
    }

    if not frame_csv.is_file():
        issues.append("missing frame_data.csv")
    if not events_json.is_file():
        issues.append("missing events.json")
    if ckpt_dir.is_dir():
        row["n_checkpoints"] = len(list(ckpt_dir.glob("checkpoint_*.json")))

    if frame_csv.is_file():
        with frame_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            row["csv_columns_ok"] = all(c in cols for c in CSV_COLS)
            if not row["csv_columns_ok"]:
                issues.append(f"frame_data.csv missing columns (have {cols})")
            row["n_frame_rows"] = sum(1 for _ in reader)

    if events_json.is_file():
        data = json.loads(events_json.read_text(encoding="utf-8"))
        events = data.get("events") or []
        row["n_events"] = len(events)
        for ev in events:
            t = str(ev.get("type", "unknown"))
            row["event_counts"][t] = row["event_counts"].get(t, 0) + 1

    row["pass"] = (
        row["has_frame_data"]
        and row["has_events_json"]
        and row["csv_columns_ok"]
        and row["n_frame_rows"] > 0
        and row["has_checkpoints"]
    )
    if row["n_checkpoints"] == 0:
        issues.append("no checkpoint_*.json files")
    return row


def scan_roots(roots: list[Path]) -> list[dict]:
    runs = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or p.name in seen:
                continue
            if not (p / "events.json").is_file():
                continue
            seen.add(p.name)
            runs.append(scan_run(p))
    return runs


def acceptance_status(runs: list[dict]) -> dict:
    by_name = {r["run"]: r for r in runs}
    found = {name: by_name.get(name) for name in ACCEPT_MATCHES}
    full_pass = [
        name
        for name in ACCEPT_MATCHES
        if found.get(name) and found[name]["pass"]
    ]
    return {
        "required_runs": list(ACCEPT_MATCHES),
        "found": {k: bool(v) for k, v in found.items()},
        "pass": {k: bool(v and v["pass"]) for k, v in found.items()},
        "acceptance_met": len(full_pass) >= 2,
        "note": "Acceptance §5.1 needs 2 full-match batch runs (P10-002 + P1-006).",
    }


def main() -> int:
    args = parse_args()
    roots = args.root or DEFAULT_ROOTS
    runs = scan_roots(roots)
    manifest = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "roots_scanned": [str(r.relative_to(ROOT)) for r in roots if r.is_dir()],
        "csv_schema": CSV_COLS,
        "runs": runs,
        "acceptance": acceptance_status(runs),
        "gaps": [
            "Dribble + Movement heuristic emits (E2) not required for manifest pass",
            "3rd-match handover session is manual — not checked by this script",
        ],
    }
    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE", out)
    acc = manifest["acceptance"]
    print(
        f"runs={len(runs)} acceptance_met={acc['acceptance_met']} "
        f"pass={acc['pass']}"
    )
    for r in runs:
        print(
            f"  {r['run']}: pass={r['pass']} rows={r['n_frame_rows']} "
            f"events={r['n_events']} ckpt={r['n_checkpoints']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
