#!/usr/bin/env python3
"""Draft fuse labels.json from a batch events cluster (coach-verify before gold)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--cluster", type=int, default=0)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--match-sec", type=float, default=15.0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def clusters(events: list, gap_s: float = 1.5, emit: float = 0.80) -> list[list]:
    hi = sorted(
        [e for e in events if float(e.get("confidence", 0)) >= emit],
        key=lambda e: float(e.get("timestamp_end", 0)),
    )
    out: list[list] = []
    cur: list = []
    for e in hi:
        if not cur:
            cur = [e]
            continue
        if float(e["timestamp_end"]) - float(cur[-1]["timestamp_end"]) <= gap_s:
            cur.append(e)
        else:
            out.append(cur)
            cur = [e]
    if cur:
        out.append(cur)
    return out


def main() -> int:
    args = parse_args()
    data = json.loads(args.events.read_text(encoding="utf-8"))
    cl = clusters(data.get("events") or [])
    if args.cluster >= len(cl):
        print(f"cluster {args.cluster} missing (have {len(cl)})", file=sys.stderr)
        return 1
    window_t0 = args.start_frame / args.src_fps
    window_t1 = window_t0 + args.match_sec
    grp = cl[args.cluster]
    t0 = float(grp[0]["timestamp_start"])
    t1 = float(grp[-1]["timestamp_end"])
    types = sorted({e.get("type") for e in grp})
    rel0 = max(0.0, round(t0 - window_t0, 2))
    rel1 = min(args.match_sec, round(t1 - window_t0, 2))
    labels = {
        "source": f"Draft from {args.events.name} cluster_{args.cluster}",
        "start_frame": args.start_frame,
        "stride": args.stride,
        "match_sec": args.match_sec,
        "holdout": True,
        "events": [
            {
                "type": types[0] if len(types) == 1 else "pass",
                "t_start": rel0,
                "t_end": rel1,
                "note": f"Single-cam {len(grp)} emits {t0:.1f}-{t1:.1f}s — verify fuse",
            }
        ],
        "negatives": [],
    }
    if rel0 > 0.5:
        labels["negatives"].append(
            {"t_start": 0.0, "t_end": round(rel0 - 0.25, 2), "note": "Before cluster"}
        )
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print("WROTE", out)
    print(json.dumps(labels, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
