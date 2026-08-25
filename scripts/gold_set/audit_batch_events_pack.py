#!/usr/bin/env python3
"""Audit batch events.json packs — rates, types, emit-conf gate."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMIT_CONF = 0.80
OUT_DIR = ROOT / "reports/events_testing"


def audit_pack(path: Path) -> dict | None:
    ev_path = path / "events.json"
    if not ev_path.is_file():
        return None
    data = json.loads(ev_path.read_text(encoding="utf-8"))
    events = data.get("events") or []
    if not events:
        return {
            "path": str(path.relative_to(ROOT)),
            "n_total": 0,
            "n_emit_conf": 0,
            "span_s": 0.0,
            "dribble_ok": True,
        }
    ts = [float(e.get("timestamp_end", e.get("timestamp_start", 0))) for e in events]
    span_s = max(ts) - min(ts) if ts else 0.0
    by_type = Counter(e.get("type", "?") for e in events)
    hi = [e for e in events if float(e.get("confidence", 0)) >= EMIT_CONF]
    hi_type = Counter(e.get("type", "?") for e in hi)
    drib = sum(1 for e in events if e.get("type") == "dribble")
    return {
        "path": str(path.relative_to(ROOT)),
        "match_id": data.get("match_id"),
        "n_total": len(events),
        "n_emit_conf": len(hi),
        "span_s": round(span_s, 2),
        "per_min_total": round(len(events) / max(span_s / 60.0, 1e-6), 2),
        "per_min_emit": round(len(hi) / max(span_s / 60.0, 1e-6), 2),
        "by_type": dict(by_type),
        "emit_by_type": dict(hi_type),
        "n_dribble": drib,
        "dribble_ok": drib <= 30,
    }


def scan_roots() -> list[Path]:
    roots = [
        ROOT / "data/output/match_4_5min",
        ROOT / "data/output/full_match_2min",
    ]
    packs = []
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "events.json").is_file():
                packs.append(d)
    return packs


def fuse_clip_scores() -> list[dict]:
    clips = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips"
    rows = []
    if not clips.is_dir():
        return rows
    for d in sorted(clips.iterdir()):
        sc_path = d / "score_real.json"
        if not sc_path.is_file():
            continue
        sc = json.loads(sc_path.read_text(encoding="utf-8"))
        rows.append({"clip": d.name, **sc})
    return rows


def main() -> int:
    packs = scan_roots()
    rows = [audit_pack(p) for p in packs]
    rows = [r for r in rows if r is not None]
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "emit_conf_gate": EMIT_CONF,
        "packs": rows,
        "match4_quad_emit": sum(
            r["n_emit_conf"]
            for r in rows
            if "match_4_5min" in r["path"] and "-match4" in r["path"]
        ),
        "all_dribble_ok": all(r.get("dribble_ok", True) for r in rows),
        "fuse_clip_scores": fuse_clip_scores(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"BATCH_EVENTS_AUDIT_{datetime.now(timezone.utc):%Y%m%d}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
