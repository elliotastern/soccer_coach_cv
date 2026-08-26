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
    span_s = max(ts) - min(ts) if len(ts) > 1 else 0.0
    if span_s == 0.0 and ts:
        span_s = 1.0  # single-timestamp cluster
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
        ROOT / "data/output/match_4_full",
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
        "match4_full_emit": sum(
            r["n_emit_conf"]
            for r in rows
            if "match_4_full" in r["path"] and "-match4" in r["path"]
        ),
        "all_dribble_ok": all(
            r.get("dribble_ok", True)
            for r in rows
            if "match_4" in r.get("path", "")
        ),
        "fuse_clip_scores": fuse_clip_scores(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"BATCH_EVENTS_AUDIT_{datetime.now(timezone.utc):%Y%m%d}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = OUT_DIR / f"BATCH_EVENTS_AUDIT_{datetime.now(timezone.utc):%Y%m%d}.md"
    lines = [
        f"# Batch events audit ({payload['ts'][:10]})",
        "",
        f"Emit gate: **{EMIT_CONF}** · Match4 quad emit conf: **{payload['match4_quad_emit']}** · "
        f"Match4 full emit: **{payload.get('match4_full_emit', 0)}**",
        f"All dribble caps OK: **{payload['all_dribble_ok']}**",
        "",
        "## Per-pack",
        "",
        "| Pack | emits | conf≥0.8 | dribble | pass/min (emit) |",
        "|------|------:|--------:|--------:|----------------:|",
    ]
    for r in rows:
        if r.get("n_total", 0) == 0 and not r.get("path"):
            continue
        pack = r["path"].replace("data/output/", "")
        lines.append(
            f"| {pack} | {r.get('n_total', 0)} | "
            f"{r.get('n_emit_conf', 0)} | {r.get('n_dribble', 0)} | "
            f"{r.get('per_min_emit', 0)} |"
        )
    if payload.get("fuse_clip_scores"):
        lines.extend(["", "## Fuse gold clips", ""])
        for sc in payload["fuse_clip_scores"]:
            lines.append(
                f"- **{sc['clip']}**: P_emit={sc.get('p_emit')} "
                f"tp={sc.get('tp')} fp={sc.get('fp')}"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("WROTE", out)
    print("WROTE", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
