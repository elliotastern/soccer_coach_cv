#!/usr/bin/env python3
"""Summarize mosaic emits_render.json vs batch events (coach / eng-loop aid)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/events_testing"


def load_emits(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "emits" in data:
        return data["emits"]
    if isinstance(data, dict) and "events" in data:
        return [
            {
                "type": e.get("type"),
                "t_end": float(e.get("timestamp_end", e.get("t_end", 0))),
                "t_start": float(e.get("timestamp_start", e.get("t_start", 0))),
                "confidence": float(e.get("confidence", 0)),
            }
            for e in data["events"]
            if float(e.get("confidence", 0)) >= 0.8
        ]
    return []


def windows(emits: list, spans: list[tuple[str, float, float]]) -> dict:
    out = {}
    for name, t0, t1 in spans:
        w = [e for e in emits if t0 <= float(e.get("t_end", 0)) <= t1]
        out[name] = {
            "t0": t0,
            "t1": t1,
            "n": len(w),
            "by_type": dict(Counter(e["type"] for e in w)),
            "emits": w,
        }
    return out


def main() -> int:
    mosaic = ROOT / (
        sys.argv[1]
        if len(sys.argv) > 1
        else "reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json"
    )
    batch = ROOT / (
        sys.argv[2]
        if len(sys.argv) > 2
        else "data/output/match_4_5min/P10-match4/events.json"
    )
    if not mosaic.is_file():
        print("MISSING", mosaic)
        return 1
    m_em = load_emits(mosaic)
    b_em = load_emits(batch) if batch.is_file() else []
    spans = [
        ("0_15s", 0.0, 15.0),
        ("20_35s", 20.0, 35.0),
        ("44_64s", 44.0, 64.0),
        ("69_84s", 69.0, 84.0),
        ("full", 0.0, 9999.0),
    ]
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mosaic": str(mosaic.relative_to(ROOT)),
        "batch": str(batch.relative_to(ROOT)) if batch.is_file() else None,
        "mosaic_n": len(m_em),
        "mosaic_by_type": dict(Counter(e["type"] for e in m_em)),
        "batch_n": len(b_em),
        "batch_by_type": dict(Counter(e["type"] for e in b_em)),
        "windows": windows(m_em, spans),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "MOSAIC_EMITS_ANALYSIS.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = OUT / "MOSAIC_EMITS_ANALYSIS.md"
    lines = [
        f"# Mosaic emits analysis ({payload['ts'][:10]})",
        "",
        f"Mosaic: `{payload['mosaic']}` · **{payload['mosaic_n']}** · {payload['mosaic_by_type']}",
        f"Batch: `{payload['batch']}` · **{payload['batch_n']}** · {payload['batch_by_type']}",
        "",
        "| Window | n | types |",
        "|--------|--:|-------|",
    ]
    for name, w in payload["windows"].items():
        if name == "full":
            continue
        lines.append(f"| {name} | {w['n']} | {w['by_type']} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("mosaic_n", "mosaic_by_type", "batch_n", "batch_by_type")}, indent=2))
    print("WROTE", out)
    print("WROTE", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
