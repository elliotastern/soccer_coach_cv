#!/usr/bin/env python3
"""Write a coach rate sheet from batch events.json (Streamlit scrub aid)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMIT = 0.80
OUT = ROOT / "reports/events_testing"


def rate_sheet(events_path: Path) -> dict:
    data = json.loads(events_path.read_text(encoding="utf-8"))
    events = data.get("events") or []
    hi = sorted(
        [e for e in events if float(e.get("confidence", 0)) >= EMIT],
        key=lambda e: float(e.get("timestamp_start", 0)),
    )
    rows = []
    for i, e in enumerate(hi):
        rows.append(
            {
                "i": i + 1,
                "type": e.get("type"),
                "t_start": round(float(e.get("timestamp_start", 0)), 2),
                "t_end": round(float(e.get("timestamp_end", 0)), 2),
                "frame": e.get("start_frame"),
                "conf": round(float(e.get("confidence", 0)), 3),
                "players": e.get("involved_players"),
                "coach_ok": None,
                "coach_note": "",
            }
        )
    return {
        "source": str(events_path.relative_to(ROOT)),
        "match_id": data.get("match_id"),
        "emit_conf": EMIT,
        "n_emit": len(hi),
        "by_type": dict(Counter(e.get("type") for e in hi)),
        "rows": rows,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    paths = [Path(p) for p in (sys.argv[1:] or [
        "data/output/match_4_5min/P10-match4/events.json",
        "data/output/match_4_5min/P8-match4/events.json",
    ])]
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = []
    for rel in paths:
        p = rel if rel.is_absolute() else ROOT / rel
        if not p.is_file():
            print("SKIP", rel)
            continue
        sheet = rate_sheet(p)
        sheets.append(sheet)
        # e.g. match_4_5min__P10-match4
        stem = f"{p.parent.parent.name}__{p.parent.name}"
        out_json = OUT / f"COACH_RATE_SHEET_{stem}.json"
        out_md = OUT / f"COACH_RATE_SHEET_{stem}.md"
        out_json.write_text(json.dumps(sheet, indent=2), encoding="utf-8")
        lines = [
            f"# Coach rate sheet — `{stem}`",
            "",
            f"Source: `{sheet['source']}` · emits conf≥{EMIT}: **{sheet['n_emit']}** · {sheet['by_type']}",
            "",
            "Fill `coach_ok` (true/false) while scrubbing Streamlit / mosaic.",
            "",
            "| # | type | t_start | t_end | frame | conf | ok? | note |",
            "|---|------|--------:|------:|------:|-----:|-----|------|",
        ]
        for r in sheet["rows"]:
            lines.append(
                f"| {r['i']} | {r['type']} | {r['t_start']} | {r['t_end']} | "
                f"{r['frame']} | {r['conf']} |  |  |"
            )
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("WROTE", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
