#!/usr/bin/env python3
"""Build projection3d block for all Match 3 calibs; write calib3d report."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.camera_projection import enrich_all_calibs  # noqa: E402
from src.mapping.match3_xy import MATCH3_CAMS  # noqa: E402

OUT = ROOT / "reports/eval_match3/calib3d/calib3d_report.json"


def main() -> int:
    rows = enrich_all_calibs(list(MATCH3_CAMS))
    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "cameras": rows,
        "pass": len(fail) == 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(r)
    print(f"wrote {OUT} pass={report['pass']}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
