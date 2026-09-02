#!/usr/bin/env python3
"""Eng-loop: fisheye viewer stays reachable (Connection Failed = fail).

≥9/10 when ≥9 of 10 timed health probes return HTTP 200 for index + one preview.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fisheye_viewer_up_stability.json"
BASE = "http://127.0.0.1:8081"
INDEX = f"{BASE}/reports/eval_match3/fisheye_dashboard/index.html"
N = 10
SLEEP = 2.0


def probe() -> dict:
    qs = urllib.parse.urlencode(
        {"cam": "P7", "k1": -0.3, "k2": -0.08, "p1": 0, "p2": 0, "alpha": 0.8}
    )
    preview = f"{BASE}/match3_fisheye_preview?{qs}"
    out = {"index": 0, "preview": 0, "ok": False}
    try:
        with urllib.request.urlopen(INDEX, timeout=5) as r:
            out["index"] = int(r.status)
            _ = r.read(64)
    except Exception as e:
        out["index_err"] = str(e)
    try:
        with urllib.request.urlopen(preview, timeout=15) as r:
            out["preview"] = int(r.status)
            n = len(r.read())
            out["preview_bytes"] = n
            if out["index"] == 200 and out["preview"] == 200 and n >= 8000:
                out["ok"] = True
    except Exception as e:
        out["preview_err"] = str(e)
    return out


def main() -> int:
    rows = []
    for i in range(N):
        row = probe()
        row["i"] = i
        rows.append(row)
        time.sleep(SLEEP)
    n_ok = sum(1 for r in rows if r.get("ok"))
    score = round(10.0 * n_ok / N, 2)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "score": score,
        "pass": score >= 9.0,
        "n_ok": n_ok,
        "n": N,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"score": score, "pass": payload["pass"], "n_ok": n_ok, "wrote": str(OUT)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
