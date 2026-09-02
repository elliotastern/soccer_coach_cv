#!/usr/bin/env python3
"""Eng-loop: fisheye dashboard still + preview stay non-black across cam tabs.

Score ≥9/10 when every cam returns orig still and undistort preview JPEG
with HTTP 200 and size ≥ 8 KiB (black/broken loads are tiny or fail).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/fisheye_dashboard_tab_stability.json"
BASE = "http://127.0.0.1:8081"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
MIN_BYTES = 8 * 1024


def fetch(url: str) -> tuple[int, int]:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
            return int(r.status), len(data)
    except urllib.error.HTTPError as e:
        return int(e.code), 0
    except Exception:
        return 0, 0


def score_cam(cam: str) -> dict:
    still = f"{BASE}/reports/eval_match3/fisheye_dashboard/stills/{cam}.jpg"
    qs = urllib.parse.urlencode(
        {"cam": cam, "k1": -0.3, "k2": 0, "p1": 0, "p2": 0, "alpha": 0.8}
    )
    prev = f"{BASE}/match3_fisheye_preview?{qs}"
    s_code, s_n = fetch(still)
    p_code, p_n = fetch(prev)
    ok = (
        s_code == 200
        and p_code == 200
        and s_n >= MIN_BYTES
        and p_n >= MIN_BYTES
    )
    return {
        "cam": cam,
        "still_http": s_code,
        "still_bytes": s_n,
        "preview_http": p_code,
        "preview_bytes": p_n,
        "ok": ok,
    }


def main() -> int:
    # Burst: hit each cam twice to catch reload races on tab switch
    rows = []
    for cam in CAMS:
        a = score_cam(cam)
        b = score_cam(cam)
        rows.append({"first": a, "second": b, "ok": a["ok"] and b["ok"]})
    n_ok = sum(1 for r in rows if r["ok"])
    score = round(10.0 * n_ok / max(len(CAMS), 1), 2)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "n_ok": n_ok,
        "n_cams": len(CAMS),
        "score": score,
        "pass": score >= 9.0,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"score": score, "pass": payload["pass"], "n_ok": n_ok, "wrote": str(OUT)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
