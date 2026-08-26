#!/usr/bin/env python3
"""Eng-loop: coach emit label dashboard is reachable via Playwright (≥9/10)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/events_testing/coach_dashboard_up_ux"
URL = "http://127.0.0.1:8502/"
HEALTH = "http://127.0.0.1:8502/_stcore/health"
PASS = 9.0
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LAUNCH = ROOT / "scripts/run_coach_emit_label_dashboard.sh"


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= int(r.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _launch(p):
    if Path(CHROME).is_file():
        return p.chromium.launch(
            headless=True, executable_path=CHROME, args=["--no-sandbox"]
        )
    return p.chromium.launch(headless=True)


def ensure_dashboard(timeout_s: float = 45.0) -> dict:
    if _http_ok(HEALTH):
        return {"ok": True, "started": False}
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "streamlit_launch.log"
    proc = subprocess.Popen(
        ["bash", str(LAUNCH)],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PORT": "8502", "PYTHONPATH": "."},
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _http_ok(HEALTH):
            return {"ok": True, "started": True, "pid": proc.pid}
        if proc.poll() is not None:
            return {
                "ok": False,
                "started": True,
                "pid": proc.pid,
                "exit": proc.returncode,
                "log_tail": log.read_text(encoding="utf-8")[-800:] if log.is_file() else "",
            }
        time.sleep(1.0)
    return {"ok": False, "started": True, "pid": proc.pid, "error": "timeout waiting health"}


def run_playwright() -> dict:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    shot = OUT / "pw_dashboard_up.png"
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(8_000)
            body = page.inner_text("body")
            title = page.title()
            page.screenshot(path=str(shot), full_page=False)
        except Exception as exc:
            browser.close()
            return {
                "name": "pw_dashboard_up",
                "pass": False,
                "error": str(exc),
                "health": _http_ok(HEALTH),
            }
        browser.close()

    low = body.lower()
    fail_ui = any(
        s in low
        for s in (
            "connection failed",
            "connection error",
            "please wait",
            "please check your connection",
        )
    ) and "emit" not in low
    has_app = (
        "fuse emit" in title.lower()
        or "emit" in low
        or "view layout" in low
        or "best ball" in low
    )
    ok = has_app and not fail_ui and _http_ok(HEALTH)
    return {
        "name": "pw_dashboard_up",
        "pass": ok,
        "title": title,
        "has_app": has_app,
        "fail_ui": fail_ui,
        "health": _http_ok(HEALTH),
        "body_snip": " ".join(body.split())[:240],
        "shot": str(shot),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure = ensure_dashboard()
    checks = [
        {
            "name": "health_endpoint",
            "pass": bool(ensure.get("ok")),
            **{k: v for k, v in ensure.items() if k != "ok"},
        }
    ]
    if ensure.get("ok"):
        checks.append(run_playwright())
    else:
        checks.append(
            {
                "name": "pw_dashboard_up",
                "pass": False,
                "error": "skipped — dashboard not healthy",
                "ensure": ensure,
            }
        )
    # still healthy after playwright?
    checks.append({"name": "health_after_pw", "pass": _http_ok(HEALTH)})

    passed = sum(1 for c in checks if c.get("pass"))
    total = len(checks)
    payload = {
        "checks": checks,
        "score": round(10.0 * passed / max(total, 1), 2),
        "pass": passed / max(total, 1) * 10 >= PASS,
        "gate": PASS,
        "passed": passed,
        "total": total,
        "url": URL,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "pw_dashboard_up_score.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
