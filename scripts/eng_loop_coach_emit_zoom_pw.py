#!/usr/bin/env python3
"""Eng-loop: coach emit label dashboard zoom stable ≥9/10 (Playwright)."""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/events_testing/coach_emit_zoom_ux"
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


def ensure_dashboard(timeout_s: float = 45.0) -> dict:
    if _http_ok(HEALTH):
        return {"ok": True, "started": False}
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "streamlit_launch.log"
    proc = __import__("subprocess").Popen(
        ["bash", str(LAUNCH)],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PORT": "8502", "PYTHONPATH": "."},
        stdout=log.open("w"),
        stderr=__import__("subprocess").STDOUT,
        start_new_session=True,
    )
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _http_ok(HEALTH):
            return {"ok": True, "started": True, "pid": proc.pid}
        if proc.poll() is not None:
            return {"ok": False, "started": True, "exit": proc.returncode}
        time.sleep(1.0)
    return {"ok": False, "started": True, "error": "timeout waiting health"}


def _launch(p):
    if Path(CHROME).is_file():
        return p.chromium.launch(
            headless=True, executable_path=CHROME, args=["--no-sandbox"]
        )
    return p.chromium.launch(headless=True)


def _parse_zoom(text: str) -> float | None:
    m = re.search(r"zoom\s*\*\*([\d.]+)x\*\*", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"zoom\s+([\d.]+)x", text, re.I)
    return float(m.group(1)) if m else None


def _caption_zoom(page) -> float | None:
    caps = page.locator("[data-testid='stCaptionContainer']")
    n = caps.count()
    for i in range(n - 1, -1, -1):
        txt = caps.nth(i).inner_text(timeout=3_000)
        if "zoom" not in txt.lower():
            continue
        z = _parse_zoom(txt)
        if z is not None:
            return z
    return None


def _st_image_width(page) -> float | None:
    loc = page.locator('[data-testid="stImage"] img').first
    if not loc.count():
        return None
    box = loc.bounding_box()
    return float(box["width"]) if box else None


def _click_button(page, label: str) -> None:
    page.get_by_role("button", name=label, exact=False).first.click(timeout=15_000)
    page.wait_for_timeout(1500)


def run_checks(page) -> list[dict]:
    results: list[dict] = []
    page.goto(f"{URL}?scrub_t=1.8", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(4000)

    z0 = _caption_zoom(page)
    w0 = _st_image_width(page)
    results.append({"name": "page_load", "pass": z0 is not None and w0 is not None, "zoom": z0, "w": w0})

    _click_button(page, "Zoom +")
    z1 = _caption_zoom(page)
    w1 = _st_image_width(page)
    results.append(
        {
            "name": "zoom_plus_once",
            "pass": z0 is not None and z1 is not None and z1 > z0,
            "zoom": z1,
            "w": w1,
        }
    )

    _click_button(page, "Zoom +")
    z2 = _caption_zoom(page)
    w2 = _st_image_width(page)
    results.append(
        {
            "name": "zoom_plus_twice",
            "pass": z1 is not None and z2 is not None and z2 > z1,
            "zoom": z2,
            "w": w2,
        }
    )

    _click_button(page, "Zoom −")
    z3 = _caption_zoom(page)
    results.append(
        {
            "name": "zoom_minus",
            "pass": z2 is not None and z3 is not None and z3 < z2,
            "zoom": z3,
        }
    )

    has_slider = page.locator("[data-testid='stSlider']").count() > 0
    results.append({"name": "zoom_slider_present", "pass": has_slider})

    for _ in range(5):
        _click_button(page, "Zoom +")
    z_many = _caption_zoom(page)
    w_many = _st_image_width(page)
    results.append(
        {
            "name": "zoom_plus_many",
            "pass": z_many is not None and z_many >= 2.2,
            "zoom": z_many,
            "w": w_many,
        }
    )

    page.goto(f"{URL}?scrub_t=1.8&zoom_val=1.0", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(3500)
    w_low = _st_image_width(page)
    page.goto(f"{URL}?scrub_t=1.8&zoom_val=2.5", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(3500)
    w_high = _st_image_width(page)
    z_high = _caption_zoom(page)
    results.append(
        {
            "name": "query_zoom_widens_image",
            "pass": (
                w_low is not None
                and w_high is not None
                and w_high >= w_low
                and z_high is not None
                and z_high >= 2.4
            ),
            "w_low": w_low,
            "w_high": w_high,
            "zoom": z_high,
        }
    )

    page.goto(f"{URL}?scrub_t=3.5&zoom_val=2.0", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(3500)
    z_scrub = _caption_zoom(page)
    results.append(
        {
            "name": "zoom_persists_after_scrub",
            "pass": z_scrub is not None and z_scrub >= 1.95,
            "zoom": z_scrub,
        }
    )

    page.goto(f"{URL}?scrub_t=1.8&zoom_step=0.5", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(3500)
    z_step = _caption_zoom(page)
    results.append(
        {
            "name": "zoom_step_query",
            "pass": z_step is not None and z_step >= 1.45,
            "zoom": z_step,
        }
    )

    results.append(
        {
            "name": "image_wider_than_default",
            "pass": (
                w0 is not None
                and w_many is not None
                and w_many >= w0
                and z_many is not None
                and z_many >= 2.0
            ),
            "w0": w0,
            "w_many": w_many,
            "zoom": z_many,
        }
    )

    return results


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    shot = OUT / "pw_zoom_dashboard.png"
    ensure = ensure_dashboard()
    payload: dict = {
        "ensure": ensure,
        "checks": [],
        "score": 0.0,
        "pass": False,
        "gate": PASS,
    }

    if not ensure.get("ok"):
        payload["error"] = "dashboard not healthy"
        out_path = OUT / "pw_zoom_score.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 1

    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        try:
            payload["checks"] = run_checks(page)
            page.screenshot(path=str(shot), full_page=True)
            payload["shot"] = str(shot)
        except Exception as exc:
            payload["error"] = str(exc)
        browser.close()

    payload["checks"].append({"name": "health_after_pw", "pass": _http_ok(HEALTH)})
    passed = sum(1 for c in payload.get("checks", []) if c.get("pass"))
    total = len(payload.get("checks", []))
    payload["score"] = round(10.0 * passed / max(total, 1), 2)
    payload["pass"] = payload["score"] >= PASS
    payload["passed"] = passed
    payload["total"] = total
    payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_path = OUT / "pw_zoom_score.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
