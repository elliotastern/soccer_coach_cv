#!/usr/bin/env python3
"""Smoke: accepted50 editor loads a frame and can enter new-box mode."""
from __future__ import annotations

import sys
from pathlib import Path

URL = (
    "http://127.0.0.1:8080/data/processed/gold_sets/"
    "match2_large_ball_accepted50/review/editor.html"
)
OUT = Path(__file__).resolve().parents[2] / "reports/match2_large_ball_harvest/accepted50_editor_smoke.png"


def main() -> int:
    from playwright.sync_api import sync_playwright

    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_function(
            """() => {
                const img = document.getElementById('reviewImage');
                return img && img.complete && img.naturalWidth > 100;
            }""",
            timeout=30000,
        )
        page.wait_for_timeout(400)
        info = page.evaluate(
            """() => {
                const img = document.getElementById('reviewImage');
                const canvas = document.getElementById('videoCanvas');
                const cs = getComputedStyle(document.querySelector('.container'));
                const ib = getComputedStyle(document.querySelector('.icon-button'));
                return {
                    nw: img.naturalWidth,
                    nh: img.naturalHeight,
                    canvasW: canvas.width,
                    canvasDispW: canvas.getBoundingClientRect().width,
                    containerH: cs.height,
                    iconW: ib.width,
                    boxes0: (boxes[0] || []).length,
                    mode: document.getElementById('statMode').textContent,
                };
            }"""
        )
        if info["nw"] < 100:
            failures.append(f"image not loaded: {info}")
        if info["canvasDispW"] < 200:
            failures.append(f"canvas too small: {info}")
        if "1080" in str(info.get("containerH", "")) and "px" in str(info["containerH"]):
            # 1080vh would compute to huge px; flag if > 5000
            pass
        ch = float(str(info["containerH"]).replace("px", "") or 0)
        if ch > 5000:
            failures.append(f"container height huge (CSS bug): {info['containerH']}")
        iw = float(str(info["iconW"]).replace("px", "") or 0)
        if iw > 200:
            failures.append(f"icon-button too wide: {info['iconW']}")

        # Enter new-box mode like Gold100 (N)
        page.evaluate("() => { if (typeof setLabel === 'function') setLabel('ball'); }")
        page.keyboard.press("n")
        page.wait_for_timeout(100)
        mode = page.evaluate("() => document.getElementById('statMode').textContent")
        if "new" not in mode.lower() and "draw" not in mode.lower() and "box" not in mode.lower():
            # check newBoxMode flag
            flag = page.evaluate("() => typeof newBoxMode !== 'undefined' && newBoxMode")
            if not flag:
                failures.append(f"N did not enter new-box mode (mode={mode}, flag={flag})")

        page.screenshot(path=str(OUT), full_page=False)

        # Draw a box (already in draw mode from N above)
        before = page.evaluate("() => (boxes[0] || []).length")
        box = page.evaluate(
            """() => {
                const c = document.getElementById('videoCanvas');
                const r = c.getBoundingClientRect();
                return {x: r.left + r.width * 0.4, y: r.top + r.height * 0.5, w: 80, h: 80};
            }"""
        )
        page.mouse.move(box["x"], box["y"])
        page.mouse.down()
        page.mouse.move(box["x"] + box["w"], box["y"] + box["h"], steps=6)
        page.mouse.up()
        page.wait_for_timeout(200)
        after = page.evaluate("() => (boxes[0] || []).length")
        if after < before + 1:
            failures.append(f"draw failed before={before} after={after}")
        frame_txt = page.evaluate(
            "() => document.querySelector('.frame-display')?.textContent || ''"
        )
        if "/ 49" not in frame_txt:
            failures.append(f"frame display wrong: {frame_txt}")

        browser.close()

    print("info:", info)
    print("mode after N:", mode)
    print("screenshot:", OUT)
    if failures:
        print("FAIL:", failures)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
