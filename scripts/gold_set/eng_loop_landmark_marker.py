#!/usr/bin/env python3
"""Playwright shots of landmark_marker names panel (readability loop)."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/eval_match3/landmark_dashboard/eng_loop"
URL = "http://127.0.0.1:8080/reports/eval_match3/landmark_dashboard/index.html?v=names-html21b"
CAMS = ["P10", "P7", "P9", "P8", "P1", "P6"]
WANT = {
    "P10": ["Halfway Left Sideline", "South Left Corner", "Center Spot", "Center Circle Left"],
    "P7": ["Halfway Right Sideline", "South Right Corner", "Center Spot", "Center Circle Right"],
    "P9": ["Halfway Left Sideline", "North Left Corner", "Center Spot", "Center Circle Left"],
    "P8": ["Halfway Right Sideline", "North Right Corner", "Center Spot", "Center Circle Right"],
    "P1": [
        "South Left Goal-Line Corner",
        "South Right Goal-Line Corner",
        "South Left 18-Yard Corner",
        "South Penalty Spot",
    ],
    "P6": [
        "North Left Goal-Line Corner",
        "North Right Goal-Line Corner",
        "North Left 18-Yard Corner",
        "North Penalty Spot",
    ],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#names")
        page.wait_for_timeout(350)
        missing = []
        for cam in CAMS:
            page.locator(f'.cams button[data-id="{cam}"]').click()
            page.wait_for_selector("#nameLabels .name-tag")
            page.wait_for_timeout(200)
            labels = page.evaluate(
                """() => [...document.querySelectorAll('#nameLabels .name-tag, #names text')]
                    .map(t => (t.textContent || '').trim()).filter(Boolean)"""
            )
            joined = " ".join(labels)
            page.locator(".work").screenshot(path=str(OUT / f"mark_{cam}.png"))
            page.locator(".map-panel").screenshot(path=str(OUT / f"names_{cam}.png"))
            task = page.locator("#taskWhat").inner_text()
            find = page.locator("#taskFind").inner_text()
            if WANT[cam][0] not in task:
                missing.append(f"{cam}: task '{task}' missing {WANT[cam][0]}")
            if len(find) < 12:
                missing.append(f"{cam}: find hint too short: {find}")
            overlap = page.evaluate(
                """() => {
                  const pills = [...document.querySelectorAll('#nameLabels .name-tag')];
                  const boxes = pills.map(r => r.getBoundingClientRect());
                  for (let i = 0; i < boxes.length; i++) {
                    for (let j = i + 1; j < boxes.length; j++) {
                      const a = boxes[i], b = boxes[j];
                      const hit = a.left < b.right && a.right > b.left &&
                        a.top < b.bottom && a.bottom > b.top;
                      if (hit) return `${i}:${j}`;
                    }
                  }
                  return '';
                }"""
            )
            if overlap:
                missing.append(f"{cam}: pill overlap {overlap}")
            sizes = page.evaluate(
                """() => [...document.querySelectorAll('#nameLabels .name-tag')].map(t => {
                    const b = t.getBoundingClientRect();
                    const fs = parseFloat(getComputedStyle(t).fontSize);
                    return { t: t.textContent.trim(), h: b.height, fs };
                  })"""
            )
            print(cam, sizes)
            for row in sizes:
                if row["fs"] < 20 or row["h"] < 28:
                    missing.append(
                        f"{cam}: '{row['t']}' fs={row['fs']:.1f} h={row['h']:.1f}"
                    )
            for n in WANT[cam]:
                if n not in joined:
                    missing.append(f"{cam}: missing '{n}' in {labels}")
        browser.close()
    if missing:
        print("READABILITY MISS")
        for m in missing:
            print(" ", m)
        return 1
    print("pretty names present, no pill overlap")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
