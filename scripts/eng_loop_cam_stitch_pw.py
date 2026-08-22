#!/usr/bin/env python3
"""Playwright-only helper: screenshot Streamlit camera views (no OpenCV)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "reports/eval_match3/improve_eng_loop/cam_stitch_boxes"
URL = "http://127.0.0.1:8501/"
VIEWS = [
    "Whole pitch (4 cameras)",
    "P1 + P6 (ends)",
    "Best camera (ball)",
    "Only P10",
]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _launch(p):
    if Path(CHROME).is_file():
        return p.chromium.launch(
            headless=True, executable_path=CHROME, args=["--no-sandbox"]
        )
    return p.chromium.launch(headless=True)


def _select_camera_view(page, view: str) -> None:
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="Camera view")
    box.first.click(timeout=15_000)
    page.wait_for_timeout(500)
    opt = page.locator('[role="option"]').filter(has_text=view)
    opt.first.click(timeout=10_000)


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    shots = {}
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception as exc:
            (OUT / "pw_meta.json").write_text(
                json.dumps({"ok": False, "error": str(exc)}), encoding="utf-8"
            )
            browser.close()
            return 1
        page.wait_for_timeout(6000)
        # Wait for Camera view widget
        try:
            page.locator('[data-testid="stSelectbox"]').filter(
                has_text="Camera view"
            ).first.wait_for(timeout=45_000)
        except Exception as exc:
            (OUT / "pw_meta.json").write_text(
                json.dumps({"ok": False, "error": f"no Camera view: {exc}"}),
                encoding="utf-8",
            )
            browser.close()
            return 1

        for view in VIEWS:
            try:
                _select_camera_view(page, view)
            except Exception as exc:
                shots[view] = {"error": str(exc)}
                continue
            # RF-DETR on multi-cam can be slow
            page.wait_for_timeout(16_000)
            path = OUT / f"pw_{view.replace(' ', '_').replace('/', '-')[:48]}.png"
            page.screenshot(path=str(path), full_page=False)
            shots[view] = {"path": str(path)}
        browser.close()
    (OUT / "pw_meta.json").write_text(
        json.dumps({"ok": True, "shots": shots}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "n": len(shots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
