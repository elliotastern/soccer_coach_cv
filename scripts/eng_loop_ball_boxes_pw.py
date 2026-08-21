#!/usr/bin/env python3
"""Playwright eng-loop: WHOLE PITCH mosaic must show an orange BALL bounding box (≥9/10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/eval_match3/improve_eng_loop/ball_boxes"
URL = "http://127.0.0.1:8501/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
GATE = 9.0


def _launch(p):
    if Path(CHROME).is_file():
        return p.chromium.launch(
            headless=True, executable_path=CHROME, args=["--no-sandbox"]
        )
    return p.chromium.launch(headless=True)


def _ensure_checkbox(page, label_substr: str, want_on: bool) -> None:
    """Toggle Streamlit checkbox by visible label text."""
    row = page.locator("[data-testid='stCheckbox']").filter(has_text=label_substr)
    row.first.wait_for(timeout=30_000)
    inp = row.locator("input").first
    checked = inp.is_checked()
    if checked != want_on:
        # Click the label / control
        row.locator("label").first.click(timeout=10_000)
        page.wait_for_timeout(1500)


def _show_sidebar_if_hidden(page) -> None:
    btn = page.get_by_role("button", name="Show sidebar")
    if btn.count() and btn.first.is_visible():
        btn.first.click()
        page.wait_for_timeout(1000)


def _orange_ball_signals(bgr: np.ndarray) -> dict:
    """Detect orange rectangles + BALL-ish orange ink in mosaic region."""
    h, w = bgr.shape[:2]
    crop = bgr[int(h * 0.08) : int(h * 0.92), 0 : int(w * 0.72)]
    # Prefer cv2; fall back to simple BGR threshold
    try:
        import cv2  # noqa: WPS433

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (5, 100, 140), (25, 255, 255))
        orange_px = int(mask.sum() // 255)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80 or area > 20000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 18 or bh < 18:
                continue
            aspect = bw / max(bh, 1)
            if 0.55 <= aspect <= 1.8:
                boxes += 1
    except Exception:
        # BGR orange ~ (0,165,255): high R+G, low B relative? In BGR: B low, G mid, R high
        b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        mask = (r > 180) & (g > 80) & (g < 220) & (b < 80)
        orange_px = int(mask.sum())
        boxes = 1 if orange_px >= 400 else 0
    return {
        "orange_px": orange_px,
        "orange_boxes": boxes,
        "crop_shape": list(crop.shape),
    }


def score_from_shot(path: Path) -> tuple[float, dict]:
    try:
        import cv2  # noqa: WPS433

        img = cv2.imread(str(path))
    except Exception:
        img = None
    if img is None:
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil)
        img = rgb[:, :, ::-1].copy()
    if img is None:
        return 0.0, {"error": "no image"}
    sig = _orange_ball_signals(img)
    has_ink = sig["orange_px"] >= 400
    has_box = sig["orange_boxes"] >= 1
    bits = [has_ink, has_box, img.shape[0] > 400]
    score = round(10.0 * sum(bits) / len(bits), 2)
    sig["bits"] = {"orange_ink": has_ink, "orange_box": has_box, "page_ok": bits[2]}
    return score, sig


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    notes = []
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        except Exception as exc:
            payload = {"score": 0.0, "pass": False, "error": f"goto: {exc}"}
            (OUT / "pw_ball_score.json").write_text(json.dumps(payload, indent=2))
            print(json.dumps(payload, indent=2))
            browser.close()
            return 1

        page.wait_for_timeout(5000)
        _show_sidebar_if_hidden(page)
        notes.append("sidebar ready")

        # Camera view should already be Whole pitch…
        try:
            page.locator("[data-testid='stSelectbox']").filter(
                has_text="Camera view"
            ).first.wait_for(timeout=60_000)
        except Exception as exc:
            payload = {"score": 0.0, "pass": False, "error": f"no camera view: {exc}"}
            (OUT / "pw_ball_score.json").write_text(json.dumps(payload, indent=2))
            print(json.dumps(payload, indent=2))
            browser.close()
            return 1

        _ensure_checkbox(page, "Defish", True)
        notes.append("defish on")
        _ensure_checkbox(page, "RF-DETR boxes", True)
        notes.append("boxes on")

        # Wait for RF-DETR multi-cam (can be slow on LaCie)
        page.wait_for_timeout(45_000)
        # Prefer mosaic image present
        try:
            page.locator("[data-testid='stImage']").first.wait_for(timeout=30_000)
        except Exception:
            notes.append("stImage wait soft-fail")

        shot = OUT / "pw_whole_pitch_ball.png"
        page.screenshot(path=str(shot), full_page=False)
        notes.append(f"shot {shot.name}")

        # Second shot after a bit more settle
        page.wait_for_timeout(8_000)
        shot2 = OUT / "pw_whole_pitch_ball_b.png"
        page.screenshot(path=str(shot2), full_page=False)

        browser.close()

    s1, sig1 = score_from_shot(shot)
    s2, sig2 = score_from_shot(shot2)
    score = max(s1, s2)
    best = shot2 if s2 >= s1 else shot
    best_sig = sig2 if s2 >= s1 else sig1
    payload = {
        "score": score,
        "pass": score >= GATE,
        "gate": GATE,
        "notes": notes,
        "best_shot": str(best),
        "sig": best_sig,
        "s1": s1,
        "s2": s2,
    }
    (OUT / "pw_ball_score.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"PW_BALL_SCORE {score}/10")
    return 0 if score >= GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
