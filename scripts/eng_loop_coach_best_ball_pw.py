#!/usr/bin/env python3
"""Eng-loop: best-ball cam shows visible ball ≥9/10 (offline + Playwright)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/events_testing/coach_best_ball_ux"
VIDEO = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_first_90s.mp4"
BATCH = ROOT / "data/output/match_4_5min"
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
    proc = subprocess.Popen(
        ["bash", str(LAUNCH)],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PORT": "8502", "PYTHONPATH": "."},
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
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


def offline_cases() -> list[dict]:
    return [
        {"name": "dribble_1_8s", "t": 1.8, "expect_quad": True},
        {"name": "early_1_25s", "t": 1.25, "expect_quad": True},
        {"name": "movement_3_5s", "t": 3.5, "expect_quad": True},
    ]


def run_offline(case: dict) -> dict:
    from apps.coach_emit_label_dashboard import (
        cached_read_frame,
        compose_best_ball_stack,
        load_video_meta,
        split_coach_stack,
        tile_ball_box_rect,
        tile_ball_box_score,
        vid_idx_for_match_t,
        QUAD_CAMS,
    )

    meta = load_video_meta(VIDEO)
    idx = vid_idx_for_match_t(meta, case["t"], 400)
    rgb = cached_read_frame(str(VIDEO), idx)
    stack, cam = compose_best_ball_stack(rgb, meta["start"] + idx * meta["stride"], BATCH)
    mosaic, _, _, offsets, tw, th = split_coach_stack(rgb)
    tiles = {
        c: mosaic[y:y + th, x:x + tw]
        for c, (x, y) in offsets.items()
    }
    quad_scores = {c: tile_ball_box_score(tiles[c]) for c in tiles}
    best_quad = max(quad_scores, key=lambda c: quad_scores[c])
    mh = mosaic.shape[0]
    top = stack[:mh]
    has_orange = tile_ball_box_rect(top) is not None
    ok_quad = cam in QUAD_CAMS and quad_scores.get(cam, 0) >= 80
    ok_ball = has_orange or tile_ball_box_rect(stack[: int(stack.shape[0] * 0.55)]) is not None
    if case.get("expect_quad"):
        pass_case = ok_quad and ok_ball
    else:
        pass_case = ok_ball
    return {
        "name": case["name"],
        "pass": pass_case,
        "cam": cam,
        "best_quad": best_quad,
        "quad_scores": quad_scores,
        "has_orange_top": has_orange,
    }


def _caption_cam(page) -> str | None:
    for txt in page.locator("[data-testid='stCaptionContainer']").all_inner_texts():
        if "cam" not in txt.lower():
            continue
        m = re.search(r"cam\s*\*\*([^*]+)\*\*", txt, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"cam\s+([P][\w_]+)", txt, re.I)
        if m:
            return m.group(1).strip()
    return None


def orange_score_bgr(bgr) -> int:
    import cv2
    import numpy as np

    mask = cv2.inRange(bgr, np.array([175, 85, 0]), np.array([255, 215, 95]))
    return int(mask.sum() // 255)


def _wait_dashboard(page, tries: int = 12, delay_ms: int = 1500) -> bool:
    for _ in range(tries):
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=20_000)
            return True
        except Exception:
            page.wait_for_timeout(delay_ms)
    return False


def _st_image_bgr(page):
    import cv2
    import numpy as np

    loc = page.locator('[data-testid="stImage"] img').first
    loc.wait_for(state="visible", timeout=30_000)
    raw = loc.screenshot()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_playwright_case(page, t: float) -> dict:
    if not _wait_dashboard(page):
        return {"name": f"pw_{t:.1f}s", "pass": False, "error": "dashboard not reachable"}
    page.goto(f"{URL}?scrub_t={t}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(5000)
    cam = _caption_cam(page)
    shot = OUT / f"pw_t{t:.1f}s.png"
    page.locator('[data-testid="stImage"] img').first.screenshot(path=str(shot))
    import cv2

    img = _st_image_bgr(page)
    h = img.shape[0]
    top = img[: int(h * 0.55), :]
    orange_px = orange_score_bgr(top)
    if t == 1.8:
        pass_case = cam in {"P7", "P10"} and orange_px >= 400
    else:
        pass_case = cam is not None and orange_px >= 400
    return {
        "name": f"pw_{t:.1f}s",
        "pass": pass_case,
        "cam": cam,
        "orange_px": orange_px,
        "shot": str(shot),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure = ensure_dashboard()
    checks: list[dict] = [{"name": "health_endpoint", "pass": bool(ensure.get("ok")), **ensure}]
    if ensure.get("ok"):
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = _launch(p)
                page = browser.new_page(viewport={"width": 1600, "height": 1100})
                checks.append(run_playwright_case(page, 1.8))
                browser.close()
        except Exception as exc:
            checks.append({"name": "pw_1.8s", "pass": False, "error": str(exc)})
    else:
        checks.append({"name": "pw_1.8s", "pass": False, "error": "dashboard not healthy"})
    checks.extend([run_offline(c) for c in offline_cases()])
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
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "pw_best_ball_score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
