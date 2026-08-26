#!/usr/bin/env python3
"""Eng-loop: emit timeline labels ≥9/10 readable, zero overlap (Playwright)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.emit_timeline_layout import (  # noqa: E402
    build_timeline_html,
    count_layout_overlaps,
    layout_timeline_events,
    timeline_events_near,
    timeline_html_height,
)

OUT = ROOT / "reports/events_testing/emit_timeline_ux"
EMITS = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json"
PASS = 9.0
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _fixture_cases() -> list[dict]:
    emits = json.loads(EMITS.read_text(encoding="utf-8")).get("emits") or []
    shot = emits[0]
    cur_t = float(shot["t_end"])
    window_events = timeline_events_near(emits, cur_t)
    if shot not in window_events:
        window_events.append(shot)
        window_events = sorted(window_events, key=lambda e: float(e["t_end"]))
    max_s = 89.75
    step_s = 0.25
    return [
        {
            "name": "emit1_window",
            "events": window_events,
            "max_s": max_s,
            "step_s": step_s,
            "t_match": 1.8,
            "cur_t": cur_t,
            "cur_type": str(shot["type"]),
        },
        {
            "name": "crowded_movement",
            "events": [e for e in emits if float(e["t_end"]) <= 16.0][:6],
            "max_s": max_s,
            "step_s": step_s,
            "t_match": 7.5,
            "cur_t": 7.75,
            "cur_type": "movement",
        },
    ]


def _write_fixture(case: dict) -> Path:
    placed = layout_timeline_events(case["events"], case["max_s"])
    scrub_pct = case["t_match"] / case["max_s"] * 100.0
    html = build_timeline_html(
        placed,
        scrub_pct,
        case["t_match"],
        case["max_s"],
        case["step_s"],
        False,
        case["cur_t"],
        case["cur_type"],
        case["t_match"],
    )
    path = OUT / f"fixture_{case['name']}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _box_overlap(a: dict, b: dict) -> float:
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["width"], b["x"] + b["width"])
    y1 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _pw_measure(page, fixture_path: Path) -> dict:
    page.goto(fixture_path.as_uri(), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector(".ev-label", timeout=10_000)
    data = page.evaluate(
        """() => {
            const labels = [...document.querySelectorAll('.ev-label')];
            const boxes = labels.map((el, i) => {
                const r = el.getBoundingClientRect();
                const cs = getComputedStyle(el);
                return {
                    i,
                    text: el.textContent.trim(),
                    lane: el.dataset.lane,
                    x: r.x, y: r.y, width: r.width, height: r.height,
                    fontSize: cs.fontSize,
                };
            });
            const overlaps = [];
            for (let i = 0; i < boxes.length; i++) {
                for (let j = i + 1; j < boxes.length; j++) {
                    const a = boxes[i], b = boxes[j];
                    const x0 = Math.max(a.x, b.x);
                    const y0 = Math.max(a.y, b.y);
                    const x1 = Math.min(a.x + a.width, b.x + b.width);
                    const y1 = Math.min(a.y + a.height, b.y + b.height);
                    if (x1 > x0 && y1 > y0) {
                        overlaps.push({
                            a: a.text, b: b.text,
                            area: (x1 - x0) * (y1 - y0),
                        });
                    }
                }
            }
            const minFont = Math.min(...boxes.map(b => parseFloat(b.fontSize) || 0));
            return { boxes, overlaps, minFont, count: boxes.length };
        }"""
    )
    return data


def _score(measure: dict) -> float:
    overlaps = measure.get("overlaps") or []
    n = measure.get("count", 0)
    min_font = float(measure.get("minFont") or 0)
    score = 10.0
    score -= min(4.0, len(overlaps) * 2.0)
    if min_font < 12:
        score -= 1.5
    if n == 0:
        score = 0.0
    return max(0.0, min(10.0, score))


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    cases = _fixture_cases()
    offline = []
    for case in cases:
        placed = layout_timeline_events(case["events"], case["max_s"])
        offline.append(
            {
                "name": case["name"],
                "labels": len(placed),
                "lanes": max(r["lane"] for r in placed) + 1,
                "layout_overlaps": count_layout_overlaps(placed),
            }
        )

    pw_results = []
    with sync_playwright() as p:
        launch_kw = {"headless": True}
        if Path(CHROME).is_file():
            launch_kw["executable_path"] = CHROME
            launch_kw["args"] = ["--no-sandbox"]
        browser = p.chromium.launch(**launch_kw)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        for case in cases:
            fixture = _write_fixture(case)
            measure = _pw_measure(page, fixture)
            shot = OUT / f"pw_{case['name']}.png"
            page.screenshot(path=str(shot), full_page=False)
            score = _score(measure)
            pw_results.append(
                {
                    "name": case["name"],
                    "score": score,
                    "pass": score >= PASS and not measure.get("overlaps"),
                    "overlaps": measure.get("overlaps"),
                    "min_font": measure.get("minFont"),
                    "labels": measure.get("count"),
                    "shot": str(shot),
                    "fixture": str(fixture),
                }
            )
        browser.close()

    worst = min(r["score"] for r in pw_results)
    report = {
        "score": worst,
        "pass": all(r["pass"] for r in pw_results),
        "gate": PASS,
        "offline": offline,
        "playwright": pw_results,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
