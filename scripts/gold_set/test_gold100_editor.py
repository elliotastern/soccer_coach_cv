#!/usr/bin/env python3
"""Playwright smoke test: load gold100 editor, drag a box, draw a box, seek frames."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8080/annotation/gold100_editor.html"


def main():
    from playwright.sync_api import sync_playwright

    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=60000)

        page.wait_for_function(
            "() => typeof boxes !== 'undefined' && Object.keys(boxes).length > 0 && "
            "document.getElementById('reviewImage')?.complete",
            timeout=30000,
        )
        page.wait_for_timeout(500)

        overlay = page.locator("#playOverlay")
        display = overlay.evaluate("el => getComputedStyle(el).display")
        if display != "none":
            failures.append(f"playOverlay visible: {display}")

        page.keyboard.press("Escape")
        page.wait_for_timeout(100)
        page.evaluate("seekToFrame(5)")
        page.wait_for_timeout(400)
        src5 = page.evaluate("() => document.getElementById('reviewImage').src")
        if "005.jpg" not in src5:
            failures.append(f"seekToFrame(5) src={src5}")

        frame_with_box = page.evaluate(
            """() => {
                for (const k of Object.keys(boxes)) {
                    if (boxes[k] && boxes[k].length) return parseInt(k, 10);
                }
                return -1;
            }"""
        )
        if frame_with_box < 0:
            failures.append("no boxes loaded from XML")
        else:
            page.evaluate("f => seekToFrame(f)", frame_with_box)
            page.wait_for_timeout(400)

            before = page.evaluate(
                """f => {
                    const b = boxes[f][0];
                    return {xtl: b.xtl, ytl: b.ytl, xbr: b.xbr, ybr: b.ybr};
                }""",
                frame_with_box,
            )

            drag = page.evaluate(
                """f => {
                    const b = boxes[f][0];
                    const canvas = document.getElementById('videoCanvas');
                    const rect = canvas.getBoundingClientRect();
                    const cx = (b.xtl + b.xbr) / 2;
                    const cy = (b.ytl + b.ybr) / 2;
                    const bufferX = (cx - videoWidth / 2) * zoomLevel + canvas.width / 2 + panX;
                    const bufferY = (cy - videoHeight / 2) * zoomLevel + canvas.height / 2 + panY;
                    return {
                        x: rect.left + bufferX * (rect.width / canvas.width),
                        y: rect.top + bufferY * (rect.height / canvas.height),
                    };
                }""",
                frame_with_box,
            )
            page.mouse.move(drag["x"], drag["y"])
            page.mouse.down()
            page.mouse.move(drag["x"] + 40, drag["y"] + 30, steps=5)
            page.mouse.up()
            page.wait_for_timeout(200)

            after = page.evaluate(
                """f => {
                    const b = boxes[f][0];
                    return {xtl: b.xtl, ytl: b.ytl, xbr: b.xbr, ybr: b.ybr};
                }""",
                frame_with_box,
            )
            moved = abs(after["xtl"] - before["xtl"]) > 5 or abs(after["ytl"] - before["ytl"]) > 5
            if not moved:
                # Diagnose hit-test
                hit = page.evaluate(
                    """([f, x, y]) => {
                        const pos = screenToCanvas(x, y);
                        const b = boxes[f][0];
                        return {
                            pos, before: b,
                            inside: pos.x >= b.xtl && pos.x <= b.xbr && pos.y >= b.ytl && pos.y <= b.ybr,
                            isDragging, zoomLevel, panX, panY,
                            canvas: {
                                w: videoCanvas.width, h: videoCanvas.height,
                                sw: videoCanvas.getBoundingClientRect().width,
                                sh: videoCanvas.getBoundingClientRect().height,
                            },
                        };
                    }""",
                    [frame_with_box, drag["x"], drag["y"]],
                )
                failures.append(f"box did not move: before={before} after={after} hit={hit}")

        page.evaluate("() => seekToFrame(0)")
        page.wait_for_timeout(300)
        page.keyboard.press("n")
        page.wait_for_timeout(100)
        mode = page.evaluate("() => newBoxMode")
        if not mode:
            failures.append("N did not enable newBoxMode")
        else:
            canvas = page.locator("#videoCanvas")
            box = canvas.bounding_box()
            x0, y0 = box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2
            page.mouse.move(x0, y0)
            page.mouse.down()
            page.mouse.move(x0 + 80, y0 + 60, steps=5)
            page.mouse.up()
            page.wait_for_timeout(200)
            has_new = page.evaluate(
                "() => (boxes[0] || []).some(b => String(b.id).startsWith('new-'))"
            )
            if not has_new:
                n_boxes = page.evaluate("() => (boxes[0] || []).length")
                failures.append(f"new box not created; boxes[0]={n_boxes}")

        # Resize via SE corner drag + keyboard nudge (pick an in-bounds box)
        if frame_with_box >= 0:
            page.evaluate("f => seekToFrame(f)", frame_with_box)
            page.wait_for_timeout(300)
            box_idx = page.evaluate(
                """f => {
                    const list = boxes[f] || [];
                    for (let i = 0; i < list.length; i++) {
                        const b = list[i];
                        if (b.xbr < videoWidth - 50 && b.ybr < videoHeight - 50 && b.xtl > 20 && b.ytl > 20) return i;
                    }
                    return 0;
                }""",
                frame_with_box,
            )
            before_r = page.evaluate(
                """([f, i]) => {
                    const b = boxes[f][i];
                    selectedBox = b;
                    return {xbr: b.xbr, ybr: b.ybr, xtl: b.xtl};
                }""",
                [frame_with_box, box_idx],
            )
            se = page.evaluate(
                """([f, i]) => {
                    const b = boxes[f][i];
                    const canvas = document.getElementById('videoCanvas');
                    const rect = canvas.getBoundingClientRect();
                    const bufferX = (b.xbr - videoWidth / 2) * zoomLevel + canvas.width / 2 + panX;
                    const bufferY = (b.ybr - videoHeight / 2) * zoomLevel + canvas.height / 2 + panY;
                    return {
                        x: rect.left + bufferX * (rect.width / canvas.width),
                        y: rect.top + bufferY * (rect.height / canvas.height),
                    };
                }""",
                [frame_with_box, box_idx],
            )
            page.mouse.move(se["x"], se["y"])
            page.mouse.down()
            page.mouse.move(se["x"] + 30, se["y"] + 25, steps=5)
            page.mouse.up()
            page.wait_for_timeout(200)
            after_r = page.evaluate(
                """([f, i]) => {
                    const b = boxes[f][i];
                    return {xbr: b.xbr, ybr: b.ybr};
                }""",
                [frame_with_box, box_idx],
            )
            if abs(after_r["xbr"] - before_r["xbr"]) < 5 and abs(after_r["ybr"] - before_r["ybr"]) < 5:
                failures.append(f"resize did not change box: before={before_r} after={after_r}")

            page.evaluate(
                """([f, i]) => { selectedBox = boxes[f][i]; }""",
                [frame_with_box, box_idx],
            )
            xtl_before_arrow = page.evaluate("([f, i]) => boxes[f][i].xtl", [frame_with_box, box_idx])
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(50)
            xtl_after_arrow = page.evaluate("([f, i]) => boxes[f][i].xtl", [frame_with_box, box_idx])
            if abs(xtl_after_arrow - xtl_before_arrow) < 0.5:
                failures.append(f"arrow nudge failed: {xtl_before_arrow} -> {xtl_after_arrow}")

        # Save must not change in-memory box coords or currentFrame
        if frame_with_box >= 0:
            page.evaluate("f => seekToFrame(f)", frame_with_box)
            page.wait_for_timeout(300)
            page.evaluate(
                """f => {
                    const b = boxes[f][0];
                    b.xtl += 17;
                    b.xbr += 17;
                    selectedBox = b;
                }""",
                frame_with_box,
            )
            before_save = page.evaluate(
                """() => ({
                    frame: currentFrame,
                    box: {xtl: boxes[currentFrame][0].xtl, ytl: boxes[currentFrame][0].ytl},
                    src: document.getElementById('reviewImage').src,
                })"""
            )
            save_result = page.evaluate(
                """async () => {
                    const orig = window.fetch;
                    window.fetch = async (url, opts) => {
                        if (url === '/save_annotations') {
                            const body = JSON.parse(opts.body);
                            body.file_path = 'data/processed/gold_sets/match1_1_100/review/_test_save.xml';
                            return orig(url, { ...opts, body: JSON.stringify(body) });
                        }
                        return orig(url, opts);
                    };
                    try {
                        await new Promise((resolve, reject) => {
                            const _alert = window.alert;
                            window.alert = (msg) => { if (String(msg).includes('Error')) reject(new Error(msg)); };
                            const prev = debugLog;
                            let saw = false;
                            window.debugLog = (m) => { if (String(m).includes('without reload')) saw = true; prev(m); };
                            saveAnnotations();
                            setTimeout(() => {
                                window.alert = _alert;
                                window.debugLog = prev;
                                if (saw) resolve('ok');
                                else resolve('no-soft-save');
                            }, 800);
                        });
                        return 'ok';
                    } catch (e) {
                        return String(e);
                    } finally {
                        window.fetch = orig;
                    }
                }"""
            )
            page.wait_for_timeout(200)
            after_save = page.evaluate(
                """() => ({
                    frame: currentFrame,
                    box: {xtl: boxes[currentFrame][0].xtl, ytl: boxes[currentFrame][0].ytl},
                    src: document.getElementById('reviewImage').src,
                })"""
            )
            if save_result != "ok":
                failures.append(f"save failed: {save_result}")
            if after_save["frame"] != before_save["frame"]:
                failures.append(f"save changed frame {before_save['frame']} -> {after_save['frame']}")
            if abs(after_save["box"]["xtl"] - before_save["box"]["xtl"]) > 0.01:
                failures.append(f"save moved box {before_save['box']} -> {after_save['box']}")
            if before_save["src"].split("?")[0] != after_save["src"].split("?")[0]:
                failures.append(f"save changed image {before_save['src']} -> {after_save['src']}")

        browser.close()

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("PASS: seek, drag-move, resize, arrow-nudge, draw-box, save, overlay hidden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
