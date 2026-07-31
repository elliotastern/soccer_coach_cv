#!/usr/bin/env python3
"""Rebuild gold100 review frames + image-sequence editor; validate alignment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_GOLD = ROOT / "data/processed/gold_sets/match1_1_100"
EDITOR_TEMPLATE = ROOT / "annotation/view_annotations_editor.html"
STRIP_MAX_WIDTH = 1920


def resize_for_strip(frame: np.ndarray, max_width: int = STRIP_MAX_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def write_review_frames(gold_dir: Path) -> tuple[list[str], int, int]:
    manifest = json.loads((gold_dir / "manifest.json").read_text())
    frames_dir = gold_dir / "review" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = []
    width = height = None
    for row in manifest["frames"]:
        idx = int(row["strip_frame"])
        src = cv2.imread(str(gold_dir / "images" / row["image"]))
        if src is None:
            raise RuntimeError(f"Missing image: {row['image']}")
        strip = resize_for_strip(src)
        height, width = strip.shape[:2]
        name = f"{idx:03d}.jpg"
        cv2.imwrite(str(frames_dir / name), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        names.append(name)
    return names, width, height


def write_strip_from_frames(gold_dir: Path, n: int, fps: int = 60) -> None:
    frames_dir = gold_dir / "review" / "frames"
    out = gold_dir / "review" / "strip_100.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%03d.jpg"),
        "-frames:v", str(n),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-g", "1", "-keyint_min", "1", "-bf", "0",
        "-x264-params", "keyint=1:min-keyint=1:scenecut=0",
        "-movflags", "+faststart", "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])


def clamp_xml_boxes(gold_dir: Path, width: int, height: int) -> int:
    import xml.etree.ElementTree as ET

    xml_path = gold_dir / "prelabels" / "annotations.xml"
    tree = ET.parse(xml_path)
    fixed = 0
    for box in tree.getroot().findall(".//box"):
        xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
        xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
        nxtl = min(max(xtl, 0.0), width - 1.0)
        nytl = min(max(ytl, 0.0), height - 1.0)
        nxbr = min(max(xbr, nxtl + 1.0), float(width))
        nybr = min(max(ybr, nytl + 1.0), float(height))
        if (nxtl, nytl, nxbr, nybr) != (xtl, ytl, xbr, ybr):
            box.set("xtl", f"{nxtl:.2f}")
            box.set("ytl", f"{nytl:.2f}")
            box.set("xbr", f"{nxbr:.2f}")
            box.set("ybr", f"{nybr:.2f}")
            fixed += 1
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return fixed


def _must_replace(html: str, old: str, new: str, label: str) -> str:
    if old not in html:
        raise RuntimeError(f"Editor patch failed: missing block [{label}]")
    return html.replace(old, new, 1)


def build_image_editor(gold_dir: Path, frame_names: list[str], width: int, height: int) -> Path:
    gold_dir = gold_dir.resolve()
    rel = gold_dir.relative_to(ROOT.resolve()).as_posix()
    last = len(frame_names) - 1
    xml_url = f"/{rel}/prelabels/annotations.xml"
    xml_save = f"{rel}/prelabels/annotations.xml"
    frames_url = f"/{rel}/review/frames/"
    files_json = json.dumps(frame_names)

    html = EDITOR_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(
        "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4",
        f"/{rel}/review/strip_100.mp4",
    )
    html = html.replace(
        "data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506_annotations.xml",
        xml_url,
    )
    html = html.replace(
        "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF_annotations.xml",
        xml_save,
    )
    html = html.replace('max="796"', f'max="{last}"')
    html = html.replace(" / 796", f" / {last}")
    html = html.replace("seekToFrame(796)", f"seekToFrame({last})")
    html = html.replace(
        "Math.min(parseInt(frameNum), 796)",
        f"Math.min(parseInt(frameNum), {last})",
    )
    html = html.replace("if (currentFrame < 796)", f"if (currentFrame < {last})")

    html = _must_replace(
        html,
        '<canvas id="videoCanvas"></canvas>',
        '<canvas id="videoCanvas"></canvas>\n'
        '                    <img id="reviewImage" alt="" style="display:none">',
        "canvas+img",
    )
    html = _must_replace(
        html,
        "#playOverlay {\n            position: absolute;",
        "#playOverlay {\n            display: none !important;\n"
        "            pointer-events: none !important;\n            position: absolute;",
        "playOverlay css",
    )
    html = _must_replace(
        html,
        "video {\n            display: none;\n        }",
        "video { display: none !important; pointer-events: none !important; }\n"
        "        #reviewImage { display: none !important; }",
        "video css",
    )

    gold_boot = f"""
        // ===== GOLD100 image-sequence boot =====
        const GOLD100 = {{
            files: {files_json},
            base: '{frames_url}',
            width: {width},
            height: {height},
            maxFrame: {last},
        }};
        let reviewImage = null;
        let _frameLoadToken = 0;
        function loadReviewFrame(idx) {{
            if (!reviewImage) reviewImage = document.getElementById('reviewImage');
            const token = ++_frameLoadToken;
            const src = GOLD100.base + GOLD100.files[idx] + '?v=1';
            return new Promise((resolve, reject) => {{
                const done = () => {{ if (token === _frameLoadToken) resolve(); }};
                reviewImage.onload = done;
                reviewImage.onerror = () => reject(new Error('frame load failed: ' + src));
                if (reviewImage.getAttribute('data-idx') === String(idx) && reviewImage.complete) {{
                    done();
                    return;
                }}
                reviewImage.setAttribute('data-idx', String(idx));
                reviewImage.src = src;
            }});
        }}
        // ===== end GOLD100 boot =====
"""
    html = _must_replace(html, "let annotations = null;", gold_boot + "\n        let annotations = null;", "boot")

    html = _must_replace(
        html,
        """                // Draw video frame - ensure video is ready and seeked
                if (video.readyState >= 2) {
                    try {
                        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
                    } catch (e) {
                        debugLog('ERROR drawing video frame: ' + e.message);
                        // If drawImage fails, fill with dark gray to show something
                        ctx.fillStyle = '#1a1a1a';
                        ctx.fillRect(0, 0, videoWidth, videoHeight);
                    }
                } else {
                    // Video not ready yet, fill with dark background
                    ctx.fillStyle = '#1a1a1a';
                    ctx.fillRect(0, 0, videoWidth, videoHeight);
                }""",
        """                // Draw exact review JPG for this strip frame
                if (reviewImage && reviewImage.complete && reviewImage.naturalWidth > 0) {
                    ctx.drawImage(reviewImage, 0, 0, videoWidth, videoHeight);
                } else {
                    ctx.fillStyle = '#1a1a1a';
                    ctx.fillRect(0, 0, videoWidth, videoHeight);
                }""",
        "drawImage",
    )

    # After max-frame replace, seekToFrame uses `last` not 796
    html = _must_replace(
        html,
        f"""        function seekToFrame(frameNum) {{
            currentFrame = Math.max(0, Math.min(parseInt(frameNum), {last}));
            document.getElementById('frameSlider').value = currentFrame;
            document.getElementById('currentFrame').textContent = currentFrame;
            if (video.readyState >= 2) {{
                video.currentTime = currentFrame / 60;
                video.pause();
            }}
            selectedBox = null;
            updateLabelSelector();
            drawFrame();
            updateBoxesList();
        }}""",
        """        function seekToFrame(frameNum) {
            currentFrame = Math.max(0, Math.min(parseInt(frameNum), GOLD100.maxFrame));
            document.getElementById('frameSlider').value = currentFrame;
            document.getElementById('currentFrame').textContent = currentFrame;
            selectedBox = null;
            updateLabelSelector();
            updateBoxesList();
            loadReviewFrame(currentFrame).then(() => {
                drawFrame();
            }).catch((e) => {
                debugLog('ERROR seek: ' + e.message);
                drawFrame();
            });
        }""",
        "seekToFrame",
    )

    html = _must_replace(
        html,
        """        // Load video and annotations
        video = document.getElementById('video');
        canvas = document.getElementById('videoCanvas');
        ctx = canvas.getContext('2d');
        
        debugLog('Initialized video and canvas elements');

        video.addEventListener('loadedmetadata', () => {
            videoWidth = video.videoWidth;
            videoHeight = video.videoHeight;
            debugLog(`Video metadata loaded: ${videoWidth}x${videoHeight}`);
            
            if (videoWidth === 0 || videoHeight === 0) {
                debugLog('ERROR: Video dimensions are 0!');
                return;
            }
            
            canvas.width = videoWidth;
            canvas.height = videoHeight;
            
            const wrapper = canvas.parentElement;
            const containerWidth = wrapper.clientWidth || 800;
            const containerHeight = wrapper.clientHeight || 600;
            
            scaleX = containerWidth / videoWidth;
            scaleY = containerHeight / videoHeight;
            const scale = Math.min(scaleX, scaleY);
            
            canvas.style.width = (videoWidth * scale) + 'px';
            canvas.style.height = (videoHeight * scale) + 'px';
            
            // Ensure video is at frame 0
            video.currentTime = 0;
            video.pause();
            
            // Wait for video to be ready, then load annotations
            const onCanPlay = () => {
                video.removeEventListener('canplay', onCanPlay);
                loadAnnotations();
                // Seek to frame 0 and draw
                seekToFrame(0);
            };
            video.addEventListener('canplay', onCanPlay, { once: true });
            
            // Fallback: if video is already ready
            if (video.readyState >= 3) {
                video.removeEventListener('canplay', onCanPlay);
                loadAnnotations();
                seekToFrame(0);
            }
            
            requestAnimationFrame(drawLoop);
            updateTimelineMarkers();
        });""",
        """        // Load image-sequence review + annotations
        video = document.getElementById('video');
        reviewImage = document.getElementById('reviewImage');
        canvas = document.getElementById('videoCanvas');
        ctx = canvas.getContext('2d');
        debugLog('GOLD100 image editor init');

        videoWidth = GOLD100.width;
        videoHeight = GOLD100.height;
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        const wrapperInit = canvas.parentElement;
        const containerWidth = wrapperInit.clientWidth || 800;
        const containerHeight = wrapperInit.clientHeight || 600;
        scaleX = containerWidth / videoWidth;
        scaleY = containerHeight / videoHeight;
        const fit = Math.min(scaleX, scaleY);
        canvas.style.width = (videoWidth * fit) + 'px';
        canvas.style.height = (videoHeight * fit) + 'px';

        const overlay = document.getElementById('playOverlay');
        if (overlay) overlay.style.display = 'none';

        loadAnnotations();
        seekToFrame(0);
        requestAnimationFrame(drawLoop);
        updateTimelineMarkers();""",
        "init",
    )

    html = _must_replace(
        html,
        """        function drawLoop() {
            if (video.readyState >= 2 && videoWidth > 0 && videoHeight > 0) {
                if (!video.paused) {
                    drawFrame();
                } else {
                    drawFrame();
                }
            }
            requestAnimationFrame(drawLoop);
        }""",
        """        function drawLoop() {
            if (videoWidth > 0 && videoHeight > 0) {
                drawFrame();
            }
            requestAnimationFrame(drawLoop);
        }""",
        "drawLoop",
    )

    html = _must_replace(
        html,
        """                rect.id = `new-${Date.now()}`;
                rect.trackId = `track-${Date.now()}`;
                
                // Save state before adding box
                saveState();
                
                boxes[currentFrame].push(rect);""",
        """                rect.id = `new-${Date.now()}`;
                rect.trackId = `track-${Date.now()}`;
                const labelSel = document.getElementById('labelDropdown');
                rect.label = (labelSel && labelSel.value) ? labelSel.value : 'player';
                rect.confidence = 1.0;
                trackMap[rect.trackId] = { label: rect.label };
                
                // Save state before adding box
                saveState();
                
                boxes[currentFrame].push(rect);""",
        "new-box-label",
    )

    html = html.replace(
        "document.getElementById('playOverlay').style.display = 'block';",
        "/* playOverlay disabled */",
    )
    html = html.replace("video.load();", "/* video.load disabled */")

    # Canvas buffer coords (not CSS display size) for draw + hit-test
    html = _must_replace(
        html,
        """                // Get canvas display dimensions
                const rect = canvas.getBoundingClientRect();
                const displayWidth = rect.width;
                const displayHeight = rect.height;
                
                // Calculate center of visible area
                const centerX = displayWidth / 2;
                const centerY = displayHeight / 2;
                
                // Apply transformations centered on the canvas
                ctx.save();
                
                // Move to center, scale, then move back with pan offset
                ctx.translate(centerX + panX, centerY + panY);
                ctx.scale(zoomLevel, zoomLevel);
                ctx.translate(-videoWidth / 2, -videoHeight / 2);""",
        """                // Transform in canvas buffer space (matches screenToCanvas)
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                ctx.save();
                ctx.translate(centerX + panX, centerY + panY);
                ctx.scale(zoomLevel, zoomLevel);
                ctx.translate(-videoWidth / 2, -videoHeight / 2);""",
        "draw-transform",
    )
    html = _must_replace(
        html,
        """        function screenToCanvas(x, y) {
            if (!canvas || videoWidth === 0 || videoHeight === 0) {
                return { x: 0, y: 0 };
            }
            
            const rect = canvas.getBoundingClientRect();
            const displayWidth = rect.width;
            const displayHeight = rect.height;
            const centerX = displayWidth / 2;
            const centerY = displayHeight / 2;
            
            // Convert screen coordinates to canvas coordinates accounting for zoom/pan
            const canvasX = (x - rect.left - centerX - panX) / zoomLevel + videoWidth / 2;
            const canvasY = (y - rect.top - centerY - panY) / zoomLevel + videoHeight / 2;
            
            return { x: canvasX, y: canvasY };
        }""",
        """        function screenToCanvas(x, y) {
            if (!canvas || videoWidth === 0 || videoHeight === 0) {
                return { x: 0, y: 0 };
            }
            const rect = canvas.getBoundingClientRect();
            if (rect.width < 1 || rect.height < 1) return { x: 0, y: 0 };
            // CSS pixels -> canvas buffer pixels, then inverse zoom/pan
            const bufferX = (x - rect.left) * (canvas.width / rect.width);
            const bufferY = (y - rect.top) * (canvas.height / rect.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            return {
                x: (bufferX - centerX - panX) / zoomLevel + videoWidth / 2,
                y: (bufferY - centerY - panY) / zoomLevel + videoHeight / 2,
            };
        }""",
        "screenToCanvas",
    )
    html = _must_replace(
        html,
        """            if (isPanning && panStart) {
                panX += e.clientX - panStart.x;
                panY += e.clientY - panStart.y;
                panStart = {x: e.clientX, y: e.clientY};
                drawFrame();
                return;
            }""",
        """            if (isPanning && panStart) {
                const rect = canvas.getBoundingClientRect();
                const sx = canvas.width / Math.max(rect.width, 1);
                const sy = canvas.height / Math.max(rect.height, 1);
                panX += (e.clientX - panStart.x) * sx;
                panY += (e.clientY - panStart.y) * sy;
                panStart = {x: e.clientX, y: e.clientY};
                drawFrame();
                return;
            }""",
        "pan-scale",
    )
    html = _must_replace(
        html,
        """                const baseHandleSize = 3;
                const handleSize = Math.max(baseHandleSize, baseHandleSize * zoomLevel);""",
        """                const baseHandleSize = 10;
                const handleSize = Math.max(baseHandleSize, baseHandleSize / zoomLevel);""",
        "handle-size",
    )
    html = _must_replace(
        html,
        """            const hitRadius = Math.max(10, 10 * zoomLevel) / 2; // 10px hit area for corners
            const edgeHitDistance = Math.max(8, 8 * zoomLevel) / 2; // 8px hit area for edges""",
        """            const hitRadius = Math.max(14, 14 / zoomLevel);
            const edgeHitDistance = Math.max(10, 10 / zoomLevel);""",
        "handle-hit",
    )
    html = _must_replace(
        html,
        "#videoCanvas {\n            max-width: 100%;\n            max-height: 100%;\n"
        "            width: auto;\n            height: auto;\n"
        "            display: block;\n            cursor: crosshair;\n"
        "            object-fit: contain;\n            transform-origin: center;\n"
        "            transition: transform 0.1s;\n        }",
        "#videoCanvas {\n            max-width: 100%;\n            max-height: 100%;\n"
        "            display: block;\n            cursor: grab;\n"
        "            object-fit: contain;\n            transform-origin: center;\n"
        "            touch-action: none;\n        }",
        "canvas-css",
    )

    html = _must_replace(
        html,
        """        canvas.setAttribute('tabindex', '0');
        canvas.addEventListener('click', () => {
            canvas.focus();
        });""",
        """        canvas.setAttribute('tabindex', '0');
        canvas.addEventListener('click', () => {
            canvas.focus();
        });

        // Forward drag outside canvas; never re-bubble (avoids double-fire / loops)
        document.addEventListener('mousemove', (e) => {
            if (!(isDragging || isResizing || isDrawing || isPanning)) return;
            if (e.target === canvas) return;
            canvas.dispatchEvent(new MouseEvent('mousemove', {
                clientX: e.clientX, clientY: e.clientY, bubbles: false, buttons: e.buttons
            }));
        });
        document.addEventListener('mouseup', (e) => {
            if (!(isDragging || isResizing || isDrawing || isPanning)) return;
            if (e.target === canvas) return;
            canvas.dispatchEvent(new MouseEvent('mouseup', {
                clientX: e.clientX, clientY: e.clientY, bubbles: false, button: e.button
            }));
        });""",
        "doc-mouse",
    )

    html = _must_replace(
        html,
        '<div class="status-item">Events: <span class="status-value" id="statEvents">0</span></div>',
        '<div class="status-item">Events: <span class="status-value" id="statEvents">0</span></div>'
        '<div class="status-item">Mode: <span class="status-value" id="statMode">pan</span></div>',
        "mode-stat",
    )
    html = _must_replace(
        html,
        """                    if (newBoxMode) {
                        debugLog('New box mode: ON - Click and drag to create box');
                        canvas.style.cursor = 'crosshair';
                        selectedBox = null;
                        updateBoxesList();
                    } else {
                        debugLog('New box mode: OFF');
                        canvas.style.cursor = 'grab'; // Default to grab for panning
                        isDrawing = false;
                        drawStart = null;
                        drawFrame();
                    }""",
        """                    const modeEl = document.getElementById('statMode');
                    if (newBoxMode) {
                        debugLog('New box mode: ON - Click and drag to create box');
                        canvas.style.cursor = 'crosshair';
                        selectedBox = null;
                        updateBoxesList();
                        if (modeEl) modeEl.textContent = 'draw (N)';
                    } else {
                        debugLog('New box mode: OFF');
                        canvas.style.cursor = 'grab';
                        isDrawing = false;
                        drawStart = null;
                        drawFrame();
                        if (modeEl) modeEl.textContent = 'pan';
                    }""",
        "mode-toggle",
    )

    html = html.replace(
        """                case ' ':
                    e.preventDefault();
                    if (video.readyState >= 2) {
                        if (video.paused) {
                            video.play().catch(e => console.log('Play failed:', e));
                            document.getElementById('playOverlay').style.display = 'none';
                        } else {
                            video.pause();
                            /* playOverlay disabled */
                        }
                    }
                    break;""",
        """                case ' ':
                    e.preventDefault();
                    break;""",
    )

    for needle in (
        "canvas.addEventListener('mousedown'",
        "function seekToFrame",
        "loadReviewFrame(currentFrame)",
        "isDragging",
        "saveAnnotations",
    ):
        if needle not in html:
            raise RuntimeError(f"Generated editor missing required code: {needle}")

    out_path = gold_dir / "review" / "editor.html"
    out_path.write_text(html, encoding="utf-8")
    (ROOT / "annotation" / "gold100_editor.html").write_text(html, encoding="utf-8")
    return out_path


def validate_pack(gold_dir: Path, max_mean_delta: float = 8.0) -> None:
    manifest = json.loads((gold_dir / "manifest.json").read_text())
    strip_path = gold_dir / "review" / "strip_100.mp4"
    cap = cv2.VideoCapture(str(strip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open strip: {strip_path}")

    failures = []
    for row in manifest["frames"]:
        idx = int(row["strip_frame"])
        src = cv2.imread(str(gold_dir / "images" / row["image"]))
        review = cv2.imread(str(gold_dir / "review" / "frames" / f"{idx:03d}.jpg"))
        if src is None or review is None:
            failures.append(f"missing assets frame {idx}")
            continue
        expect = resize_for_strip(src)
        if expect.shape != review.shape:
            failures.append(f"frame {idx}: shape {review.shape} != {expect.shape}")
            continue
        delta = float(np.mean(cv2.absdiff(expect, review)))
        if delta > max_mean_delta:
            failures.append(f"frame {idx}: review vs source delta={delta:.2f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, strip_frame = cap.read()
        if not ok or strip_frame is None:
            failures.append(f"frame {idx}: strip read failed")
            continue
        if strip_frame.shape != review.shape:
            failures.append(f"frame {idx}: strip shape mismatch")
            continue
        strip_delta = float(np.mean(cv2.absdiff(strip_frame, review)))
        if strip_delta > 25.0:
            failures.append(f"frame {idx}: strip vs review delta={strip_delta:.2f}")

    cap.release()

    import xml.etree.ElementTree as ET
    tree = ET.parse(gold_dir / "prelabels" / "annotations.xml")
    sample = cv2.imread(str(gold_dir / "review" / "frames" / "000.jpg"))
    h, w = sample.shape[:2]
    oob = 0
    for box in tree.findall(".//box"):
        xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
        xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
        if xtl < -1 or ytl < -1 or xbr > w + 1 or ybr > h + 1:
            oob += 1
    if oob:
        failures.append(f"{oob} XML boxes out of strip bounds {w}x{h}")

    if failures:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(failures[:20]))
    print(f"OK: validated {len(manifest['frames'])} frames (image/review/strip + XML bounds)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--skip-validate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gold_dir = args.gold_dir.resolve()
    print(f"Hardening {gold_dir}")
    names, width, height = write_review_frames(gold_dir)
    print(f"Wrote {len(names)} review frames @ {width}x{height}")
    n_clamp = clamp_xml_boxes(gold_dir, width, height)
    print(f"Clamped {n_clamp} XML boxes to {width}x{height}")
    write_strip_from_frames(gold_dir, len(names))
    print("Wrote all-intra strip from review frames")
    editor = build_image_editor(gold_dir, names, width, height)
    print(f"Wrote image-sequence editor: {editor}")
    manifest_path = gold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["strip_size"] = [width, height]
    manifest["review_mode"] = "image_sequence"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not args.skip_validate:
        validate_pack(gold_dir)
    print("Done. Hard-refresh http://localhost:8080/gold100")


if __name__ == "__main__":
    main()
