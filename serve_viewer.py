#!/usr/bin/env python3
"""
Simple HTTP server to view annotations
Run this and open http://localhost:8080/view_annotations.html in your browser (default port 8080)
"""
import argparse
import http.server
import socketserver
import os
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

PORT = 8080
FALLBACK_PORTS = (8081, 8082, 9000, 3000, 8765, 8877)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Concurrent GETs — single-thread TCPServer deadlocks under browser load."""

    allow_reuse_address = True
    daemon_threads = True

SCRIPT_DIR = Path(__file__).resolve().parent
MARK_UI_RELPATH = 'data/output/2dmap_manual_mark/mark_ui.html'
PITCH_DIAGRAM_RELPATH = 'data/output/2dmap_manual_mark/pitch_diagram_reference.html'
MARK_UI_HELP_HTML = (
    b'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Mark UI - generate first</title></head>'
    b'<body style="font-family:sans-serif;margin:2em;max-width:40em;">'
    b'<h1>Mark UI not generated yet</h1>'
    b'<p>Generate the marking UI first (from the project root):</p>'
    b'<pre style="background:#eee;padding:1em;">python scripts/test_2dmap_manual_mark.py --mark --web</pre>'
    b'<p>You can Ctrl+C after it prints the URL. Then refresh this page.</p>'
    b'<p>Expected file: <code>data/output/2dmap_manual_mark/mark_ui.html</code></p>'
    b'</body></html>'
)
PITCH_DIAGRAM_HELP_HTML = (
    b'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Pitch diagram - generate first</title></head>'
    b'<body style="font-family:sans-serif;margin:2em;max-width:40em;">'
    b'<h1>Pitch diagram not generated yet</h1>'
    b'<p>From the project root run:</p>'
    b'<pre style="background:#eee;padding:1em;">python scripts/test_pitch_diagram.py</pre>'
    b'<p>Then refresh this page.</p>'
    b'<p>Expected file: <code>data/output/2dmap_manual_mark/pitch_diagram_reference.html</code></p>'
    b'</body></html>'
)

def _port_is_free(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def _parse_byte_range(range_header: str, file_size: int):
    """Return (start, end_inclusive) or None if unsatisfiable / ignored."""
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[6:].strip()
    if "," in spec:
        spec = spec.split(",", 1)[0].strip()
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    try:
        if start_s == "":
            # suffix: bytes=-N
            length = int(end_s)
            if length <= 0:
                return None
            start = max(0, file_size - length)
            end = file_size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start < 0 or start >= file_size:
        return None
    end = min(end, file_size - 1)
    if end < start:
        return None
    return start, end


def _first_available_port(host, preferred, fallbacks=FALLBACK_PORTS):
    if _port_is_free(host, preferred):
        return preferred
    for p in fallbacks:
        if _port_is_free(host, p):
            return p
    return None


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow loading local files
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _serve_file_ranges(self, file_path: Path):
        """Serve local file with HTTP Range (needed for <video> scrubbing)."""
        try:
            file_size = file_path.stat().st_size
        except OSError:
            self.send_error(404, "File not found")
            return
        ctype = self.guess_type(str(file_path))
        range_hdr = self.headers.get("Range")
        byte_range = _parse_byte_range(range_hdr, file_size) if range_hdr else None
        try:
            with open(file_path, "rb") as f:
                if byte_range is None:
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    if self.command == "HEAD":
                        return
                    self.copyfile(f, self.wfile)
                    return
                start, end = byte_range
                length = end - start + 1
                f.seek(start)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if self.command == "HEAD":
                    return
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError:
            self.send_error(404, "File not found")

    def do_HEAD(self):
        # Same short routes as GET so curl -I / healthchecks work
        self.do_GET(head_only=True)

    def do_GET(self, head_only=False):
        if head_only:
            # reuse redirect + file logic; handlers check self.command
            pass
        # Short URL for gold100 correction editor
        if self.path in ('/gold100', '/gold100/', '/gold100_editor.html'):
            self.send_response(302)
            self.send_header('Location', '/data/processed/gold_sets/match1_1_100/review/editor.html')
            self.end_headers()
            return
        # Match 2 gold 100 frames in a row confidence scrubber
        if self.path in ('/match2-100row', '/match2-100row/', '/match2_100row'):
            self.send_response(302)
            self.send_header('Location', '/annotation/match2_100row_viewer.html')
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in ('/5x5', '/5x5/', '/5x5-clips'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/5x5_clips_bestcam/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in ('/4quad', '/4quad/', '/4-quad', '/quad'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/4quad_test/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/4quad-cvat',
            '/4quad-cvat/',
            '/4quad-label',
            '/4quad_label',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_label/review/editor.html',
            )
            self.end_headers()
            return
        # One quad at a time (not all 4 combined)
        quad_one = self.path.split('?', 1)[0].rstrip('/')
        _quad_ranges = {
            '/4quad-cvat/center': (0, 24),
            '/4quad-cvat/center_start': (0, 24),
            '/4quad-cvat/bottom': (25, 54),
            '/4quad-cvat/bottom_right': (25, 54),
            '/4quad-cvat/top_left': (55, 79),
            '/4quad-cvat/top-left': (55, 79),
            '/4quad-cvat/top_right': (80, 104),
            '/4quad-cvat/top-right': (80, 104),
        }
        if quad_one in (
            '/4quad-cvat/top_left',
            '/4quad-cvat/top-left',
        ):
            # Standalone pack: only Top Left frames (0..N), nothing before.
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_top_left/review/editor.html',
            )
            self.end_headers()
            return
        if quad_one in (
            '/4quad-cvat/top_left_p7',
            '/4quad-cvat/top-left-p7',
            '/4quad-cvat/top_left_P7',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_top_left_p7/review/editor.html',
            )
            self.end_headers()
            return
        if quad_one in (
            '/4quad-cvat/center_start_cam4plus',
            '/4quad-cvat/center-start-cam4plus',
            '/4quad-cvat/center_start_Cam4plus',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_center_start_cam4plus/review/editor.html',
            )
            self.end_headers()
            return
        if quad_one in (
            '/4quad-cvat/top_left_cam4plus',
            '/4quad-cvat/top-left-cam4plus',
            '/4quad-cvat/top_left_Cam4plus',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_top_left_cam4plus/review/editor.html',
            )
            self.end_headers()
            return
        if quad_one in _quad_ranges:
            start, end = _quad_ranges[quad_one]
            frames = ','.join(str(i) for i in range(start, end + 1))
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_4quad_label/review/editor.html'
                f'?frames={frames}&frame={start}',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/ball_postprocessing_test',
            '/ball_postprocessing_test/',
            '/ball-postprocessing-test',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/ball_postprocessing_test/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/multicam-proxy',
            '/multicam_proxy',
            '/top-left-proxy',
            '/proxy095',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/top_left_multicam_baseline/proxy_gallery/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match2-aim',
            '/match2-aim/',
            '/aim-zoom',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/aim_zoom_guide/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/camera-coverage',
            '/camera_coverage',
            '/pitch-coverage',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/camera_pitch_coverage/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-pitchmap',
            '/match3-pitchmap/',
            '/match3_pitchmap',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/pitchmap_gallery/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-pitchmap-v11',
            '/match3-pitchmap-v11/',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/pitchmap_gallery_v11/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-pitchmap-v12',
            '/match3-pitchmap-v12/',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/pitchmap_gallery_v12_hard/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-pitchmap-defish',
            '/match3-pitchmap-defish/',
            '/match3_pitchmap_defish',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/pitchmap_gallery_defish/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-pitchmap-nodefish',
            '/match3-pitchmap-nodefish/',
            '/match3_pitchmap_nodefish',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/pitchmap_gallery_nodefish/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-quad-v12',
            '/match3-quad-v12/',
            '/match3-dashboard-v12',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/quad_pitchmap_gallery_v12_hard/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-m1',
            '/match3-m1/',
            '/match3_m1',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match3_m1_hub.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-m1-p10',
            '/match3-m1-p10/',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match3_quad_p10_31/review/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-m1-p8',
            '/match3-m1-p8/',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match3_quad_p8_87/review/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match3-fisheye',
            '/match3-fisheye/',
            '/match3_fisheye',
            '/fisheye_dashboard',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/fisheye_dashboard/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] == '/match3_fisheye_preview':
            try:
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(self.path).query)
                cam = (qs.get('cam') or ['P1'])[0]
                k1 = float((qs.get('k1') or ['-0.2'])[0])
                k2 = float((qs.get('k2') or ['0'])[0])
                p1 = float((qs.get('p1') or ['0'])[0])
                p2 = float((qs.get('p2') or ['0'])[0])
                alpha = float((qs.get('alpha') or ['0.5'])[0])
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import match3_fisheye_dashboard as fish

                fish = importlib.reload(fish)
                jpeg = fish.render_preview_jpeg(cam, k1, k2, p1, p2, alpha)
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode())
            return
        if self.path.split('?', 1)[0] == '/match3_landmark_defish_preview':
            try:
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(self.path).query)
                cam = (qs.get('cam') or ['P8'])[0]
                k1 = float((qs.get('k1') or ['-0.3'])[0])
                k2 = float((qs.get('k2') or ['0'])[0])
                p1 = float((qs.get('p1') or ['0'])[0])
                p2 = float((qs.get('p2') or ['0'])[0])
                alpha = float((qs.get('alpha') or ['0.8'])[0])
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import match3_landmarks as lm

                lm = importlib.reload(lm)
                jpeg = lm.render_defish_raw_jpeg(cam, k1, k2, p1, p2, alpha)
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode())
            return
        if self.path.split('?', 1)[0] in (
            '/phase1-handover',
            '/phase1-handover/',
            '/phase1_handover',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/improve_eng_loop/phase1_handover/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/landmark_marker',
            '/landmark_marker/',
            '/match3-landmarks',
            '/match3-landmarks/',
            '/match3_landmarks',
            '/match3-landmark',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match3/landmark_dashboard/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/match2-pitchmap',
            '/match2-pitchmap/',
            '/pitchmap',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/locked_oos_pitchmap_gallery/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/ball_sahi_hurt_test',
            '/ball_sahi_hurt_test/',
            '/sahi-hurt',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/ball_sahi_hurt_test/index.html',
            )
            self.end_headers()
            return
        if self.path.split('?', 1)[0] in (
            '/ball_sahi_next_test',
            '/ball_sahi_next_test/',
            '/sahi-next',
        ):
            self.send_response(302)
            self.send_header(
                'Location',
                '/reports/eval_match2_v10/ball_sahi_next_test/index.html',
            )
            self.end_headers()
            return
        harvest_path = self.path.split('?', 1)[0]
        if self.path.split('?', 1)[0] in ('/accepted50', '/accepted50/', '/match2-accepted50'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_large_ball_accepted50/review/editor.html',
            )
            self.end_headers()
            return
        if harvest_path in ('/match2-gold', '/match2-gold/', '/match2_gold'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_gold_frames/review/editor.html',
            )
            self.end_headers()
            return
        if harvest_path in ('/match2-train100', '/match2-train100/', '/train100'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/match2_train_label100/review/editor.html',
            )
            self.end_headers()
            return
        if harvest_path in ('/match2-harvest', '/match2-harvest/', '/harvest'):
            qs = ''
            if '?' in self.path:
                qs = '?' + self.path.split('?', 1)[1]
            self.send_response(302)
            self.send_header('Location', '/annotation/match2_harvest_editor.html' + qs)
            self.end_headers()
            return
        if harvest_path in ('/batch3', '/batch3/', '/math1-batch3'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/math_1_training_batch3/review/editor.html',
            )
            self.end_headers()
            return
        if harvest_path in ('/batch2', '/batch2/', '/math1-batch2'):
            self.send_response(302)
            self.send_header(
                'Location',
                '/data/processed/gold_sets/math_1_training_batch2/review/editor.html',
            )
            self.end_headers()
            return
        if self.path in ('/2dmap', '/2dmap/', '/2d-map'):
            self.send_response(302)
            self.send_header('Location', '/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html')
            self.end_headers()
            return
        if self.path in ('/pitch', '/pitch/', '/pitch-diagram'):
            self.send_response(302)
            self.send_header('Location', '/' + PITCH_DIAGRAM_RELPATH)
            self.end_headers()
            return
        if self.path in ('/mark', '/mark/', '/mark-ui'):
            self.send_response(302)
            self.send_header('Location', '/data/output/2dmap_manual_mark/mark_ui.html')
            self.end_headers()
            return

        raw_path = self.path.split('?', 1)[0].lstrip('/')
        file_path = SCRIPT_DIR / raw_path
        if (
            file_path.is_file()
            and (
                self.headers.get("Range")
                or file_path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
            )
        ):
            self._serve_file_ranges(file_path)
            return

        if self.path.endswith('.xml') or raw_path.endswith('.xml'):
            try:
                if file_path.exists():
                    self.send_response(200)
                    self.send_header('Content-type', 'application/xml')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    if self.command != "HEAD":
                        with open(file_path, 'rb') as f:
                            self.copyfile(f, self.wfile)
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        elif self.path.endswith('.html') or raw_path.endswith('.html'):
            # Handle large HTML files by streaming in chunks. Resolve from script dir (project root) per cursorrules.
            try:
                if raw_path == MARK_UI_RELPATH:
                    file_path = SCRIPT_DIR / MARK_UI_RELPATH
                elif raw_path == PITCH_DIAGRAM_RELPATH:
                    file_path = SCRIPT_DIR / PITCH_DIAGRAM_RELPATH
                else:
                    file_path = SCRIPT_DIR / raw_path
                if file_path.exists():
                    file_size = os.path.getsize(file_path)
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    if self.command == "HEAD":
                        return
                    chunk_size = 8192
                    with open(file_path, 'rb') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                elif raw_path == MARK_UI_RELPATH:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(MARK_UI_HELP_HTML)))
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(MARK_UI_HELP_HTML)
                elif raw_path == PITCH_DIAGRAM_RELPATH:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(PITCH_DIAGRAM_HELP_HTML)))
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(PITCH_DIAGRAM_HELP_HTML)
                else:
                    self.send_response(404)
                    self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        else:
            # Use parent class to serve other files normally
            try:
                if self.command == "HEAD":
                    super().do_HEAD()
                else:
                    super().do_GET()
            except (BrokenPipeError, ConnectionResetError):
                # Browser cancelled a large video/image download; keep server alive.
                return

    def do_POST(self):
        if self.path == '/save_match3_fisheye_tags':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import match3_fisheye_dashboard as fish

                fish = importlib.reload(fish)
                tags = fish.apply_tag_updates(data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'tags': tags}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/match3_landmark_defish_tune':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import match3_landmarks as lm

                lm = importlib.reload(lm)
                result = lm.tune_defish(
                    str(data['camera']),
                    float(data.get('k1', -0.3)),
                    float(data.get('alpha', 0.8)),
                    data.get('image_points') or [],
                    data.get('landmarks') or [],
                    apply=bool(data.get('apply')),
                    estimate=bool(data.get('estimate')),
                )
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_match3_landmark':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import match3_landmarks
                match3_landmarks = importlib.reload(match3_landmarks)
                result = match3_landmarks.save_clicks(
                    str(data['camera']),
                    str(data['order']),
                    data.get('image_points') or [],
                    landmark_names=data.get('landmarks'),
                    dry_run=bool(data.get('dry_run')),
                )
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_phase1_handover_labels':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                path = (
                    SCRIPT_DIR
                    / 'reports/eval_match3/improve_eng_loop/phase1_handover/labels.json'
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                data['updated_at'] = datetime.now(timezone.utc).isoformat()
                path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'path': str(path)}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_match3_m1_labels':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                gold = str(SCRIPT_DIR / 'scripts' / 'gold_set')
                if gold not in sys.path:
                    sys.path.insert(0, gold)
                import importlib
                import rematch_match3_m1_gold as rematch
                rematch = importlib.reload(rematch)
                payload = rematch.rematch_labels(data)
                payload['human_reviewed'] = True
                pack_dir = rematch.pack_dir_for(payload)
                path = pack_dir / 'labels.json'
                path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                man = pack_dir / 'manifest.json'
                if man.is_file():
                    m = json.loads(man.read_text(encoding='utf-8'))
                    m['n_clear'] = payload.get('n_clear')
                    m['n_gold_xy'] = payload.get('n_gold_xy')
                    m['human_reviewed'] = True
                    m['provisional'] = False
                    man.write_text(json.dumps(m, indent=2), encoding='utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'ok': True,
                    'pack': payload.get('pack'),
                    'n_clear': payload.get('n_clear'),
                    'n_gold_xy': payload.get('n_gold_xy'),
                    'labels': payload,
                }).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_marks':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'ok': False, 'error': 'No content'}).encode())
                    return
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                marks_path = SCRIPT_DIR / 'data/output/2dmap_manual_mark/manual_marks.json'
                marks_path.parent.mkdir(parents=True, exist_ok=True)
                with open(marks_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"ok": true}')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_harvest_keep':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                pack = data.get('pack', 'data/processed/gold_sets/match2_large_ball_harvest')
                keep_path = SCRIPT_DIR / pack / 'keep.json'
                keep_path.parent.mkdir(parents=True, exist_ok=True)
                keep_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"ok": true}')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
            return
        if self.path == '/save_annotations':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'error': 'No content provided'}).encode())
                    return
                
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                xml_content = data.get('xml')
                file_path = data.get('file_path', 'data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506_annotations.xml')
                
                if not xml_content:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'error': 'No XML content provided'}).encode())
                    return
                
                # Write to file
                full_path = Path(file_path)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'success': True, 'message': f'File saved to {file_path}'})
                self.wfile.write(response.encode())
                
            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'success': False, 'error': f'Invalid JSON: {str(e)}'})
                self.wfile.write(response.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'success': False, 'error': str(e)})
                self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'error': 'Endpoint not found'}).encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    parser = argparse.ArgumentParser(description="HTTP server for annotation/viewer HTML")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help=f"Port to listen on (default: {PORT}, with fallbacks if busy)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail if the requested port is busy (used by run_viewer_stable.sh)",
    )
    args = parser.parse_args()
    explicit = args.port is not None
    port = args.port if explicit else PORT
    if (not explicit) and (not args.no_fallback):
        avail = _first_available_port("127.0.0.1", PORT)
        if avail is None:
            print(f"Port {PORT} and fallbacks {FALLBACK_PORTS} are in use. Use --port N to try another.")
            return
        if avail != PORT:
            print(f"Port {PORT} in use, using port {avail}")
        port = avail
    elif args.no_fallback or explicit:
        # Don't pre-bind-check (races). ThreadingHTTPServer will raise if busy.
        pass

    os.chdir(Path(__file__).parent)

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), MyHTTPRequestHandler)
    except OSError as e:
        print(f"Could not bind port {port}: {e}")
        print(f"Try: python3 serve_viewer.py --port N")
        return

    # Persist port for launch scripts / open_gold100 helpers
    (Path("/tmp") / "soccer_coach_serve_viewer.port").write_text(str(port))
    (Path("/tmp") / "soccer_coach_serve_viewer.pid").write_text(str(os.getpid()))

    gold_editor = (
        f"http://127.0.0.1:{port}/data/processed/gold_sets/"
        "match1_1_100/review/editor.html"
    )
    print("=" * 60)
    print("Annotation Viewer Server Started (threaded)")
    print("=" * 60)
    print(f"Server:      http://127.0.0.1:{port}")
    print(f"Gold100:     {gold_editor}")
    print(f"Match2 100row: http://127.0.0.1:{port}/match2-100row")
    print(f"5x5 clips:   http://127.0.0.1:{port}/5x5")
    print(f"4 quad test: http://127.0.0.1:{port}/4quad")
    print(f"4 quad CVAT: http://127.0.0.1:{port}/4quad-cvat")
    print(f"Ball postproc: http://127.0.0.1:{port}/ball_postprocessing_test")
    print(f"Multicam proxy gallery: http://127.0.0.1:{port}/multicam-proxy")
    print(f"Camera pitch coverage: http://127.0.0.1:{port}/camera-coverage")
    print(f"Match2 aim/zoom guide: http://127.0.0.1:{port}/match2-aim")
    print(f"Match3 pitchmap: http://127.0.0.1:{port}/match3-pitchmap")
    print(f"Match3 pitchmap defish: http://127.0.0.1:{port}/match3-pitchmap-defish")
    print(f"Match3 M1 hub:   http://127.0.0.1:{port}/match3-m1")
    print(f"  P10 strip:     http://127.0.0.1:{port}/match3-m1-p10")
    print(f"  P8 strip:      http://127.0.0.1:{port}/match3-m1-p8")
    print(f"Match3 fisheye: http://127.0.0.1:{port}/match3-fisheye")
    print(f"Phase 1 handover: http://127.0.0.1:{port}/phase1-handover")
    print(f"landmark_marker: http://127.0.0.1:{port}/landmark_marker")
    print(f"Match2 pitchmap: http://127.0.0.1:{port}/match2-pitchmap")
    print(f"SAHI hurt test: http://127.0.0.1:{port}/ball_sahi_hurt_test")
    print(f"SAHI next test: http://127.0.0.1:{port}/ball_sahi_next_test")
    print(f"Match2 harvest: http://127.0.0.1:{port}/match2-harvest")
    print(f"math_1_train: http://127.0.0.1:{port}/data/processed/gold_sets/math_1_training/review/editor.html")
    print(f"math_1_batch3: http://127.0.0.1:{port}/batch3")
    print(f"math_1_batch2: http://127.0.0.1:{port}/data/processed/gold_sets/math_1_training_batch2/review/editor.html")
    print(f"2D map:      http://127.0.0.1:{port}/2dmap")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        httpd.shutdown()
    finally:
        for p in (
            Path("/tmp/soccer_coach_serve_viewer.port"),
            Path("/tmp/soccer_coach_serve_viewer.pid"),
        ):
            try:
                p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
