#!/usr/bin/env python3
"""
Simple HTTP server to view annotations
Run this and open http://localhost:6851/view_annotations.html in your browser (default port 6851)
"""
import argparse
import http.server
import socketserver
import os
import json
import socket
from pathlib import Path

PORT = 8080
FALLBACK_PORTS = (8081, 8082, 9000, 3000)

SCRIPT_DIR = Path(__file__).resolve().parent
MARK_UI_RELPATH = 'data/output/2dmap_manual_mark/mark_ui.html'
PITCH_DIAGRAM_RELPATH = 'data/output/2dmap_manual_mark/pitch_diagram_reference.html'
MARK_UI_HELP_HTML = (
    b'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Mark UI – generate first</title></head>'
    b'<body style="font-family:sans-serif;margin:2em;max-width:40em;">'
    b'<h1>Mark UI not generated yet</h1>'
    b'<p>Generate the marking UI first (from the project root):</p>'
    b'<pre style="background:#eee;padding:1em;">python scripts/test_2dmap_manual_mark.py --mark --web</pre>'
    b'<p>You can Ctrl+C after it prints the URL. Then refresh this page.</p>'
    b'<p>Expected file: <code>data/output/2dmap_manual_mark/mark_ui.html</code></p>'
    b'</body></html>'
)
PITCH_DIAGRAM_HELP_HTML = (
    b'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Pitch diagram – generate first</title></head>'
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
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Short URL for 2D map report
        if self.path == '/2dmap' or self.path == '/2dmap/':
            self.send_response(302)
            self.send_header('Location', '/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html')
            self.end_headers()
            return
        # Pitch diagram reference (FIFA terminology)
        if self.path == '/pitch_diagram' or self.path == '/pitch_diagram/':
            self.send_response(302)
            self.send_header('Location', '/' + PITCH_DIAGRAM_RELPATH)
            self.end_headers()
            return
        # Marking UI (4 corners + halfway line) – use port 8080 to avoid "Unable to connect"
        if self.path == '/mark_ui' or self.path == '/mark_ui.html' or self.path == '/mark_ui/':
            self.send_response(302)
            self.send_header('Location', '/data/output/2dmap_manual_mark/mark_ui.html')
            self.end_headers()
            return
        # Handle GET requests with cache control for XML files
        if self.path.endswith('.xml'):
            try:
                file_path = self.path.lstrip('/')
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/xml')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        elif self.path.endswith('.html'):
            # Handle large HTML files by streaming in chunks. Resolve from script dir (project root) per cursorrules.
            try:
                raw_path = self.path.lstrip('/')
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
                    self.wfile.write(MARK_UI_HELP_HTML)
                elif raw_path == PITCH_DIAGRAM_RELPATH:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(PITCH_DIAGRAM_HELP_HTML)))
                    self.end_headers()
                    self.wfile.write(PITCH_DIAGRAM_HELP_HTML)
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        else:
            # Use parent class to serve other files normally
            super().do_GET()

    def do_POST(self):
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
    parser.add_argument("--port", "-p", type=int, default=PORT,
                        help=f"Port to listen on (default: {PORT})")
    args = parser.parse_args()
    port = args.port
    if port == PORT:
        avail = _first_available_port("", PORT)
        if avail is None:
            print(f"Port {PORT} and fallbacks {FALLBACK_PORTS} are in use. Use --port N to try another.")
            return
        if avail != PORT:
            print(f"Port {PORT} in use, using port {avail}")
        port = avail
    else:
        if not _port_is_free("", port):
            print(f"Port {port} is in use. Choose another with --port N.")
            return

    os.chdir(Path(__file__).parent)

    # Allow socket reuse to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), MyHTTPRequestHandler)
    print("=" * 60)
    print("🌐 Annotation Viewer Server Started")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:{port}")
    print(f"📄 2D map (short): http://localhost:{port}/2dmap")
    print(f"📄 Mark UI (4 corners + halfway): http://localhost:{port}/mark_ui")
    print(f"📄 2D map (long): http://localhost:{port}/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html")
    print(f"📄 Open in browser: http://localhost:{port}/view_annotations_editor.html")
    print(f"📄 37a results: http://localhost:{port}/data/output/37a_20frames/viewer.html")
    print(f"📄 37a frames+bboxes: http://localhost:{port}/data/output/37a_20frames/viewer_with_frames.html")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
