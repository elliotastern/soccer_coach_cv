#!/usr/bin/env python3
"""
Serve the 2D map marking UI only (port 5006).
Use this when you get "Unable to connect" or after the main script exited.

1. Generate the mark UI once: python scripts/test_2dmap_manual_mark.py --mark --web
   (If it exits before you mark, or you closed the terminal, continue to step 2.)
2. Run this server: python scripts/serve_2dmap_mark_ui.py
3. In Cursor: View -> Ports -> Forward a Port -> 5006
4. Open http://localhost:5006/mark_ui.html (or the URL shown next to port 5006)
5. Mark the 6 spots and click Save. Then run: python scripts/test_2dmap_manual_mark.py --use-saved
"""
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "output" / "2dmap_manual_mark"
MARKS_PATH = OUT_DIR / "manual_marks.json"
PORT = 5006


class MarkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/mark_ui.html"):
            ui_path = OUT_DIR / "mark_ui.html"
            if not ui_path.exists():
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Mark UI not generated. Run: python scripts/test_2dmap_manual_mark.py --mark --web\n")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(ui_path.read_bytes())
        elif self.path == "/mark_frame.jpg":
            frame_path = OUT_DIR / "mark_frame.jpg"
            if not frame_path.exists():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(frame_path.read_bytes())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save_marks":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(MARKS_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    if not (OUT_DIR / "mark_ui.html").exists():
        print("Mark UI not found. Run once: python scripts/test_2dmap_manual_mark.py --mark --web")
        print("(You can Ctrl+C after it prints the URL; this script will then serve that UI.)")
        return
    server = HTTPServer(("0.0.0.0", PORT), MarkHandler)
    print("")
    print("2D map marking UI server: http://localhost:{}".format(PORT))
    print("Open: http://localhost:{}/mark_ui.html".format(PORT))
    print("")
    print("If you see 'Unable to connect': View -> Ports -> Forward a Port -> {}".format(PORT))
    print("Then open the URL that appears next to port {} in the Ports list.".format(PORT))
    print("")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
