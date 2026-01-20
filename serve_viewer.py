#!/usr/bin/env python3
"""
Simple HTTP server to view annotations
Run this and open http://localhost:8000/view_annotations.html in your browser
"""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow loading local files
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print("🌐 Annotation Viewer Server Started")
        print("=" * 60)
        print(f"📍 Server running at: http://localhost:{PORT}")
        print(f"📄 Open in browser: http://localhost:{PORT}/view_annotations.html")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
