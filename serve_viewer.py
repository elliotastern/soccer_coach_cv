#!/usr/bin/env python3
"""
Simple HTTP server to view annotations
Run this and open http://localhost:8000/view_annotations.html in your browser
"""
import http.server
import socketserver
import os
import json
from pathlib import Path

PORT = 8000

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
        else:
            # Use parent class to serve other files normally
            super().do_GET()

    def do_POST(self):
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
    os.chdir(Path(__file__).parent)
    
    # Allow socket reuse to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), MyHTTPRequestHandler)
    print("=" * 60)
    print("🌐 Annotation Viewer Server Started")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:{PORT}")
    print(f"📄 Open in browser: http://localhost:{PORT}/view_annotations_editor.html")
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
