#!/usr/bin/env python3
"""
Flask-based HTTP server for viewing annotations and HTML reports.
More stable than SimpleHTTPRequestHandler with better error handling,
compression, and proper HTTP/1.1 support.
Run this and open http://localhost:6851/view_annotations.html in your browser (default port 6851)
"""
import argparse
import os
import json
from pathlib import Path

try:
    import sys
    # Add user site-packages to path if Flask installed there
    import site
    site.addsitedir('/root/.local/lib/python3.11/site-packages')
    
    from flask import Flask, send_from_directory, send_file, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not installed. Install with: pip install --user flask flask-cors")
    print("   Falling back to basic server...")

PROJECT_ROOT = Path(__file__).resolve().parent
PORT = 8080

if FLASK_AVAILABLE:
    app = Flask(__name__, static_folder=None)
    CORS(app)  # Enable CORS for all routes
    
    # Configure Flask for better performance
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache static files for 1 hour
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    @app.route('/save_annotations', methods=['POST', 'OPTIONS'])
    def save_annotations():
        """Save annotations XML file."""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No content provided'}), 400
            
            xml_content = data.get('xml')
            file_path = data.get('file_path', 'data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506_annotations.xml')
            
            if not xml_content:
                return jsonify({'success': False, 'error': 'No XML content provided'}), 400
            
            # Write to file
            full_path = PROJECT_ROOT / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            return jsonify({'success': True, 'message': f'File saved to {file_path}'}), 200
            
        except json.JSONDecodeError as e:
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/<path:filepath>')
    def serve_file(filepath):
        """Serve static files from project root."""
        # Security: prevent directory traversal
        if '..' in filepath or filepath.startswith('/'):
            return 'Forbidden', 403
        
        file_path = PROJECT_ROOT / filepath
        
        # Check if file exists
        if not file_path.exists():
            return 'File not found', 404
        
        # Check if it's a directory
        if file_path.is_dir():
            return 'Directory listing not allowed', 403
        
        # Special handling for XML files (no cache)
        if filepath.endswith('.xml'):
            response = send_file(str(file_path), mimetype='application/xml')
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        # For other files, use Flask's send_file with proper MIME types
        return send_file(str(file_path))
    
    @app.route('/')
    def index():
        """Serve index or redirect to a default page."""
        return '''
        <html>
        <head><title>Annotation Viewer Server</title></head>
        <body>
            <h1>Annotation Viewer Server</h1>
            <p>Server is running. Available files:</p>
            <ul>
                <li><a href="/view_annotations_editor.html">View Annotations Editor</a></li>
                <li><a href="/data/output/37a_20frames/viewer.html">37a Results Viewer</a></li>
                <li><a href="/data/output/37a_20frames/viewer_with_frames.html">37a Frames + Bboxes</a></li>
                <li><a href="/data/output/fisheye_test/test_fisheye.html">Fisheye Test</a></li>
                <li><a href="/data/output/homography_test/test_homography.html">Homography Test</a></li>
                <li><a href="/data/output/landmark_test/test_landmarks.html">Landmark Test</a></li>
                <li><a href="/data/output/2d_map.mp4">2D Map Video</a></li>
            </ul>
        </body>
        </html>
        ''', 200
    
    def main():
        parser = argparse.ArgumentParser(description="Flask HTTP server for annotation/viewer HTML")
        parser.add_argument("--port", "-p", type=int, default=PORT,
                            help=f"Port to listen on (default: {PORT})")
        parser.add_argument("--host", type=str, default="0.0.0.0",
                            help="Host to bind to (default: 0.0.0.0)")
        parser.add_argument("--debug", action="store_true",
                            help="Enable debug mode")
        args = parser.parse_args()
        
        print("=" * 60)
        print("🌐 Flask Annotation Viewer Server")
        print("=" * 60)
        print(f"📍 Server running at: http://localhost:{args.port}")
        print(f"📄 Open in browser: http://localhost:{args.port}/view_annotations_editor.html")
        print(f"📄 37a results: http://localhost:{args.port}/data/output/37a_20frames/viewer.html")
        print(f"📄 37a frames+bboxes: http://localhost:{args.port}/data/output/37a_20frames/viewer_with_frames.html")
        print(f"📄 Fisheye test: http://localhost:{args.port}/data/output/fisheye_test/test_fisheye.html")
        print(f"📄 Homography test: http://localhost:{args.port}/data/output/homography_test/test_homography.html")
        print(f"📄 Landmark test: http://localhost:{args.port}/data/output/landmark_test/test_landmarks.html")
        print(f"📄 2D map video: http://localhost:{args.port}/data/output/2d_map.mp4")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        print()
        
        # Run Flask development server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,  # Handle multiple requests
            use_reloader=False  # Disable reloader for stability
        )

else:
    # Fallback if Flask is not available
    def main():
        print("❌ Flask is not installed.")
        print("   Install with: pip install flask flask-cors")
        print("   Or use the original serve_viewer.py instead")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
