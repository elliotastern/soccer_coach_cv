#!/usr/bin/env python3
"""
Defish tuning tool — browser version.
Run this, then open http://localhost:8080 in your browser.
Adjust sliders until pitch lines look straight; copy the printed parameters into your pipeline.
"""
import io
import cv2
import numpy as np
from flask import Flask, request, send_file, Response

# CONFIGURATION
VIDEO_PATH = '/workspace/soccer_coach_cv/data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4'
SCALE = 0.5
PORT = 8080

app = Flask(__name__)

# Load one frame and camera matrix at startup
_frame = None
_K = None
_w = None
_h = None


def load_frame():
    global _frame, _K, _w, _h
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, _frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to load video: " + VIDEO_PATH)
    _h, _w = _frame.shape[:2]
    _K = np.array([
        [_w, 0, _w / 2],
        [0, _w, _h / 2],
        [0, 0, 1],
    ], dtype=np.float64)


def undistort_frame(k1, k2, p1, p2):
    dist = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(_K, dist, (_w, _h), 1, (_w, _h))
    out = cv2.undistort(_frame, _K, dist, None, new_K)
    cv2.line(out, (0, _h // 2), (_w, _h // 2), (0, 0, 255), 2)
    cv2.line(out, (int(_w / 2), 0), (_w // 2, _h), (0, 0, 255), 2)
    return out


@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Defish Tuner</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }
    h1 { font-size: 1.2rem; }
    .controls { margin: 16px 0; }
    label { display: inline-block; width: 140px; }
    input[type="range"] { width: 220px; vertical-align: middle; }
    .val { display: inline-block; width: 72px; text-align: right; font-family: monospace; }
    #img { max-width: 100%; border: 1px solid #444; display: block; margin-top: 12px; }
    #params { margin-top: 16px; padding: 12px; background: #2a2a2a; border-radius: 8px; font-family: monospace; font-size: 14px; white-space: pre-wrap; }
    button { margin-top: 8px; padding: 8px 16px; cursor: pointer; background: #0a7; color: #fff; border: none; border-radius: 6px; }
    button:hover { background: #08c; }
  </style>
</head>
<body>
  <h1>Defish Tuner</h1>
  <p>Adjust sliders until pitch lines look straight. Red lines = center reference.</p>
  <div class="controls">
    <div><label>K1 (Radial)</label> <input type="range" id="k1" min="0" max="1000" value="500"> <span class="val" id="v1">0.000</span></div>
    <div><label>K2 (Radial)</label> <input type="range" id="k2" min="0" max="1000" value="500"> <span class="val" id="v2">0.000</span></div>
    <div><label>P1 (Tangential)</label> <input type="range" id="p1" min="0" max="1000" value="500"> <span class="val" id="v3">0.000</span></div>
    <div><label>P2 (Tangential)</label> <input type="range" id="p2" min="0" max="1000" value="500"> <span class="val" id="v4">0.000</span></div>
  </div>
  <button id="copy">Copy parameters to clipboard</button>
  <div id="params"></div>
  <img id="img" alt="Undistorted frame">
  <script>
    var k1=document.getElementById('k1'), k2=document.getElementById('k2'), p1=document.getElementById('p1'), p2=document.getElementById('p2');
    var v1=document.getElementById('v1'), v2=document.getElementById('v2'), v3=document.getElementById('v3'), v4=document.getElementById('v4');
    var img=document.getElementById('img'), params=document.getElementById('params');
    function val(slider){ return ((slider.value|0)-500)/1000; }
    function fmt(n){ return n.toFixed(4); }
    function updateLabels(){
      v1.textContent=fmt(val(k1)); v2.textContent=fmt(val(k2)); v3.textContent=fmt(val(p1)); v4.textContent=fmt(val(p2));
    }
    function refresh(){
      var k1v=val(k1), k2v=val(k2), p1v=val(p1), p2v=val(p2);
      img.src='/frame?k1='+k1v+'&k2='+k2v+'&p1='+p1v+'&p2='+p2v+'&t='+Date.now();
      params.textContent = 'k1: '+k1v+'\\nk2: '+k2v+'\\np1: '+p1v+'\\np2: '+p2v;
    }
    function debounce(f, ms){
      var t; return function(){ clearTimeout(t); t=setTimeout(f, ms); };
    }
    k1.oninput=k2.oninput=p1.oninput=p2.oninput=function(){ updateLabels(); };
    [k1,k2,p1,p2].forEach(function(s){ s.addEventListener('input', debounce(refresh, 80)); });
    updateLabels(); refresh();
    document.getElementById('copy').onclick=function(){
      var k1v=val(k1), k2v=val(k2), p1v=val(p1), p2v=val(p2);
      var text = 'k1: '+k1v+'\\nk2: '+k2v+'\\np1: '+p1v+'\\np2: '+p2v;
      navigator.clipboard.writeText(text).then(function(){ params.textContent = 'Copied to clipboard:\\n'+text; });
    };
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.route("/frame")
def frame():
    k1 = float(request.args.get("k1", 0))
    k2 = float(request.args.get("k2", 0))
    p1 = float(request.args.get("p1", 0))
    p2 = float(request.args.get("p2", 0))
    out = undistort_frame(k1, k2, p1, p2)
    out_small = cv2.resize(out, (0, 0), fx=SCALE, fy=SCALE)
    _, buf = cv2.imencode(".jpg", out_small)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


def main():
    load_frame()
    print("Defish tuner: open http://localhost:%s in your browser" % PORT)
    print("Adjust sliders until pitch lines look straight, then use 'Copy parameters'.")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
