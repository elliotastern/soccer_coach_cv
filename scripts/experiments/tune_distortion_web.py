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
PORT = 40399

app = Flask(__name__)

# Load one frame and camera matrix at startup
_frame = None
_K = None
_w = None
_h = None


def load_frame():
    global _frame, _K, _w, _h
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: " + VIDEO_PATH)
    # Skip first few frames in case they are black
    for _ in range(15):
        ret, _frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, _frame = cap.read()
            break
    cap.release()
    if not ret or _frame is None:
        raise RuntimeError("Failed to read frame from: " + VIDEO_PATH)
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
    body { font-family: system-ui, sans-serif; margin: 4px; background: #1a1a1a; color: #e0e0e0; }
    .top { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
    .top h1 { font-size: 0.85rem; margin: 0; }
    .top .hint { font-size: 0.7rem; color: #888; }
    .top button { padding: 2px 8px; font-size: 0.75rem; cursor: pointer; background: #0a7; color: #fff; border: none; border-radius: 3px; }
    .top button:hover { background: #08c; }
    .controls { display: flex; flex-wrap: wrap; gap: 4px 12px; margin-bottom: 4px; align-items: center; }
    .controls > div { display: flex; align-items: center; gap: 4px; }
    label { font-size: 0.75rem; width: 22px; }
    input[type="range"] { width: 80px; vertical-align: middle; }
    .val { width: 44px; text-align: right; font-family: monospace; font-size: 0.7rem; }
    #params { display: inline; font-size: 0.7rem; color: #8a8; margin-left: 8px; }
    #img { max-width: 100%; height: auto; min-height: 270px; max-height: 75vh; border: 1px solid #444; display: block; }
    #defish_debug { font-size: 0.7rem; color: #f80; margin-top: 6px; font-family: monospace; }
  </style>
</head>
<body>
  <img id="img" src="/frame?k1=0&amp;k2=0&amp;p1=0&amp;p2=0" alt="Undistorted frame">
  <div class="top">
    <h1>Defish Tuner</h1>
    <span class="hint">Sliders until lines straight; red = center. If image is black, open /frame?k1=0&k2=0&p1=0&p2=0 in a new tab.</span>
    <button id="copy">Copy params</button>
    <button id="refresh">Refresh</button>
    <span id="params"></span>
  </div>
  <div class="controls">
    <div><label>K1</label><input type="range" id="k1" min="0" max="1000" value="500"><span class="val" id="v1">0.000</span></div>
    <div><label>K2</label><input type="range" id="k2" min="0" max="1000" value="500"><span class="val" id="v2">0.000</span></div>
    <div><label>P1</label><input type="range" id="p1" min="0" max="1000" value="500"><span class="val" id="v3">0.000</span></div>
    <div><label>P2</label><input type="range" id="p2" min="0" max="1000" value="500"><span class="val" id="v4">0.000</span></div>
  </div>
  <div id="defish_debug">loading...</div>
  <script>document.getElementById("defish_debug").textContent="v11";</script>
  <script src="/tuner.js?v=11"></script>
</body>
</html>
"""
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


TUNER_JS = r"""
(function(){
  var dbEl = document.getElementById("defish_debug");
  function db(m) { if (dbEl) dbEl.textContent = m; }
  db("v11 js loaded");
  var defishImgTimer, k1, k2, p1, p2, v1, v2, v3, v4;
  k1 = document.getElementById("k1"); k2 = document.getElementById("k2");
  p1 = document.getElementById("p1"); p2 = document.getElementById("p2");
  v1 = document.getElementById("v1"); v2 = document.getElementById("v2");
  v3 = document.getElementById("v3"); v4 = document.getElementById("v4");
  if (!k1 || !v1) { db("missing elements"); return; }
  function updateFromSliders() {
    function num(inp) { return ((Number(inp.value) || 0) - 500) / 1000; }
    var a = num(k1), b = num(k2), c = num(p1), d = num(p2);
    v1.textContent = a.toFixed(4); v2.textContent = b.toFixed(4);
    v3.textContent = c.toFixed(4); v4.textContent = d.toFixed(4);
    var p = document.getElementById("params");
    if (p) p.textContent = "k1: " + a + " k2: " + b + " p1: " + c + " p2: " + d;
    clearTimeout(defishImgTimer);
    defishImgTimer = setTimeout(function() {
      var base = window.location.origin || (window.location.protocol + "//" + window.location.host);
      var img = document.getElementById("img");
      if (img) img.src = base + "/frame?k1=" + a + "&k2=" + b + "&p1=" + c + "&p2=" + d + "&t=" + Date.now();
    }, 100);
    db("ok");
  }
  k1.oninput = updateFromSliders; k1.onchange = updateFromSliders;
  k2.oninput = updateFromSliders; k2.onchange = updateFromSliders;
  p1.oninput = updateFromSliders; p1.onchange = updateFromSliders;
  p2.oninput = updateFromSliders; p2.onchange = updateFromSliders;
  updateFromSliders();
  document.getElementById("copy").onclick = function() {
    function n(x) { return ((Number(x.value) || 0) - 500) / 1000; }
    navigator.clipboard.writeText("k1: " + n(k1) + "\nk2: " + n(k2) + "\np1: " + n(p1) + "\np2: " + n(p2))
      .then(function() { var p = document.getElementById("params"); if (p) p.textContent = "Copied."; });
  };
  document.getElementById("refresh").onclick = function() { location.reload(); };
})();
"""


@app.route("/tuner.js")
def tuner_js():
    resp = Response(TUNER_JS.strip(), mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


def _make_error_image(message):
    img = np.zeros((120, 500, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    for i, line in enumerate(message.split("\n")[:4]):
        cv2.putText(img, line[:60], (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.route("/frame")
def frame():
    try:
        k1 = float(request.args.get("k1", 0))
        k2 = float(request.args.get("k2", 0))
        p1 = float(request.args.get("p1", 0))
        p2 = float(request.args.get("p2", 0))
        out = undistort_frame(k1, k2, p1, p2)
        out_small = cv2.resize(out, (0, 0), fx=SCALE, fy=SCALE)
        _, buf = cv2.imencode(".jpg", out_small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp
    except Exception as e:
        err = "Error: " + str(e)
        resp = send_file(io.BytesIO(_make_error_image(err)), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp


def main():
    load_frame()
    print("Defish tuner: open http://localhost:%s in your browser" % PORT)
    print("Adjust sliders until pitch lines look straight, then use 'Copy parameters'.")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
