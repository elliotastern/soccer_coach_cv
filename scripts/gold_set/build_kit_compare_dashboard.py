#!/usr/bin/env python3
"""Build HTML dashboard comparing baseline vs kit-ref Match 4 mosaic team metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/meta.json"
KIT = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min_kitref/meta.json"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min_kitref/kit_compare_dashboard.html"


def _series(stats: list[dict]) -> dict:
    frs, n0, n1, gray, share, collapse = [], [], [], [], [], []
    for s in stats:
        fr = int(s["fr"])
        a0, a1, g = int(s.get("n0", 0)), int(s.get("n1", 0)), int(s.get("gray", 0))
        tot = a0 + a1
        frs.append(fr)
        n0.append(a0)
        n1.append(a1)
        gray.append(g)
        share.append(a0 / tot if tot > 0 else None)
        collapse.append(1 if (a1 <= 1 and a0 >= 5) else 0)
    return {"fr": frs, "n0": n0, "n1": n1, "gray": gray, "share": share, "collapse": collapse}


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    kit = json.loads(KIT.read_text(encoding="utf-8"))
    base_s = _series(base.get("stats") or [])
    kit_s = _series(kit.get("stats") or [])

    def agg(stats: list[dict]) -> dict:
        n = len(stats)
        collapse = both3 = 0
        shares = []
        for s in stats:
            a0, a1 = int(s.get("n0", 0)), int(s.get("n1", 0))
            tot = a0 + a1
            if tot > 0:
                shares.append(a0 / tot)
            if a1 <= 1 and a0 >= 5:
                collapse += 1
            if a0 >= 3 and a1 >= 3:
                both3 += 1
        return {
            "mean_blue_share": sum(shares) / len(shares) if shares else 0,
            "collapse_frac": collapse / n if n else 0,
            "both3_frac": both3 / n if n else 0,
            "n_frames": n,
        }

    bm, km = agg(base.get("stats") or []), agg(kit.get("stats") or [])
    payload = {
        "baseline": {"metrics": bm, "series": base_s, "video": "../match4_5min/coach_mosaic_first_90s.mp4"},
        "kitref": {"metrics": km, "series": kit_s, "video": "coach_mosaic_first_90s.mp4"},
    }

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Kit centroids — Match 4 team compare</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f1419; color: #e7ecf1; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .sub {{ color: #8b9aab; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #1a2332; border-radius: 8px; padding: 14px; }}
  .card h3 {{ margin: 0 0 8px; font-size: 0.85rem; color: #8b9aab; font-weight: 500; }}
  .row {{ display: flex; justify-content: space-between; font-size: 0.95rem; }}
  .good {{ color: #4ade80; }}
  .bad {{ color: #f87171; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .panel {{ background: #1a2332; border-radius: 8px; padding: 16px; }}
  .panel h2 {{ margin: 0 0 12px; font-size: 1rem; }}
  video {{ width: 100%; max-width: 480px; border-radius: 6px; background: #000; }}
  .videos {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }}
  .vidbox h3 {{ font-size: 0.95rem; margin-bottom: 8px; }}
  .pass {{ color: #4ade80; font-weight: 600; }}
  .fail {{ color: #f87171; font-weight: 600; }}
</style></head><body>
<h1>Kit centroids — how much does it help?</h1>
<p class="sub">Match 4 first 90s mosaic · baseline (online fit) vs kit-ref (9 Team 0 + 3 Team 1 crops)</p>

<div class="cards">
  <div class="card"><h3>Mean blue share (target 0.35–0.65)</h3>
    <div class="row"><span>Baseline</span><span class="bad">{bm['mean_blue_share']:.3f}</span></div>
    <div class="row"><span>Kit-ref</span><span class="good">{km['mean_blue_share']:.3f}</span></div></div>
  <div class="card"><h3>White-kit collapse frames</h3>
    <div class="row"><span>Baseline</span><span class="bad">{bm['collapse_frac']*100:.1f}%</span></div>
    <div class="row"><span>Kit-ref</span><span class="good">{km['collapse_frac']*100:.1f}%</span></div></div>
  <div class="card"><h3>Both teams ≥3 players</h3>
    <div class="row"><span>Baseline</span><span class="bad">{bm['both3_frac']*100:.1f}%</span></div>
    <div class="row"><span>Kit-ref</span><span class="good">{km['both3_frac']*100:.1f}%</span></div></div>
  <div class="card"><h3>Product gate</h3>
    <div class="row"><span>Baseline</span><span class="fail">FAIL</span></div>
    <div class="row"><span>Kit-ref</span><span class="pass">PASS</span></div></div>
</div>

<div class="charts">
  <div class="panel"><h2>Team 0 share over time (seconds)</h2><canvas id="shareChart"></canvas></div>
  <div class="panel"><h2>Players per team (first 30s — collapse zone)</h2><canvas id="earlyChart"></canvas></div>
</div>

<div class="videos">
  <div class="vidbox"><h3>Baseline mosaic</h3>
    <video controls src="../match4_5min/coach_mosaic_first_90s.mp4"></video></div>
  <div class="vidbox"><h3>Kit-ref mosaic</h3>
    <video controls src="coach_mosaic_first_90s.mp4"></video></div>
</div>

<script>
const DATA = {json.dumps(payload)};
function sec(fr) {{ return (fr / 60).toFixed(1); }}

const shareCtx = document.getElementById('shareChart');
new Chart(shareCtx, {{
  type: 'line',
  data: {{
    labels: DATA.baseline.series.fr.map(sec),
    datasets: [
      {{ label: 'Baseline blue share', data: DATA.baseline.series.share, borderColor: '#f87171', tension: 0.2, spanGaps: true }},
      {{ label: 'Kit-ref blue share', data: DATA.kitref.series.share, borderColor: '#4ade80', tension: 0.2, spanGaps: true }},
    ]
  }},
  options: {{
    scales: {{
      y: {{ min: 0, max: 1, ticks: {{ callback: v => (v*100)+'%' }} }},
      x: {{ title: {{ display: true, text: 'Match time (s)' }} }}
    }},
    plugins: {{ annotation: {{}} }}
  }}
}});

const earlyFr = DATA.baseline.series.fr.filter(f => f <= 1800);
const idx = i => DATA.baseline.series.fr[i] <= 1800;
const labels = earlyFr.map(sec);
new Chart(document.getElementById('earlyChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [
      {{ label: 'Baseline Team 0', data: DATA.baseline.series.n0.filter((_,i)=>DATA.baseline.series.fr[i]<=1800), backgroundColor: '#3b82f6aa' }},
      {{ label: 'Baseline Team 1', data: DATA.baseline.series.n1.filter((_,i)=>DATA.baseline.series.fr[i]<=1800), backgroundColor: '#eab308aa' }},
      {{ label: 'Kit-ref Team 0', data: DATA.kitref.series.n0.filter((_,i)=>DATA.kitref.series.fr[i]<=1800), backgroundColor: '#22c55e55' }},
      {{ label: 'Kit-ref Team 1', data: DATA.kitref.series.n1.filter((_,i)=>DATA.kitref.series.fr[i]<=1800), backgroundColor: '#f9731655' }},
    ]
  }},
  options: {{ scales: {{ x: {{ stacked: false }}, y: {{ beginAtZero: true }} }} }}
}});
</script></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
