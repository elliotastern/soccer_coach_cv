"""Playwright eng-loop: score landmark diagram on 5 clarity subgoals (need 9+/10 each)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/eval_match3/landmark_dashboard/eng_loop"
URL = "http://127.0.0.1:8080/reports/eval_match3/landmark_dashboard/index.html?v=clarity-score-v1"
CAMS = ["P10", "P7", "P9", "P8", "P1", "P6", "P_Goal1", "P_Goal2"]
SVG_W = 560.0
PASS = 9.0

SNAPSHOT = """
() => {
  const svg = document.getElementById('names');
  const pitch = svg.querySelector('rect');
  const plot = document.querySelector('.map-plot').getBoundingClientRect();
  const pb = pitch.getBoundingClientRect();
  const sr = svg.getBoundingClientRect();
  const pad = 6;
  const dots = [...svg.querySelectorAll('circle[data-slot]')].map(c => {
    const b = c.getBoundingClientRect();
    return {
      slot: +c.dataset.slot,
      name: c.dataset.name,
      active: c.dataset.active === '1',
      r: +c.getAttribute('r'),
      cx: +c.getAttribute('cx'),
      cy: +c.getAttribute('cy'),
      onPitch: b.right >= pb.left - 4 && b.left <= pb.right + 4 &&
               b.bottom >= pb.top - 4 && b.top <= pb.bottom + 4,
    };
  });
  const tags = [...document.querySelectorAll('#nameLabels .name-tag[data-slot]')].map(t => {
    const b = t.getBoundingClientRect();
    const lab = {
      lx: b.left + b.width / 2 - sr.left,
      ly: b.top + b.height / 2 - sr.top,
      tw: b.width, th: b.height,
    };
    return {
      slot: +t.dataset.slot,
      name: t.dataset.name,
      text: t.textContent.trim(),
      active: t.dataset.active === '1' || t.classList.contains('active-tag'),
      fs: parseFloat(getComputedStyle(t).fontSize),
      h: b.height,
      onPlot: b.right > plot.left && b.left < plot.right &&
              b.bottom > plot.top && b.top < plot.bottom,
      coversLine: hitsPitchLinesPx(svg, lab, 6),
      box: {l:b.left, r:b.right, t:b.top, b:b.bottom},
    };
  });
  const cams = [...svg.querySelectorAll('[data-cam]')].map(g => {
    const c = g.querySelector('circle');
    const t = g.querySelector('text');
    const b = c.getBoundingClientRect();
    return {
      id: g.dataset.cam,
      r: +c.getAttribute('r'),
      fs: +t.getAttribute('font-size'),
      screenR: b.width / 2,
      on: c.getAttribute('fill') === '#e8c547',
      outsidePitch: b.right < pb.left - 2 || b.left > pb.right + 4 ||
                   b.bottom < pb.top - 2 || b.top > pb.bottom + 4,
      box: {l:b.left, r:b.right, t:b.top, b:b.bottom},
    };
  });
  const expected = liveOrder.map((lm, i) => {
    const [sx, sy] = m2svg(lm.xy[0], lm.xy[1]);
    return { slot: i, name: lm.name, title: prettyName(lm), sx, sy };
  });
  function hit(a, b) {
    return a.l - pad < b.r + pad && a.r + pad > b.l - pad &&
           a.t - pad < b.b + pad && a.b + pad > b.t - pad;
  }
  const popups = [
    ...tags.map(t => ({t: t.text, box: t.box})),
    ...cams.map(c => ({t: c.id, box: c.box})),
  ];
  const overlaps = [];
  for (let i = 0; i < popups.length; i++) {
    for (let j = i + 1; j < popups.length; j++) {
      if (hit(popups[i].box, popups[j].box)) {
        overlaps.push(popups[i].t + ' / ' + popups[j].t);
      }
    }
  }
  return {
    expected,
    wantCams: (data.cams || []).map(c => c.id),
    dots, tags, cams, overlaps,
    compass: {
      n: (document.querySelector('[data-compass="north"]') || {}).textContent || '',
      s: (document.querySelector('[data-compass="south"]') || {}).textContent || '',
      w: (document.querySelector('[data-compass="left"]') || {}).textContent || '',
      e: (document.querySelector('[data-compass="right"]') || {}).textContent || '',
    },
    orient: (document.getElementById('mapOrient') || {}).textContent || '',
    legendKeys: [...document.querySelectorAll('#mapKey [data-key]')].map(el => el.dataset.key),
    look: !!document.querySelector('#names [data-look="north"]'),
    lookLabel: !!document.querySelector('#names [data-look-label]'),
    taskWhat: (document.getElementById('taskWhat') || {}).textContent || '',
    taskFind: (document.getElementById('taskFind') || {}).textContent || '',
    taskN: (document.getElementById('taskN') || {}).textContent || '',
  };
}
"""


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def side_of(name: str) -> str:
    if "Left" in name:
        return "left"
    if "Right" in name:
        return "right"
    return "mid"


def score_orientation(info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    c = info["compass"]
    if "NORTH" not in c["n"] or "P6" not in c["n"]:
        score -= 3
        notes.append("north compass weak")
    if "SOUTH" not in c["s"] or "P1" not in c["s"]:
        score -= 3
        notes.append("south compass weak")
    if "LEFT" not in c["w"]:
        score -= 2
        notes.append("left compass missing")
    if "RIGHT" not in c["e"]:
        score -= 2
        notes.append("right compass missing")
    orient = info["orient"].lower()
    if "looking north" not in orient or "p1" not in orient:
        score -= 2
        notes.append("orient banner missing looking-north/P1")
    if not info["look"] or not info["lookLabel"]:
        score -= 1.5
        notes.append("on-pitch looking-north cue missing")
    if sorted(info["legendKeys"]) != ["cam", "grey", "yellow"]:
        score -= 1
        notes.append(f"legend keys {info['legendKeys']}")
    n_fs = 0  # measured via text presence only
    if "LEFT" in c["w"] and "P10" not in c["w"] and "P9" not in c["w"]:
        score -= 0.5
        notes.append("left side lacks cam cue")
    if "RIGHT" in c["e"] and "P7" not in c["e"] and "P8" not in c["e"]:
        score -= 0.5
        notes.append("right side lacks cam cue")
    return clamp(score), notes


def score_cameras(cam: str, info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    want = info.get("wantCams") or CAMS
    have = {c["id"]: c for c in info["cams"]}
    for cid in want:
        m = have.get(cid)
        if not m:
            score -= 1.5
            notes.append(f"missing {cid}")
            continue
        if not m["outsidePitch"]:
            score -= 1
            notes.append(f"{cid} on pitch")
        if m["r"] < 42 or m["fs"] < 26:
            score -= 1
            notes.append(f"{cid} small r/fs")
        if m["screenR"] < 18:
            score -= 1
            notes.append(f"{cid} screenR={m['screenR']:.1f}")
    active = [c["id"] for c in have.values() if c.get("on")]
    if active != [cam]:
        score -= 2
        notes.append(f"highlight {active} want [{cam}]")
    return clamp(score), notes


def score_labels(info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    if len(info["tags"]) != 4:
        score -= 3
        notes.append(f"{len(info['tags'])} tags")
    for t in info["tags"]:
        if not t["onPlot"]:
            score -= 1.5
            notes.append(f"clipped {t['text']}")
        if t["fs"] < 18 or t["h"] < 26:
            score -= 1
            notes.append(f"small {t['text']}")
        if t["coversLine"]:
            score -= 2
            notes.append(f"covers line {t['text']}")
    if info["overlaps"]:
        score -= min(5, 1.5 * len(info["overlaps"]))
        notes.append("overlap " + "; ".join(info["overlaps"][:3]))
    return clamp(score), notes


def score_targets(cam: str, info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    if len(info["dots"]) != 4 or len(info["expected"]) != 4:
        return 3.0, ["need 4 dots/targets"]
    want0 = info["expected"][0]["title"]
    if want0 not in info["taskWhat"]:
        score -= 3
        notes.append(f"task '{info['taskWhat']}' missing {want0}")
    if len(info["taskFind"]) < 20:
        score -= 2
        notes.append("find hint too short")
    if cam not in info["taskN"]:
        score -= 1
        notes.append("task header missing cam id")
    act_dots = [d for d in info["dots"] if d["active"]]
    act_tags = [t for t in info["tags"] if t["active"]]
    if len(act_dots) != 1:
        score -= 2
        notes.append(f"active dots {len(act_dots)}")
    elif act_dots[0]["r"] < 20:
        score -= 1
        notes.append("active dot not enlarged")
    if len(act_tags) != 1:
        score -= 1.5
        notes.append(f"active tags {len(act_tags)}")
    elif want0 not in act_tags[0]["text"]:
        score -= 1.5
        notes.append("active tag != task")
    for exp in info["expected"]:
        dot = next((d for d in info["dots"] if d["slot"] == exp["slot"]), None)
        if not dot or not dot["onPitch"]:
            score -= 1
            notes.append(f"slot {exp['slot']+1} off pitch")
        elif math.hypot(dot["cx"] - exp["sx"], dot["cy"] - exp["sy"]) > 2:
            score -= 0.5
            notes.append(f"slot {exp['slot']+1} misplaced")
    return clamp(score), notes


def score_spatial(info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    left_x, right_x = [], []
    for exp in info["expected"]:
        dot = next((d for d in info["dots"] if d["slot"] == exp["slot"]), None)
        if not dot:
            score -= 1
            continue
        side = side_of(exp["title"])
        if side == "left":
            if dot["cx"] >= SVG_W / 2 - 8:
                score -= 1.5
                notes.append(f"{exp['title']} not left")
            left_x.append(dot["cx"])
        elif side == "right":
            if dot["cx"] <= SVG_W / 2 + 8:
                score -= 1.5
                notes.append(f"{exp['title']} not right")
            right_x.append(dot["cx"])
    if not left_x or not right_x:
        score -= 2
        notes.append("not both sidelines")
    elif max(left_x) >= min(right_x):
        score -= 2
        notes.append("left/right not split")
    named = [d["name"] for d in info["dots"]] + [
        c.get("name", "") for c in info["dots"]
    ]
    # goal boxes from snapshot via any left_box/right_box in page extras — re-check via expected titles
    has_south = any("South" in e["title"] for e in info["expected"]) or any(
        "south" in (e["name"] if False else "") for e in info["expected"]
    )
    # Prefer presence of north+south box rects via look for dots with box in name from catalog not available;
    # use compass+both ends already elsewhere. Check titles span ends when goal order.
    titles = " ".join(e["title"] for e in info["expected"])
    if "Goal-Line" in titles or "18-Yard" in titles:
        if "North" not in titles and "South" not in titles:
            score -= 1
            notes.append("goal landmarks lack N/S wording")
    return clamp(score), notes


def score_cam(cam: str, info: dict) -> dict:
    scores = {}
    notes = {}
    scores["orientation"], notes["orientation"] = score_orientation(info)
    scores["cameras"], notes["cameras"] = score_cameras(cam, info)
    scores["labels"], notes["labels"] = score_labels(info)
    scores["targets"], notes["targets"] = score_targets(cam, info)
    scores["spatial"], notes["spatial"] = score_spatial(info)
    return {"scores": scores, "notes": notes}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_scores = {k: [] for k in [
        "orientation", "cameras", "labels", "targets", "spatial"
    ]}
    fails = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#names")
        page.wait_for_timeout(350)
        for cam in CAMS:
            page.locator(f'.cams button[data-id="{cam}"]').click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(350)
            page.evaluate(
                """() => {
                  const spec = data.cams.find(c => c.id === cam);
                  document.getElementById('order').value = spec.order;
                  setLiveFromOrder();
                  clicks = [null, null, null, null];
                  active = 0;
                  refresh();
                }"""
            )
            page.wait_for_selector("#nameLabels .name-tag[data-slot]")
            page.wait_for_timeout(200)
            info = page.evaluate(SNAPSHOT)
            report = score_cam(cam, info)
            page.locator(".map-panel").screenshot(path=str(OUT / f"names_{cam}.png"))
            page.locator(".work").screenshot(path=str(OUT / f"mark_{cam}.png"))
            print(cam, report["scores"])
            for k, v in report["scores"].items():
                all_scores[k].append(v)
                if v < PASS:
                    fails.append(f"{cam}.{k}={v} {report['notes'][k]}")
        browser.close()

    mins = {k: min(vs) for k, vs in all_scores.items()}
    means = {k: round(sum(vs) / len(vs), 1) for k, vs in all_scores.items()}
    summary = {"mins": mins, "means": means, "pass": PASS, "fails": fails}
    (OUT / "clarity_scores.json").write_text(json.dumps(summary, indent=2))
    print("MINS", mins)
    print("MEANS", means)
    if fails:
        print("BELOW 9")
        for f in fails:
            print(" ", f)
        return 1
    print("all 5 subgoals >= 9/10 on every camera")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
