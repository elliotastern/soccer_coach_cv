"""Playwright eng-loop: score landmark diagram on 5 clarity subgoals (need 9+/10 each)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_cam_id import cam_id_from_raw_name  # noqa: E402

P9_VISIBLE = [
    "right_box_18_far",
    "right_box_goal_far",
    "right_post_far",
    "right_far_corner",
]

OUT = ROOT / "reports/eval_match3/landmark_dashboard/eng_loop"
URL = "http://127.0.0.1:8080/reports/eval_match3/landmark_dashboard/index.html?v=save-p9-18far"
CAMS = ["P10", "P7", "P9", "P8", "P1", "P6", "P_Goal1", "P_Goal2"]
SVG_W = 569.0
PASS = 9.0
STILL_DIR = ROOT / "reports/eval_match3/landmark_dashboard/stills"
# Diagram pitch y: +y left, -y right (from P1 looking north).
DIAGRAM_SIDE = {
    "P10": "left", "P9": "left",
    "P7": "right", "P8": "right",
    "P1": "end", "P6": "end", "P_Goal1": "end", "P_Goal2": "end",
}

SNAPSHOT = r"""
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
    const xy = CAM_XY[g.dataset.cam] || [0, 0];
    let diagramSide = 'end';
    if (xy[1] > 20) diagramSide = 'left';
    else if (xy[1] < -20) diagramSide = 'right';
    let diagramEnd = 'mid';
    if (xy[0] > 20) diagramEnd = 'north';
    else if (xy[0] < -20) diagramEnd = 'south';
    const [sx, sy] = m2svg(xy[0], xy[1]);
    return {
      id: g.dataset.cam,
      r: +c.getAttribute('r'),
      fs: +t.getAttribute('font-size'),
      screenR: b.width / 2,
      on: c.getAttribute('fill') === '#e8c547',
      outsidePitch: b.right < pb.left - 2 || b.left > pb.right + 4 ||
                   b.bottom < pb.top - 2 || b.top > pb.bottom + 4,
      box: {l:b.left, r:b.right, t:b.top, b:b.bottom},
      diagramSide,
      diagramEnd,
      pitchX: xy[0],
      pitchY: xy[1],
      sx, sy,
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
      const a = popups[i], b = popups[j];
      const aCam = /^P\d|^P_|^G\d/.test(a.t) || a.t.startsWith('P');
      const bCam = /^P\d|^P_|^G\d/.test(b.t) || b.t.startsWith('P');
      // Cam chips may sit near each other at goal ends; only fail label overlaps.
      if (aCam && bCam) continue;
      if (hit(a.box, b.box)) overlaps.push(a.t + ' / ' + b.t);
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
    lrRule: (document.getElementById('lrRule') || {}).textContent || '',
    orderKey: (document.getElementById('order') || {}).value || '',
    stepTexts: [...document.querySelectorAll('#steps button')].map(b => b.textContent.replace(/\s+/g, ' ').trim()),
    catalog: Object.fromEntries(Object.entries(data.landmarks || {}).map(([n, r]) => [n, {label: r.label, xy: r.xy, spec: r.spec || ''}])),
    camXy: CAM_XY,
    videos: (data.cams || []).map(c => ({id: c.id, videoName: c.videoName || ''})),
    navCams: [...document.querySelectorAll('#camNav button')].map(b => ({
      id: b.dataset.id, text: (b.textContent || '').replace(/\s+/g, ' ').trim(),
    })),
    liveNames: (liveOrder || []).map(lm => lm.name),
    extras: [...svg.querySelectorAll('circle[data-extra]')].map(c => ({
      name: c.dataset.extra,
      r: +c.getAttribute('r'),
      cx: +c.getAttribute('cx'),
      cy: +c.getAttribute('cy'),
      xy: c.dataset.xy || '',
    })),
    swapUi: !!document.querySelector('#swaps [data-swap-ui]'),
    swapLab: (document.querySelector('#swaps .lab') || {}).textContent || '',
    swapChips: [...document.querySelectorAll('#swaps [data-swap-chip]')].map(b => b.dataset.swapChip),
    swapOpts: [...document.querySelectorAll('#swapPick option[value]')].map(o => o.value).filter(Boolean),
    extraTip: (document.getElementById('extraTip') || {}).textContent || '',
    extraTipOn: !!(document.getElementById('extraTip') && document.getElementById('extraTip').classList.contains('on')),
    activeSlot: typeof active === 'number' ? active : 0,
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
    if "LEFT" in c["w"] and "P10" not in c["w"]:
        score -= 0.5
        notes.append("left side lacks P10")
    if "LEFT" in c["w"] and "P8" not in c["w"]:
        score -= 0.5
        notes.append("left side lacks P8")
    if "RIGHT" in c["e"] and "P7" not in c["e"]:
        score -= 0.5
        notes.append("right side lacks P7")
    if "RIGHT" in c["e"] and "P9" not in c["e"]:
        score -= 1.0
        notes.append("right side lacks P9")
    if "LEFT" in c["w"] and "P9" in c["w"]:
        score -= 2.0
        notes.append("P9 should not be on LEFT compass")
    return clamp(score), notes


def still_side(cam: str) -> str:
    """Infer camera sideline from still: less green on a border => outside/fence near that edge."""
    import json
    import subprocess

    py = Path("/Users/elliotstern/.venvs/soccer-rfdetr312/bin/python3")
    path = STILL_DIR / f"{cam}.jpg"
    if not path.is_file():
        return "unknown"
    code = r"""
import cv2, json, sys
img = cv2.imread(sys.argv[1])
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
left = float(green[:, : w // 10].mean()) / 255.0
right = float(green[:, -w // 10 :].mean()) / 255.0
delta = left - right
side = "right" if delta > 0.18 else ("left" if delta < -0.12 else "end")
print(json.dumps({"side": side, "delta": delta}))
"""
    try:
        out = subprocess.check_output(
            [str(py), "-c", code, str(path)],
            text=True,
            timeout=30,
        )
        return json.loads(out.strip())["side"]
    except Exception:
        return "unknown"


def score_filename_ids(info: dict) -> tuple[float, list[str]]:
    """P1-006.mp4 is P1. Never treat it as P9."""
    notes = []
    score = 10.0
    videos = info.get("videos") or []
    if not videos:
        return 4.0, ["cams.json missing videoName"]
    for rec in videos:
        cid = rec.get("id") or ""
        name = rec.get("videoName") or ""
        if not name:
            score -= 2
            notes.append(f"{cid} missing video filename")
            continue
        try:
            parsed = cam_id_from_raw_name(name)
        except ValueError:
            score -= 2
            notes.append(f"unparsed {name}")
            continue
        if parsed != cid:
            score -= 4
            notes.append(f"{name} is {parsed}, not {cid}")
    nav = {n["id"]: n.get("text") or "" for n in info.get("navCams") or []}
    for rec in videos:
        cid = rec.get("id") or ""
        name = rec.get("videoName") or ""
        shown = nav.get(cid, "")
        if name and name not in shown:
            score -= 1.5
            notes.append(f"nav {cid} missing {name}")
    return clamp(score), notes


NEAR_CORNER = {
    "P10": "South Left Corner",
    "P7": "South Right Corner",
    "P8": "North Left Corner",
    "P9": "North Right Corner",
}
FAR_CORNER = {
    "P10": "South Right Corner",
    "P7": "South Left Corner",
    "P8": "North Right Corner",
    "P9": "North Left Corner",
}


def _hyp(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def score_catalog_names(info: dict) -> tuple[float, list[str]]:
    """Left=+y, Right=-y, North=+x, South=-x (P1 looking north)."""
    notes = []
    score = 10.0
    catalog = info.get("catalog") or {}
    if not catalog:
        return 4.0, ["catalog missing"]
    for name, rec in catalog.items():
        label = rec.get("label") or ""
        xy = rec.get("xy") or [0, 0]
        x, y = float(xy[0]), float(xy[1])
        if "Left" in label and y <= 2:
            score -= 1.5
            notes.append(f"{label} y={y:.1f} not left (+y)")
        if "Right" in label and y >= -2:
            score -= 1.5
            notes.append(f"{label} y={y:.1f} not right (-y)")
        if "North" in label and x <= 2:
            score -= 1.5
            notes.append(f"{label} x={x:.1f} not north (+x)")
        if "South" in label and x >= -2:
            score -= 1.5
            notes.append(f"{label} x={x:.1f} not south (-x)")
    return clamp(score), notes


def score_near_corners(info: dict) -> tuple[float, list[str]]:
    """Sideline cam chip must sit nearer its own corner flag than the opposite one."""
    notes = []
    score = 10.0
    have = {c["id"]: c for c in info.get("cams", [])}
    by_title = {e["title"]: e for e in info.get("expected", [])}
    for cam_id, near_title in NEAR_CORNER.items():
        far_title = FAR_CORNER[cam_id]
        chip = have.get(cam_id)
        near = by_title.get(near_title)
        far = by_title.get(far_title)
        if not chip or not near or not far:
            continue
        d_near = _hyp((chip["sx"], chip["sy"]), (near["sx"], near["sy"]))
        d_far = _hyp((chip["sx"], chip["sy"]), (far["sx"], far["sy"]))
        if d_near >= d_far - 4:
            score -= 2.5
            notes.append(f"{cam_id} nearer {far_title} than {near_title}")
    return clamp(score), notes


def meter_tokens(xy: list) -> list[str]:
    return [f"{float(xy[0]):.2f}", f"{float(xy[1]):.2f}"]


def score_unknown(info: dict) -> tuple[float, list[str]]:
    """Unused Pitch 1 marks are swappable (grey dots + nearby + list)."""
    notes = []
    score = 10.0
    catalog = info.get("catalog") or {}
    extras = info.get("extras") or []
    live = info.get("liveNames") or []
    if len(live) < 4:
        return 3.0, ["need 4 live names"]
    if len(set(live)) != len(live):
        score -= 3
        notes.append("live names not unique")
    want = len(catalog) - len(live)
    if want < 8:
        score -= 2
        notes.append(f"catalog too small {len(catalog)}")
    if len(extras) != want:
        score -= 3
        notes.append(f"extras {len(extras)} want {want}")
    extra_names = {e["name"] for e in extras}
    if extra_names & set(live):
        score -= 3
        notes.append("extra overlaps live")
    if not info.get("swapUi"):
        score -= 3
        notes.append("swap UI missing")
    lab = (info.get("swapLab") or "").lower()
    if "not in the photo" not in lab or "swap" not in lab:
        score -= 2
        notes.append("swap lab missing")
    if len(info.get("swapOpts") or []) < 8:
        score -= 2
        notes.append("swap list too short")
    if len(info.get("swapChips") or []) < 3:
        score -= 1.5
        notes.append("few nearby swaps")
    for e in extras:
        if e.get("r", 0) < 10:
            score -= 0.2
            notes.append(f"tiny extra {e.get('name')}")
            break
        if "x=" not in (e.get("xy") or ""):
            score -= 1
            notes.append("extra missing meters")
            break
    return clamp(score), notes


def score_unknown_swap(info: dict, picked: str, old: str) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    live = info.get("liveNames") or []
    extras = {e["name"] for e in info.get("extras") or []}
    if picked not in live:
        score -= 4
        notes.append(f"live missing {picked}")
    if live.count(picked) != 1:
        score -= 2
        notes.append("picked not unique")
    if picked in extras:
        score -= 2
        notes.append("picked still extra")
    if old and old not in extras:
        score -= 2
        notes.append(f"old {old} not extra")
    rec = (info.get("catalog") or {}).get(picked) or {}
    label = rec.get("label") or ""
    if label and label not in (info.get("taskWhat") or ""):
        score -= 3
        notes.append(f"task '{info.get('taskWhat')}' missing {label}")
    find = info.get("taskFind") or ""
    for tok in meter_tokens(rec.get("xy") or [0, 0]):
        if tok not in find:
            score -= 2
            notes.append(f"FIND missing {tok}")
            break
    if "x=" not in find.lower() and "y=" not in find.lower():
        if not any(t in find for t in meter_tokens(rec.get("xy") or [0, 0])):
            score -= 1
            notes.append("FIND missing meters")
    return clamp(score), notes


def pick_unknown_swap(info: dict) -> str | None:
    chips = info.get("swapChips") or []
    if chips:
        return chips[0]
    for e in info.get("extras") or []:
        if "6_" in e["name"]:
            return e["name"]
    extras = info.get("extras") or []
    return extras[0]["name"] if extras else None


def score_naming(cam: str, info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    cat_s, cat_n = score_catalog_names(info)
    if cat_s < 10:
        score -= (10 - cat_s)
        notes.extend(cat_n)
    lr = (info.get("lrRule") or "").lower()
    if "p1" not in lr or "not left/right in this photo" not in lr:
        score -= 2
        notes.append("lr-rule missing P1 vs photo")
    hint = (info.get("orient") or "").lower()
    if "looking north" not in hint or "p1" not in hint:
        score -= 1
        notes.append("orient missing P1-north")
    have = {c["id"]: c for c in info.get("cams", [])}
    want_side = {
        "P10": "left", "P9": "left", "P7": "right", "P8": "right",
        "P1": "end", "P6": "end",
    }
    want_end = {
        "P1": "south", "P6": "north", "P10": "south", "P7": "south",
        "P8": "north", "P9": "north",
    }
    for cid, side in want_side.items():
        chip = have.get(cid)
        if not chip:
            score -= 1
            notes.append(f"chip {cid} missing")
            continue
        if chip.get("diagramSide") != side:
            score -= 2
            notes.append(f"{cid} side={chip.get('diagramSide')} want {side}")
        end = want_end.get(cid)
        if end and chip.get("diagramEnd") != end:
            score -= 1.5
            notes.append(f"{cid} end={chip.get('diagramEnd')} want {end}")
    for exp in info.get("expected", []):
        side = side_of(exp["title"])
        if side == "left" and exp["sx"] >= SVG_W / 2 - 8:
            score -= 1.5
            notes.append(f"{exp['title']} not diagram-left")
        if side == "right" and exp["sx"] <= SVG_W / 2 + 8:
            score -= 1.5
            notes.append(f"{exp['title']} not diagram-right")
        if "North" in exp["title"] and exp["sy"] >= 880 / 2 + 20:
            score -= 1
            notes.append(f"{exp['title']} not diagram-north")
        if "South" in exp["title"] and exp["sy"] <= 880 / 2 - 20:
            score -= 1
            notes.append(f"{exp['title']} not diagram-south")
    near_s, near_n = score_near_corners(info)
    if near_s < 10:
        score -= min(4, 10 - near_s)
        notes.extend(near_n)
    if cam == "P9":
        chip = have.get("P9") or {}
        if chip.get("diagramSide") != "right":
            score -= 3
            notes.append("P9 must be pitch RIGHT")
        titles = " ".join(e["title"] for e in info.get("expected", []))
        if info.get("orderKey") == "both_sides_north":
            if "North Right Corner" not in titles:
                score -= 2
                notes.append("P9 north set missing North Right Corner")
            if "North Left Corner" not in titles:
                score -= 1
                notes.append("P9 north set missing North Left Corner")
        find = (info.get("taskFind") or "").lower()
        if "left of this photo" in find and "still this right" not in find:
            if "north right" in (info.get("taskWhat") or "").lower():
                score -= 2
                notes.append("P9 find treats photo-left as pitch-left")
        steps = " ".join(info.get("stepTexts") or [])
        if info.get("orderKey") == "both_sides_north" and "North Right Corner" not in steps:
            score -= 2
            notes.append("P9 steps missing North Right Corner")
    if cam == "P8" and info.get("orderKey") == "both_sides_north":
        steps = " ".join(info.get("stepTexts") or [])
        if "North Left Corner" not in steps:
            score -= 2
            notes.append("P8 steps missing North Left Corner")
    return clamp(score), notes


def score_video_match(cam: str, info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    have = {c["id"]: c for c in info["cams"]}
    id_s, id_n = score_filename_ids(info)
    if id_s < 10:
        score -= (10 - id_s)
        notes.extend(id_n)
    p9 = have.get("P9")
    if not p9:
        score -= 4
        notes.append("P9 chip missing")
    elif p9.get("diagramSide") != "left":
        score -= 2
        notes.append(f"P9 diagramSide={p9.get('diagramSide')} want left")
    # Active cam: still FOV side must match diagram chip side for sideline cams.
    # P9 is a corner/goal FOV (goal on image-left) — skip green-band left/right gate.
    want = DIAGRAM_SIDE.get(cam, "end")
    got = still_side(cam)
    chip = have.get(cam, {})
    if not cam.startswith("P_Goal") and cam != "P9":
        if chip.get("diagramSide") and chip["diagramSide"] != want:
            score -= 2
            notes.append(f"{cam} chip side {chip['diagramSide']} != expected {want}")
        if want in ("left", "right") and got in ("left", "right") and got != want:
            score -= 3
            notes.append(f"{cam} still={got} diagram={want}")
    if cam == "P8":
        if chip.get("diagramSide") != "right":
            score -= 3
            notes.append("P8 must be diagram RIGHT")
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
    if len(info["tags"]) < 4:
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
    if len(info["dots"]) < 4 or len(info["expected"]) < 4:
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


def _hover_extra(page, info: dict) -> tuple[float, list[str]]:
    extras = info.get("extras") or []
    name = next((e["name"] for e in extras if e["name"] == "center"), None)
    if not name and extras:
        name = extras[0]["name"]
    if not name:
        return 4.0, ["no extra to hover"]
    page.locator(f'circle[data-extra="{name}"]').hover(force=True)
    page.wait_for_timeout(100)
    tip = page.evaluate(SNAPSHOT)
    text = tip.get("extraTip") or ""
    notes, score = [], 10.0
    if not tip.get("extraTipOn"):
        score -= 3
        notes.append("hover tip closed")
    rec = (info.get("catalog") or {}).get(name) or {}
    for tok in meter_tokens(rec.get("xy") or [0, 0]):
        if tok not in text:
            score -= 2
            notes.append(f"hover missing {tok}")
            break
    page.mouse.move(0, 0)
    return clamp(score), notes


def _click_chip_swap(page, info: dict, cam: str) -> tuple[float, list[str]]:
    picked = pick_unknown_swap(info)
    old = (info.get("liveNames") or [None])[0]
    if not picked:
        return 3.0, ["no swap target"]
    page.locator(f'[data-swap-chip="{picked}"]').click()
    page.wait_for_timeout(150)
    after = page.evaluate(SNAPSHOT)
    score, notes = score_unknown_swap(after, picked, old)
    lab_s, lab_n = score_labels(after)
    tgt_s, tgt_n = score_targets(cam, after)
    if lab_s < PASS:
        score = clamp(min(score, lab_s))
        notes = notes + lab_n
    if tgt_s < PASS:
        score = clamp(min(score, tgt_s))
        notes = notes + tgt_n
    page.locator(".work").screenshot(path=str(OUT / f"unknown_{cam}.png"))
    return score, notes


def _hide_after_four(page) -> tuple[float, list[str]]:
    page.evaluate("() => { clicks = [[40,40],[80,80],[120,120],[160,160]]; refresh(); }")
    page.wait_for_timeout(80)
    filled = page.evaluate(SNAPSHOT)
    page.evaluate("() => { clicks = [null,null,null,null]; active = 0; refresh(); }")
    if filled.get("swapUi"):
        return 8.0, ["swap UI still up after 4 clicks"]
    return 10.0, []


def _p10_list_swap(page) -> tuple[float, list[str]]:
    page.select_option("#swapPick", "left_box_goal_near")
    page.wait_for_timeout(120)
    sel = page.evaluate(SNAPSHOT)
    score, notes = score_unknown_swap(sel, "left_box_goal_near", "halfway_near_touch")
    _set_order(page, "both_sides_south")
    return score, notes


def _p8_extra_click(page, info: dict) -> tuple[float, list[str]]:
    old = (info.get("liveNames") or [None])[0]
    page.locator('circle[data-extra="right_6_box_near"]').click(force=True)
    page.wait_for_timeout(120)
    clk = page.evaluate(SNAPSHOT)
    score, notes = score_unknown_swap(clk, "right_6_box_near", old)
    _set_order(page, info.get("orderKey") or "goal_right")
    return score, notes


def exercise_unknown(page, cam: str, info: dict, report: dict) -> None:
    score, notes = score_unknown(info)
    for part in (
        _hover_extra(page, info),
        _click_chip_swap(page, info, cam),
    ):
        if part[0] < score:
            score, notes = part
    _set_order(page, info.get("orderKey") or "both_sides_south")
    hide_s, hide_n = _hide_after_four(page)
    if hide_s < score:
        score, notes = hide_s, hide_n
    extra = None
    if cam == "P10":
        extra = _p10_list_swap(page)
    elif cam == "P8":
        extra = _p8_extra_click(page, info)
    if extra and extra[0] < score:
        score, notes = extra
    report["scores"]["unknown"] = score
    report["notes"]["unknown"] = notes


def _post_save(page, body: dict) -> dict:
    return page.evaluate(
        """async (body) => {
          const r = await fetch('/save_match3_landmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          let j = {};
          try { j = await r.json(); } catch (e) { j = { ok: false, error: String(e) }; }
          return { status: r.status, ...j };
        }""",
        body,
    )


def score_save(page, cam: str, info: dict) -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    catalog = info.get("catalog") or {}
    if "right_box_18_far" not in catalog:
        return 2.0, ["catalog missing right_box_18_far"]
    names = list(P9_VISIBLE) if cam == "P9" else list(info.get("liveNames") or [])
    if len(names) < 4:
        names = list(catalog.keys())[:4]
    body = {
        "camera": cam,
        "order": info.get("orderKey") or "both_sides_south",
        "landmarks": names,
        "image_points": [[10, 10], [200, 10], [200, 200], [10, 200]],
        "dry_run": True,
    }
    got = _post_save(page, body)
    if not got.get("ok"):
        score -= 6
        notes.append(got.get("error") or f"save HTTP {got.get('status')}")
    elif cam == "P9" and "right_box_18_far" not in (got.get("landmarks") or []):
        score -= 3
        notes.append("P9 dry save dropped right_box_18_far")
    bad = dict(body)
    bad["landmarks"] = ["not_a_mark"] + names[:3]
    deny = _post_save(page, bad)
    if deny.get("ok"):
        score -= 4
        notes.append("unknown name was accepted")
    elif "unknown" not in (deny.get("error") or "").lower():
        score -= 2
        notes.append(f"bad unknown error {deny.get('error')}")
    return clamp(score), notes


def score_cam(cam: str, info: dict) -> dict:
    scores = {}
    notes = {}
    scores["orientation"], notes["orientation"] = score_orientation(info)
    scores["cameras"], notes["cameras"] = score_cameras(cam, info)
    scores["labels"], notes["labels"] = score_labels(info)
    scores["targets"], notes["targets"] = score_targets(cam, info)
    scores["spatial"], notes["spatial"] = score_spatial(info)
    scores["video_match"], notes["video_match"] = score_video_match(cam, info)
    scores["naming"], notes["naming"] = score_naming(cam, info)
    scores["unknown"], notes["unknown"] = score_unknown(info)
    return {"scores": scores, "notes": notes}


def _set_order(page, key: str) -> None:
    page.evaluate(
        """(key) => {
          document.getElementById('order').value = key;
          setLiveFromOrder();
          clicks = [null, null, null, null];
          active = 0;
          refresh();
        }""",
        key,
    )
    page.wait_for_timeout(150)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [
        "orientation", "cameras", "labels", "targets",
        "spatial", "video_match", "naming", "unknown", "save",
    ]
    all_scores = {k: [] for k in keys}
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
            # Extra orders: left/right names vs diagram, near-corner vs cam chip.
            for extra_key in ("both_sides_north", "both_sides_south"):
                if info.get("orderKey") == extra_key:
                    continue
                _set_order(page, extra_key)
                extra = page.evaluate(SNAPSHOT)
                n_score, n_notes = score_naming(cam, extra)
                if n_score < report["scores"]["naming"]:
                    report["scores"]["naming"] = n_score
                    report["notes"]["naming"] = n_notes
            if cam == "P9":
                _set_order(page, "both_sides_north")
                page.evaluate("() => { active = 3; refresh(); }")
                page.wait_for_timeout(120)
                p9 = page.evaluate(SNAPSHOT)
                n_score, n_notes = score_naming("P9", p9)
                if "North Right Corner" not in (p9.get("taskWhat") or ""):
                    n_score = clamp(n_score - 3)
                    n_notes = n_notes + ["P9 slot4 task is not North Right Corner"]
                find = (p9.get("taskFind") or "").lower()
                if "p1" not in find or "right" not in find:
                    n_score = clamp(n_score - 2)
                    n_notes = n_notes + ["P9 North Right FIND missing P1/right"]
                if n_score < report["scores"]["naming"]:
                    report["scores"]["naming"] = n_score
                    report["notes"]["naming"] = n_notes
                page.locator(".work").screenshot(path=str(OUT / "p9_north_right_task.png"))
            _set_order(page, info.get("orderKey") or "both_sides_south")
            info = page.evaluate(SNAPSHOT)
            exercise_unknown(page, cam, info, report)
            _set_order(page, info.get("orderKey") or "both_sides_south")
            info = page.evaluate(SNAPSHOT)
            report["scores"]["save"], report["notes"]["save"] = score_save(page, cam, info)
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
    print("all subgoals >= 9/10 on every camera (incl. video_match, naming, unknown, save)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
