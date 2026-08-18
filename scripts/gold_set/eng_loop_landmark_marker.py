"""Playwright: 4 yellow points on pitch, both sidelines, compass, labels clear of lines."""
from __future__ import annotations

import math
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/eval_match3/landmark_dashboard/eng_loop"
URL = "http://127.0.0.1:8080/reports/eval_match3/landmark_dashboard/index.html?v=no-popup-overlap"
CAMS = ["P10", "P7", "P9", "P8", "P1", "P6", "P_Goal1", "P_Goal2"]
SVG_W = 560.0

DIAGRAM = """
() => {
  const svg = document.getElementById('names');
  const pitch = svg.querySelector('rect');
  const plot = document.querySelector('.map-plot').getBoundingClientRect();
  const pb = pitch.getBoundingClientRect();
  const dots = [...svg.querySelectorAll('circle[data-slot]')];
  const tags = [...document.querySelectorAll('#nameLabels .name-tag[data-slot]')];
  const cams = [...svg.querySelectorAll('[data-cam]')].map(g => {
    const c = g.querySelector('circle');
    const t = g.querySelector('text');
    const b = c.getBoundingClientRect();
    return {
      id: g.dataset.cam,
      cx: +c.getAttribute('cx'),
      cy: +c.getAttribute('cy'),
      r: +c.getAttribute('r'),
      fs: +t.getAttribute('font-size'),
      screenR: b.width / 2,
      on: c.getAttribute('fill') === '#e8c547',
      outsidePitch: b.right < pb.left - 2 || b.left > pb.right + 4 ||
                   b.bottom < pb.top - 2 || b.top > pb.bottom + 4,
    };
  });
  const expected = liveOrder.map((lm, i) => {
    const [sx, sy] = m2svg(lm.xy[0], lm.xy[1]);
    return { slot: i, name: lm.name, title: prettyName(lm), sx, sy };
  });
  return {
    expected,
    wantCams: (data.cams || []).map(c => c.id),
    cams,
    dots: dots.map(c => {
      const b = c.getBoundingClientRect();
      return {
        slot: +c.dataset.slot,
        name: c.dataset.name,
        cx: +c.getAttribute('cx'),
        cy: +c.getAttribute('cy'),
        onPitch: b.right >= pb.left - 4 && b.left <= pb.right + 4 &&
                 b.bottom >= pb.top - 4 && b.top <= pb.bottom + 4,
      };
    }),
    tags: tags.map(t => {
      const b = t.getBoundingClientRect();
      return {
        slot: +t.dataset.slot,
        name: t.dataset.name,
        text: t.textContent.trim(),
        onPlot: b.right > plot.left && b.left < plot.right &&
                b.bottom > plot.top && b.top < plot.bottom,
        fs: parseFloat(getComputedStyle(t).fontSize),
        h: b.height,
        coversLine: (() => {
          const sr = svg.getBoundingClientRect();
          const lab = {
            lx: b.left + b.width / 2 - sr.left,
            ly: b.top + b.height / 2 - sr.top,
            tw: b.width, th: b.height,
          };
          return hitsPitchLinesPx(svg, lab, 6);
        })(),
      };
    }),
    compass: {
      n: (document.querySelector('.compass-n') || {}).textContent || '',
      s: (document.querySelector('.compass-s') || {}).textContent || '',
      w: (document.querySelector('.compass-w') || {}).textContent || '',
      e: (document.querySelector('.compass-e') || {}).textContent || '',
    },
  };
}
"""


def side_of(name: str) -> str:
    if "Left" in name:
        return "left"
    if "Right" in name:
        return "right"
    return "mid"


def check_compass(cam: str, compass: dict, missing: list) -> None:
    n = compass.get("n", "")
    w = compass.get("w", "")
    e = compass.get("e", "")
    if "NORTH" not in n:
        missing.append(f"{cam}: missing NORTH compass ({n!r})")
    if "LEFT" not in w:
        missing.append(f"{cam}: missing LEFT compass ({w!r})")
    if "RIGHT" not in e:
        missing.append(f"{cam}: missing RIGHT compass ({e!r})")


def check_cameras(cam: str, info: dict, missing: list) -> None:
    want = info.get("wantCams") or CAMS
    have = {c["id"]: c for c in info.get("cams") or []}
    for cid in want:
        m = have.get(cid)
        if not m:
            missing.append(f"{cam}: missing camera chip {cid}")
            continue
        if not m["outsidePitch"]:
            missing.append(f"{cam}: camera chip {cid} should sit outside the pitch")
        if m["r"] < 42 or m["fs"] < 26:
            missing.append(
                f"{cam}: camera chip {cid} too small r={m['r']} fs={m['fs']}"
            )
        if m["screenR"] < 18:
            missing.append(
                f"{cam}: camera chip {cid} screen radius {m['screenR']:.1f}px too small"
            )
    active = [c["id"] for c in have.values() if c.get("on")]
    if cam not in have:
        missing.append(f"{cam}: active camera chip missing")
    elif cam not in active:
        missing.append(f"{cam}: camera chip {cam} is not highlighted")
    if len(active) != 1:
        missing.append(f"{cam}: expected 1 highlighted cam chip, got {active}")


def check_diagram(cam: str, info: dict, missing: list) -> None:
    if len(info["dots"]) != 4 or len(info["expected"]) != 4:
        missing.append(
            f"{cam}: expected 4 diagram dots, got {len(info['dots'])} "
            f"live {len(info['expected'])}"
        )
        return
    by_slot = {d["slot"]: d for d in info["dots"]}
    tags = {t["slot"]: t for t in info["tags"]}
    left_x, right_x = [], []
    for exp in info["expected"]:
        i = exp["slot"]
        title = exp["title"]
        dot = by_slot.get(i)
        if not dot:
            missing.append(f"{cam}: no diagram dot for slot {i + 1} {title}")
            continue
        if not dot["onPitch"]:
            missing.append(f"{cam}: slot {i + 1} {title} is not on the pitch drawing")
        dist = math.hypot(dot["cx"] - exp["sx"], dot["cy"] - exp["sy"])
        if dist > 2:
            missing.append(
                f"{cam}: {title} at ({dot['cx']:.1f},{dot['cy']:.1f}) "
                f"want ({exp['sx']:.1f},{exp['sy']:.1f})"
            )
        tag = tags.get(i)
        if not tag or title not in tag["text"]:
            missing.append(f"{cam}: no on-diagram label for {title}")
        elif not tag["onPlot"]:
            missing.append(f"{cam}: label '{tag['text']}' clipped off the diagram")
        elif tag["fs"] < 18 or tag["h"] < 26:
            missing.append(f"{cam}: '{tag['text']}' fs={tag['fs']:.1f} h={tag['h']:.1f}")
        elif tag.get("coversLine"):
            missing.append(f"{cam}: label '{tag['text']}' covers pitch lines")
        side = side_of(title)
        if side == "left":
            if dot["cx"] >= SVG_W / 2 - 8:
                missing.append(f"{cam}: {title} should be on the left of the diagram")
            left_x.append(dot["cx"])
        elif side == "right":
            if dot["cx"] <= SVG_W / 2 + 8:
                missing.append(f"{cam}: {title} should be on the right of the diagram")
            right_x.append(dot["cx"])
    if not left_x or not right_x:
        missing.append(f"{cam}: diagram does not mark both sidelines")
    elif max(left_x) >= min(right_x):
        missing.append(f"{cam}: left/right dots are not split across the pitch")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#names")
        page.wait_for_timeout(350)
        missing = []
        for cam in CAMS:
            page.locator(f'.cams button[data-id="{cam}"]').click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(400)
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
            labels = page.evaluate(
                """() => [...document.querySelectorAll('#nameLabels .name-tag, #names text')]
                    .map(t => (t.textContent || '').trim()).filter(Boolean)"""
            )
            info = page.evaluate(DIAGRAM)
            print(cam, [
                (d["slot"] + 1, d["name"], round(d["cx"], 1), round(d["cy"], 1), d["onPitch"])
                for d in info["dots"]
            ])
            check_compass(cam, info["compass"], missing)
            check_cameras(cam, info, missing)
            check_diagram(cam, info, missing)
            want = [e["title"] for e in info["expected"]]
            page.locator(".work").screenshot(path=str(OUT / f"mark_{cam}.png"))
            page.locator(".map-panel").screenshot(path=str(OUT / f"names_{cam}.png"))
            task = page.locator("#taskWhat").inner_text()
            find = page.locator("#taskFind").inner_text()
            if want[0] not in task:
                missing.append(f"{cam}: task '{task}' missing {want[0]}")
            if len(find) < 12:
                missing.append(f"{cam}: find hint too short: {find}")
            overlap = page.evaluate(
                """() => {
                  const pad = 6;
                  const pills = [...document.querySelectorAll('#nameLabels .name-tag')].map(r => ({
                    t: r.textContent.trim(),
                    b: (() => { const x = r.getBoundingClientRect();
                      return {l:x.left, r:x.right, t:x.top, b:x.bottom}; })(),
                    quiet: r.classList.contains('quiet'),
                    fs: parseFloat(getComputedStyle(r).fontSize),
                    h: r.getBoundingClientRect().height,
                  }));
                  const cams = [...document.querySelectorAll('#names [data-cam]')].map(g => {
                    const x = g.getBoundingClientRect();
                    return { t: g.dataset.cam,
                      b: {l:x.left, r:x.right, t:x.top, b:x.bottom} };
                  });
                  const all = [...pills, ...cams];
                  for (let i = 0; i < pills.length; i++) {
                    if (pills[i].quiet && (pills[i].fs < 14 || pills[i].h < 20)) {
                      return `small ${pills[i].t} fs=${pills[i].fs} h=${pills[i].h}`;
                    }
                  }
                  for (let i = 0; i < all.length; i++) {
                    for (let j = i + 1; j < all.length; j++) {
                      const a = all[i].b, b = all[j].b;
                      const hit = a.l - pad < b.r + pad && a.r + pad > b.l - pad &&
                        a.t - pad < b.b + pad && a.b + pad > b.t - pad;
                      if (hit) return `${all[i].t} / ${all[j].t}`;
                    }
                  }
                  return '';
                }"""
            )
            if overlap:
                missing.append(f"{cam}: popup overlap {overlap}")
            n_tags = page.evaluate(
                """() => document.querySelectorAll('#nameLabels .name-tag').length"""
            )
            if n_tags != 4:
                missing.append(f"{cam}: expected 4 name tags, got {n_tags}")
            boxes = page.evaluate(
                """() => {
                  const n = document.querySelector('#names rect[data-box="north"]');
                  const s = document.querySelector('#names rect[data-box="south"]');
                  const named = [...document.querySelectorAll(
                    '#names circle[data-extra], #names circle[data-name]'
                  )].map(c => c.dataset.extra || c.dataset.name);
                  return {
                    north: n && { y: +n.getAttribute('y'), h: +n.getAttribute('height') },
                    south: s && { y: +s.getAttribute('y'), h: +s.getAttribute('height') },
                    named,
                  };
                }"""
            )
            if not boxes["north"] or not boxes["south"]:
                missing.append(f"{cam}: need goal boxes at both ends")
            elif boxes["south"]["y"] <= boxes["north"]["y"] + boxes["north"]["h"]:
                missing.append(f"{cam}: south box is not below the north box")
            names_on = boxes["named"]
            if not any(n.startswith("left_box") for n in names_on):
                missing.append(f"{cam}: south box dots missing from diagram")
            if not any(n.startswith("right_box") for n in names_on):
                missing.append(f"{cam}: north box dots missing from diagram")
            for n in want:
                if n not in " ".join(labels):
                    missing.append(f"{cam}: missing '{n}' in {labels}")
        browser.close()
    if missing:
        print("DIAGRAM MISS")
        for m in missing:
            print(" ", m)
        return 1
    print("no popup overlaps; compass + P chips ok; labels clear of lines")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
