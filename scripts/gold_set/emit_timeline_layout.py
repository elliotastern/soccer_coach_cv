"""Timeline label layout — no horizontal overlap, lines at true event time."""
from __future__ import annotations

import html as html_lib

TRACK_WIDTH_PX = 1120
LANE_HEIGHT_PX = 28
MAX_LANES = 6
LABEL_FONT_PX = 13


def timeline_events_near(
    emits: list[dict], cur_t: float, window: float = 15.0, limit: int = 12
) -> list[dict]:
    recent = sorted(
        [e for e in emits if abs(float(e["t_end"]) - cur_t) <= window],
        key=lambda e: float(e["t_end"]),
    )
    return recent[-limit:]


def estimate_label_width_pct(text: str, track_width_px: float = TRACK_WIDTH_PX) -> float:
    px = len(text) * 7.4 + 20.0
    return min(32.0, max(8.0, 100.0 * px / track_width_px))


def event_label_text(event: dict, short: bool = False) -> str:
    tp = str(event.get("type", ""))
    t = float(event.get("t_end", 0))
    if short:
        return f"{tp} {t:.1f}"
    return f"{tp} @{t:.1f}s"


def layout_timeline_events(
    events: list[dict],
    max_s: float,
    track_width_px: float = TRACK_WIDTH_PX,
) -> list[dict]:
    """Place each label in a lane without horizontal box overlap."""
    lanes: list[list[tuple[float, float]]] = []
    placed: list[dict] = []

    for e in sorted(events, key=lambda x: float(x["t_end"])):
        t = float(e["t_end"])
        pct = 0.0 if max_s <= 0 else min(100.0, max(0.0, t / max_s * 100.0))
        short = False
        label_txt = event_label_text(e, short=False)
        half_w = estimate_label_width_pct(label_txt, track_width_px) / 2.0
        left, right = max(0.0, pct - half_w), min(100.0, pct + half_w)

        lane = _pick_lane(lanes, left, right)
        if lane is None:
            short = True
            label_txt = event_label_text(e, short=True)
            half_w = estimate_label_width_pct(label_txt, track_width_px) / 2.0
            left, right = max(0.0, pct - half_w), min(100.0, pct + half_w)
            lane = _pick_lane(lanes, left, right)
        if lane is None:
            lane = min(MAX_LANES - 1, max(0, len(lanes) - 1))
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append((left, right))
        placed.append(
            {
                "event": e,
                "t": t,
                "pct": pct,
                "lane": lane,
                "label": label_txt,
                "left_pct": left,
                "right_pct": right,
            }
        )
    return placed


def _pick_lane(
    lanes: list[list[tuple[float, float]]],
    left: float,
    right: float,
) -> int | None:
    for lane_i, boxes in enumerate(lanes):
        if not _overlaps_any(left, right, boxes):
            return lane_i
    if len(lanes) < MAX_LANES:
        return len(lanes)
    return None


def _overlaps_any(left: float, right: float, boxes: list[tuple[float, float]]) -> bool:
    for b0, b1 in boxes:
        if right > b0 and left < b1:
            return True
    return False


def count_layout_overlaps(placed: list[dict]) -> int:
    """Count pairwise horizontal overlaps (same lane, intersecting boxes)."""
    n = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if placed[i]["lane"] != placed[j]["lane"]:
                continue
            a0, a1 = placed[i]["left_pct"], placed[i]["right_pct"]
            b0, b1 = placed[j]["left_pct"], placed[j]["right_pct"]
            if a1 > b0 and a0 < b1:
                n += 1
    return n


def build_timeline_html(
    placed: list[dict],
    scrub_pct: float,
    t_match: float,
    max_s: float,
    step_s: float,
    playing: bool,
    cur_t: float,
    cur_type: str,
    t_match_scrub: float,
) -> str:
    labels_html = []
    lines_html = []
    dots_html = []
    for row in placed:
        em = row["event"]
        tp = str(em.get("type", ""))
        t = row["t"]
        pct = row["pct"]
        lane = row["lane"]
        col = _event_color(tp)
        is_label = tp == cur_type and abs(t - cur_t) < 0.2
        is_scrub = abs(t - t_match_scrub) < 0.3
        top = lane * LANE_HEIGHT_PX
        border = "2px solid #fff" if is_scrub else "1px solid #333"
        weight = "700" if is_label else "500"
        label_txt = html_lib.escape(row["label"])
        labels_html.append(
            f'<button class="ev-label" data-t="{t:.3f}" data-lane="{lane}" '
            f'style="left:{pct:.2f}%;top:{top}px;background:{col};'
            f'font-weight:{weight};border:{border}" '
            f'onclick="jumpTo({t:.3f})">{label_txt}</button>'
        )
        lines_html.append(f'<div class="ev-line" style="left:{pct:.2f}%"></div>')
        dots_html.append(f'<div class="ev-dot" style="left:{pct:.2f}%;background:{col}"></div>')

    label_rows = max(1, max((r["lane"] for r in placed), default=0) + 1)
    labels_h = label_rows * LANE_HEIGHT_PX + 10
    lines_h = 34
    play_flag = "true" if playing else "false"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #0e1117; color: #fafafa;
    font-family: "Source Sans Pro", sans-serif; }}
  .wrap {{ padding: 4px 10px 0 10px; }}
  .labels {{ position: relative; height: {labels_h}px; margin-bottom: 0; }}
  .ev-label {{
    position: absolute; transform: translateX(-50%);
    padding: 4px 10px; border-radius: 5px; color: #111;
    font-size: {LABEL_FONT_PX}px; cursor: pointer; white-space: nowrap;
    line-height: 1.2; box-shadow: 0 1px 3px rgba(0,0,0,0.45);
  }}
  .ev-label:hover {{ filter: brightness(1.08); }}
  .lines {{ position: relative; height: {lines_h}px; }}
  .ev-line {{
    position: absolute; top: 0; width: 2px; height: 100%;
    background: rgba(255,255,255,0.6); transform: translateX(-50%);
  }}
  .track-wrap {{ position: relative; height: 14px; margin: 0 0 6px 0; }}
  .track {{
    position: absolute; left: 0; right: 0; top: 5px; height: 4px;
    background: #3a3a3a; border-radius: 2px;
  }}
  .ev-dot {{
    position: absolute; top: 2px; width: 10px; height: 10px;
    border-radius: 50%; transform: translateX(-50%);
    border: 1px solid #222;
  }}
  .needle {{
    position: absolute; top: 0; width: 3px; height: 14px;
    background: #ff4b4b; transform: translateX(-50%); border-radius: 1px;
    box-shadow: 0 0 4px rgba(255,75,75,0.8);
  }}
  .scrub-label {{ font-size: 12px; margin-bottom: 2px; color: #ccc; }}
  .scrub-row {{ display: flex; align-items: center; gap: 8px; }}
  input[type=range] {{ flex: 1; accent-color: #ff4b4b; height: 20px; margin: 0; }}
  .scrub-val {{ font-size: 12px; color: #ff4b4b; min-width: 44px; }}
  .tick {{ font-size: 10px; color: #666; margin-top: 0; line-height: 1.1; }}
</style></head><body>
<div class="wrap">
  <div class="labels">{chr(10).join(labels_html)}</div>
  <div class="lines">{chr(10).join(lines_html)}</div>
  <div class="track-wrap">
    <div class="track"></div>
    {chr(10).join(dots_html)}
    <div class="needle" style="left:{scrub_pct:.2f}%"></div>
  </div>
  <div class="scrub-label">Seconds</div>
  <div class="scrub-row">
    <input type="range" id="scrub" min="0" max="{max_s:.4f}" step="{step_s:.4f}"
      value="{t_match:.4f}"
      oninput="scrubLive(this.value)"
      onchange="scrubTo(this.value, {play_flag})">
    <span class="scrub-val" id="sv">{t_match:.1f}s</span>
  </div>
  <div class="tick">0s — {max_s:.1f}s</div>
</div>
<script>
  const MAX_S = {max_s:.4f};
  const IS_PLAYING = {play_flag};
  let scrubTimer = null;

  function setNeedle(t) {{
    const pct = MAX_S > 0 ? Math.min(100, Math.max(0, t / MAX_S * 100)) : 0;
    const needle = document.querySelector(".needle");
    if (needle) needle.style.left = pct.toFixed(2) + "%";
    document.getElementById("sv").textContent = parseFloat(t).toFixed(1) + "s";
  }}

  function scrubLive(t) {{
    setNeedle(t);
    clearTimeout(scrubTimer);
    scrubTimer = setTimeout(() => scrubTo(t, IS_PLAYING), 100);
  }}

  function scrubTo(t, keepPlay) {{
    try {{
      const u = new URL(window.parent.location.href);
      u.searchParams.set("scrub_t", t);
      if (keepPlay) u.searchParams.set("keep_play", "1");
      else u.searchParams.delete("keep_play");
      window.parent.location.href = u.toString();
    }} catch (e) {{
      let q = "scrub_t=" + t;
      if (keepPlay) q += "&keep_play=1";
      window.parent.location.search = q;
    }}
  }}
  function jumpTo(t) {{
    try {{
      const u = new URL(window.parent.location.href);
      u.searchParams.set("jump_t", t);
      window.parent.location.href = u.toString();
    }} catch (e) {{
      window.parent.location.search = "jump_t=" + t;
    }}
  }}
</script>
</body></html>"""


def timeline_html_height(placed: list[dict]) -> int:
    rows = max(1, max((r["lane"] for r in placed), default=0) + 1)
    return rows * LANE_HEIGHT_PX + 10 + 34 + 14 + 54


def _event_color(tp: str) -> str:
    colors = {
        "pass": "#80b4ff",
        "shot": "#4040ff",
        "recovery": "#50dc78",
        "dribble": "#c8a050",
        "movement": "#b4b4b4",
    }
    return colors.get(tp, "#888888")
