"""Generate Phase 1 handover coach validation dashboard HTML."""
from __future__ import annotations

from pathlib import Path

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Phase 1 handover — validate events</title>
  <style>
    :root {
      --bg: #0b1220; --card: #111827; --line: #1f2937;
      --text: #e5e7eb; --muted: #9ca3af; --accent: #22c55e;
      --warn: #f59e0b; --bad: #ef4444; --pass: #3b82f6; --drib: #a855f7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45; }
    header { padding: 14px 20px; border-bottom: 1px solid var(--line); background: #0f172a; }
    h1 { margin: 0 0 4px; font-size: 1.3rem; }
    .sub { color: var(--muted); font-size: 0.9rem; }
    main { max-width: 1280px; margin: 0 auto; padding: 14px 20px 28px; }
    .layout { display: grid; grid-template-columns: 1fr 380px; gap: 14px; align-items: start; }
    @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
    video { width: 100%; border-radius: 8px; background: #000; }
    .legend { font-size: 0.82rem; color: var(--muted); margin-top: 8px; }
    .pill { display: inline-block; background: #1f2937; border-radius: 999px;
      padding: 3px 9px; margin: 2px 4px 2px 0; font-size: 0.8rem; }
    .pill.ok { background: #14532d; color: #86efac; }
    .pill.flag { background: #7f1d1d; color: #fecaca; }
    .pill.pass { background: #1e3a5f; color: #93c5fd; }
    .pill.dribble { background: #4c1d95; color: #d8b4fe; }
    .event-row { border: 1px solid var(--line); border-radius: 10px; padding: 10px;
      margin-bottom: 8px; background: #0b1220; }
    .event-row.good { border-color: #166534; }
    .event-row.bad { border-color: #991b1b; }
    .event-row h3 { margin: 0 0 4px; font-size: 0.95rem; }
    .row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
    button { border-radius: 8px; padding: 7px 11px; cursor: pointer; font-size: 0.85rem;
      border: 1px solid var(--line); background: #1f2937; color: var(--text); }
    button.primary { background: var(--accent); color: #052e16; border: none; font-weight: 700; }
    button.good.active { background: #166534; border-color: var(--accent); }
    button.bad.active { background: var(--bad); border-color: #f87171; }
    button.secondary { background: #1f2937; }
    label.group { display: block; margin: 8px 0 4px; font-weight: 600; font-size: 0.85rem; }
    .opts { display: flex; flex-wrap: wrap; gap: 5px; }
    button.opt { padding: 5px 9px; }
    button.opt.active { background: #2563eb; border-color: #3b82f6; }
    button.opt.good.active { background: #166534; }
    button.opt.bad.active { background: var(--bad); }
    textarea { width: 100%; min-height: 56px; margin-top: 6px; border-radius: 8px;
      border: 1px solid var(--line); background: #0b1220; color: var(--text); padding: 8px; }
    #status { margin-top: 8px; color: var(--muted); font-size: 0.85rem; }
    a { color: #93c5fd; }
    .timeline { height: 28px; background: #1f2937; border-radius: 6px; position: relative;
      margin: 10px 0 4px; }
    .tick { position: absolute; top: 2px; width: 10px; height: 22px; border-radius: 3px;
      transform: translateX(-50%); cursor: pointer; opacity: 0.85; }
    .tick.pass { background: var(--pass); }
    .tick.dribble { background: var(--drib); }
    .tick.active { outline: 2px solid #fff; opacity: 1; }
  </style>
</head>
<body>
<header>
  <h1>Phase 1 — validate mosaic + pitch + events</h1>
  <div class="sub">20 s fuse clip · mosaic (P10|P9 / P7|P8) + Pitch 1 map + events bar · stride 1 · 60 fps</div>
</header>
<main>
  <div class="layout">
    <div class="card">
      <video id="vid" controls playsinline preload="metadata" src="coach_mosaic_pitch_min.mp4"></video>
      <div class="row" style="margin-top:8px">
        <button type="button" class="secondary" id="prev">← Frame</button>
        <button type="button" class="secondary" id="next">Frame →</button>
        <span class="pill" id="timePill">t+0.0s</span>
        <span class="pill" id="frPill">fr —</span>
      </div>
      <div class="timeline" id="timeline"></div>
      <p class="legend">Orange boxes = detections · yellow dot = fused ball on map · flashes = emitted events (pass blue · dribble purple)</p>
      <p id="eventsHere" class="sub"></p>
    </div>
    <div>
      <div class="card" style="margin-bottom:12px">
        <h2 style="margin:0 0 8px;font-size:1.05rem">Events to validate</h2>
        <p class="sub" style="margin:0 0 8px">Confirm or flag each emit, then merge into fuse gold.</p>
        <div id="eventList"></div>
        <div class="row">
          <button type="button" class="secondary" id="confirmAll">Confirm all</button>
          <button type="button" class="primary" id="mergeGold">Merge fuse gold</button>
        </div>
      </div>
      <div class="card">
        <h2 style="margin:0 0 6px;font-size:1.05rem">Frame QA</h2>
        <div id="stats" class="sub" style="margin-bottom:6px"></div>
        <label class="group">Ball visible?</label>
        <div class="opts" data-field="ball_visible" data-values="yes,no,unclear"></div>
        <label class="group">Ball box OK?</label>
        <div class="opts" data-field="ball_box_ok" data-values="good,bad,na"></div>
        <label class="group">Map ball dot OK?</label>
        <div class="opts" data-field="pitch_ball_ok" data-values="good,bad,na"></div>
        <label class="group">Team colours OK?</label>
        <div class="opts" data-field="team_ok" data-values="good,bad,na"></div>
        <label class="group">Event flash sensible?</label>
        <div class="opts" data-field="event_ok" data-values="good,bad,na"></div>
        <label class="group">Flag follow-up</label>
        <div class="opts" data-field="flag" data-values="false,true"></div>
        <textarea id="note" placeholder="Note for engineer…"></textarea>
        <div class="row">
          <button type="button" class="primary" id="save">Save frame</button>
          <button type="button" class="secondary" id="export">Download labels</button>
        </div>
        <div id="status"></div>
      </div>
    </div>
  </div>
</main>
<script>
const QA = { good: "Good", bad: "Bad", na: "Not sure", unset: "—" };
const VIS = { yes: "Yes", no: "No", unclear: "Not sure" };
let meta = null, emits = [], labels = { reviewer: "", frames: {} }, clipIdx = 0;

function frameKey(fid) { return `fr_${fid}`; }

function defaultFrame() {
  return { ball_visible: "unclear", ball_box_ok: "unset", pitch_ball_ok: "unset",
    team_ok: "unset", event_ok: "unset", flag: false, note: "", reviewed: false };
}

function nearestIdx(t) {
  const fps = meta?.out_fps || 15;
  let i = Math.round(t * fps);
  return Math.max(0, Math.min(i, (meta?.frames_src?.length || 1) - 1));
}

function srcFrame() {
  return meta?.frames_src?.[clipIdx] ?? null;
}

function loadCur() {
  const fid = srcFrame();
  if (fid == null) return defaultFrame();
  return { ...defaultFrame(), ...(labels.frames[frameKey(fid)] || {}) };
}

function setFrame(fid, patch) {
  const key = frameKey(fid);
  labels.frames[key] = { ...defaultFrame(), ...(labels.frames[key] || {}), ...patch };
}

function paintOpts() {
  const cur = loadCur();
  document.querySelectorAll(".opts[data-field]").forEach((wrap) => {
    const field = wrap.dataset.field;
    const vals = wrap.dataset.values.split(",");
    wrap.innerHTML = "";
    vals.forEach((v) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "opt";
      if (field === "flag") {
        b.textContent = v === "true" ? "Flag" : "OK";
        if (String(cur.flag) === v) b.classList.add(v === "true" ? "bad" : "good");
      } else if (field === "ball_visible") {
        b.textContent = VIS[v] || v;
        if (cur.ball_visible === v) b.classList.add("active");
      } else {
        b.textContent = QA[v] || v;
        if (cur[field] === v) b.classList.add(v === "good" ? "good" : v === "bad" ? "bad" : "active");
      }
      b.onclick = () => {
        const fid = srcFrame();
        if (fid == null) return;
        const rec = loadCur();
        if (field === "flag") rec.flag = v === "true";
        else rec[field] = v;
        labels.frames[frameKey(fid)] = rec;
        paintOpts();
        paintEventList();
      };
      wrap.appendChild(b);
    });
  });
  document.getElementById("note").value = cur.note || "";
}

function paintStats() {
  const frames = Object.values(labels.frames);
  const n = frames.filter((f) => f.reviewed).length;
  const flagged = frames.filter((f) => f.flag).length;
  const goodEv = frames.filter((f) => f.event_ok === "good").length;
  document.getElementById("stats").textContent =
    `Reviewed ${n} frames · ${goodEv} event OK · flagged ${flagged}`;
}

function jumpToEmit(em) {
  const idx = meta.frames_src.indexOf(em.frame_id);
  if (idx >= 0) {
    clipIdx = idx;
    document.getElementById("vid").currentTime = idx / (meta.out_fps || 15);
  }
  paintTime();
}

function paintEventList() {
  const dur = meta?.duration_s || 15;
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  emits.forEach((em) => {
    const tick = document.createElement("div");
    tick.className = `tick ${em.type === "dribble" ? "dribble" : "pass"}`;
    tick.style.left = `${(em.t_end / dur) * 100}%`;
    tick.title = `${em.type} @ ${em.t_end.toFixed(2)}s`;
    tick.onclick = () => jumpToEmit(em);
    tl.appendChild(tick);
  });
  const list = document.getElementById("eventList");
  list.innerHTML = "";
  emits.forEach((em, i) => {
    const fr = labels.frames[frameKey(em.frame_id)] || {};
    const row = document.createElement("div");
    row.className = "event-row";
    if (fr.event_ok === "good") row.classList.add("good");
    if (fr.event_ok === "bad") row.classList.add("bad");
    const pill = em.type === "dribble" ? "dribble" : "pass";
    row.innerHTML = `<h3><span class="pill ${pill}">${em.type.toUpperCase()}</span>
      t=${em.t_end.toFixed(2)}s · conf ${em.confidence} · fr ${em.frame_id}</h3>
      <div class="sub">${fr.event_ok === "good" ? "✓ confirmed" : fr.event_ok === "bad" ? "✗ flagged" : "not reviewed"}</div>`;
    const btns = document.createElement("div");
    btns.className = "row";
    const jump = document.createElement("button");
    jump.textContent = "Jump";
    jump.onclick = () => jumpToEmit(em);
    const ok = document.createElement("button");
    ok.textContent = "Confirm";
    ok.className = fr.event_ok === "good" ? "good active" : "";
    ok.onclick = () => markEvent(em, "good");
    const bad = document.createElement("button");
    bad.textContent = "Flag";
    bad.className = fr.event_ok === "bad" ? "bad active" : "";
    bad.onclick = () => markEvent(em, "bad");
    btns.append(jump, ok, bad);
    row.appendChild(btns);
    list.appendChild(row);
  });
}

function markEvent(em, verdict) {
  setFrame(em.frame_id, {
    ball_visible: "yes", ball_box_ok: "good", pitch_ball_ok: "good",
    event_ok: verdict, flag: verdict === "bad", reviewed: true,
    reviewed_at: new Date().toISOString(), suggested_type: em.type,
    note: verdict === "good" ? `confirm ${em.type}` : `reject ${em.type}`,
  });
  paintEventList();
  paintOpts();
  paintStats();
}

function paintTime() {
  const vid = document.getElementById("vid");
  clipIdx = nearestIdx(vid.currentTime);
  const fid = srcFrame();
  document.getElementById("timePill").textContent = `t+${vid.currentTime.toFixed(2)}s`;
  document.getElementById("frPill").textContent = fid != null ? `fr ${fid}` : "fr —";
  const here = emits.filter((e) => e.frame_id === fid);
  document.getElementById("eventsHere").textContent = here.length
    ? "At this frame: " + here.map((e) => `${e.type} (${e.confidence})`).join(" · ")
    : "No event flash at this frame.";
  paintOpts();
  paintStats();
  paintEventList();
}

async function saveLabels() {
  labels.updated_at = new Date().toISOString();
  const res = await fetch("/save_phase1_handover_labels", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(labels),
  });
  return res.json();
}

async function loadAll() {
  const ts = Date.now();
  meta = await fetch(`meta.json?v=${ts}`).then((r) => r.json());
  emits = (await fetch(`emits_render.json?v=${ts}`).then((r) => r.json())).emits || [];
  try {
    labels = await fetch(`labels.json?v=${ts}`).then((r) => r.json());
    labels.frames = labels.frames || {};
  } catch (_) {
    labels = { reviewer: "", frames: {} };
  }
  if (!labels.suggested_events?.length && emits.length) {
    labels.suggested_events = emits.map((e) => ({
      type: e.type, t_start: e.t_start, t_end: e.t_end,
      confidence: e.confidence, frame_id: e.frame_id,
    }));
  }
  paintTime();
}

document.getElementById("vid").addEventListener("timeupdate", paintTime);
document.getElementById("vid").addEventListener("seeked", paintTime);
document.getElementById("prev").onclick = () => {
  clipIdx = Math.max(0, clipIdx - 1);
  document.getElementById("vid").currentTime = clipIdx / (meta.out_fps || 15);
};
document.getElementById("next").onclick = () => {
  clipIdx = Math.min((meta.frames_src?.length || 1) - 1, clipIdx + 1);
  document.getElementById("vid").currentTime = clipIdx / (meta.out_fps || 15);
};
document.getElementById("save").onclick = async () => {
  const fid = srcFrame();
  if (fid == null) return;
  const rec = loadCur();
  rec.note = document.getElementById("note").value.trim();
  rec.reviewed = true;
  rec.reviewed_at = new Date().toISOString();
  labels.frames[frameKey(fid)] = rec;
  const js = await saveLabels();
  document.getElementById("status").textContent = js.ok ? `Saved frame ${fid}` : `Fail: ${js.error}`;
  paintStats();
};
document.getElementById("confirmAll").onclick = async () => {
  emits.forEach((em) => markEvent(em, "good"));
  labels.reviewer = labels.reviewer || "coach";
  const js = await saveLabels();
  document.getElementById("status").textContent = js.ok
    ? `Confirmed ${emits.length} events` : `Fail: ${js.error}`;
};
document.getElementById("mergeGold").onclick = async () => {
  const res = await fetch("/merge_phase1_handover_gold", { method: "POST" });
  const js = await res.json();
  document.getElementById("status").textContent = js.ok
    ? `Merged ${js.coach_events || 0} coach events into fuse gold`
    : `Merge failed: ${js.error || res.status}`;
};
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(labels, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "phase1_handover_labels.json";
  a.click();
};
loadAll();
</script>
</body>
</html>
"""


def write_index(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(HTML, encoding="utf-8")
    return path
