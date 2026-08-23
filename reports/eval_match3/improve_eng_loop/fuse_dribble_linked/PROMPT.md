# Fuse dribble on stable-id timeline — eng-loop prompt (#1)

**#1 priority:** stable fuse ids exposed false dribble cling (teleporting player); true carrier ~3 m from ball on fuse xy  
**Entrypoint:** `python3 scripts/eng_loop_fuse_dribble_linked.py` → gates ≥ **9/10**

---

## Problem

After `fuse_player_id_stable`, carrier id is **stable** (34 through carry) but **dribble vanished** on linked timeline: co-move cos fails when fuse xy lags ball ~2.7–3 m.

Raw timeline dribble used a **teleporting** player id (wrong attribution).

**Goal:** Proximity cling for slow fuse carries; dribble TP on **linked** timeline with **correct** `involved_players`; keep batch cap and parent loops green.

---

## Strategy

```mermaid
flowchart LR
  A[Co-move cling] --> B[Proximity fallback 3.5m]
  B --> C[Linked timeline dribble TP]
  C --> D[Stable carrier pid on emit]
```

| Change | Detail |
|--------|--------|
| `_dribble_proximity_ok` | Same stable pid near ball prev+cur, ball speed &lt; pass×0.92 |
| `_dribble_cling_step` | Fallback when cos &lt; threshold |
| No emit conf drop | ≥ 0.80 |

---

## Gates

| Id | Pass |
|----|------|
| 01 | PROMPT + PROMPT_EVAL |
| 02 | heuristic + dribble_precision + fuse_event_recall + fuse_carry + fuse_player_id regression |
| 03 | **Linked** timeline dribble TP ≥1 (10.5–11.5 s) |
| 04 | Linked dribble `involved_players` = stable carrier id (not slot shuffle) |
| 05 | Raw timeline dribble TP still ≥1 |
| 06 | Batch dribble ≤ 30 |

Report: `reports/eval_match3/improve_eng_loop/fuse_dribble_linked/scores.json`

---

## Done

Linked fuse gold dribbles with honest carrier pid; eng-loop exit 0.
