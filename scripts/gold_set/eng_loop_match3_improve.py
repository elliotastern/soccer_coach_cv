#!/usr/bin/env python3
"""Score Match 3 multicam improve plan + T1 wire (need 9+/10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))
PLAN = ROOT / "docs/product/MATCH3_MULTICAM_IMPROVE_PLAN.md"
POLICY = ROOT / "scripts/gold_set/multicam_select_policy.py"
DEMO = ROOT / "scripts/gold_set/demo_locked_oos_pitchmap.py"
OUT = ROOT / "reports/eval_match3/improve_eng_loop"
PASS = 9.0


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def score_plan() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    text = PLAN.read_text(encoding="utf-8") if PLAN.is_file() else ""
    if not text:
        return 0.0, ["plan missing"]
    low = text.lower()
    need = [
        ("p_emit", "P_emit goal"),
        ("0.80", "0.80 gate"),
        ("clear-ball", "clear-ball R"),
        ("4 m", "agree m"),
        ("do not average", "no midpoint / hard no"),
        ("match-3 detect thr", "T1 thr"),
        ("overlapping", "L2 landmarks"),
        ("min_support", "H1 hull"),
        ("pitch 1", "Pitch 1 meters"),
        ("phase 2", "no Phase 2 fusion"),
        ("video title", "cam id"),
        ("eng_loop_match3_improve", "eng-loop wire"),
    ]
    for needle, label in need:
        if needle not in low:
            score -= 0.8
            notes.append(f"plan missing {label}")
    if "≥ 0.80" not in text and ">= 0.80" not in text:
        score -= 1.0
        notes.append("plan missing ≥ 0.80 wording")
    if "p7" not in low and "0.60" not in text:
        score -= 0.5
        notes.append("plan should call out P7@0.60")
    return clamp(score), notes


def score_t1_wire() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    from multicam_select_policy import MATCH3_THR_BY_CAM, TOP_LEFT_THR_BY_CAM, thr_for_cam

    if thr_for_cam(MATCH3_THR_BY_CAM, "P7") != 0.20:
        score -= 4.0
        notes.append("MATCH3 P7 thr != 0.20")
    if thr_for_cam(TOP_LEFT_THR_BY_CAM, "P7") != 0.60:
        score -= 2.0
        notes.append("Match2 P7 thr should stay 0.60")
    demo = DEMO.read_text(encoding="utf-8") if DEMO.is_file() else ""
    if "MATCH3_THR_BY_CAM" not in demo:
        score -= 3.0
        notes.append("demo missing MATCH3_THR_BY_CAM")
    if "match3_all_cam_thr020" not in POLICY.read_text(encoding="utf-8"):
        score -= 2.0
        notes.append("policy missing MATCH3_POLICY_ID")
    if not notes:
        notes.append("T1 wired")
    return clamp(score), notes


def score_l1_calibs() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    for cam in ("P8", "P9", "P_Goal1"):
        path = ROOT / f"reports/eval_match3/match3_pitch_calib/{cam}_manual.json"
        if not path.is_file():
            score -= 3.0
            notes.append(f"{cam} calib missing")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("version") != "manual_clicks":
            score -= 2.5
            notes.append(f"{cam} not manual_clicks")
        if len(rec.get("landmark_names") or []) < 4:
            score -= 2.5
            notes.append(f"{cam} <4 clicks")
        if any("penalty" in n for n in rec.get("landmark_names") or []):
            score -= 2.0
            notes.append(f"{cam} has penalty invent")
    if not notes:
        notes.append("L1 DLT on P8/P9/P_Goal1")
    return clamp(score), notes


def score_l2_overlap() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    for cam in ("P1", "P6"):
        path = ROOT / f"reports/eval_match3/match3_pitch_calib/{cam}_manual.json"
        if not path.is_file():
            score -= 4.0
            notes.append(f"{cam} missing")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        names = rec.get("landmark_names") or []
        if len(names) < 5:
            score -= 3.0
            notes.append(f"{cam} <5 landmarks")
        if "center" not in names:
            score -= 3.0
            notes.append(f"{cam} missing center overlap")
        if float(rec.get("roundtrip_max_m", 99.0)) > 0.15:
            score -= 2.0
            notes.append(f"{cam} RT > 0.15")
    if not notes:
        notes.append("L2 P1/P6 center overlap ≥5")
    return clamp(score), notes


def score_h1_support() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    text = (ROOT / "src/mapping/match3_xy.py").read_text(encoding="utf-8")
    if "MIN_SUPPORT = 0.20" not in text and "MIN_SUPPORT=0.20" not in text:
        score -= 4.0
        notes.append("MIN_SUPPORT not locked at 0.20 (H1 promote)")
    ab = OUT / "h1_minsupport_ab.json"
    if ab.is_file():
        data = json.loads(ab.read_text(encoding="utf-8"))
        if not data.get("promote_0_20"):
            score -= 2.0
            notes.append("h1_minsupport_ab did not promote 0.20")
        else:
            notes.append("H1 MIN_SUPPORT=0.20 (holdout A/B promote)")
    else:
        ab_old = OUT / "h1_support_ab.json"
        if ab_old.is_file():
            rows = json.loads(ab_old.read_text(encoding="utf-8")).get("ab") or {}
            a = rows.get("0.35") or {}
            b = rows.get("0.25") or {}
            if int(b.get("agree") or 0) < int(a.get("agree") or 0):
                score -= 3.0
                notes.append("0.25 agree did not beat 0.35")
        else:
            score -= 1.0
            notes.append("missing h1_minsupport_ab.json")
    if not notes:
        notes.append("H1 MIN_SUPPORT=0.20")
    return clamp(score), notes


def score_f_post() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    path = OUT / "f_post_ab.json"
    if not path.is_file():
        return 0.0, ["missing f_post_ab.json — run ab_match3_fuse_post.py"]
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner")
    allowed = {
        "F1",
        "F2",
        "F3",
        "F1+F2",
        "F1+F2+F3",
        "F1+F2+F0",
        "F1+F2+F0+F3",
    }
    if not winner:
        score -= 5.0
        notes.append("no winner passing P_emit>=0.80")
    elif winner not in allowed:
        score -= 4.0
        notes.append(f"winner {winner} not in F0–F3 set")
    strips = data.get("strips") or {}
    # Prefer multi-strip shape; fall back to legacy flat variants
    locked = None
    if strips:
        for block in strips.values():
            variants = block.get("variants") or {}
            locked = (
                variants.get("F1+F2+F0+F3")
                or variants.get("F1+F2+F0")
                or variants.get(winner)
                or {}
            )
            if not locked.get("poc_pass_P_emit"):
                score -= 3.0
                notes.append(f"{block.get('pack')} failed P_emit")
            if float(locked.get("clear_ball_R") or 0) < 0.80:
                score -= 2.0
                notes.append(f"{block.get('pack')} clear_R < 0.80")
    else:
        variants = data.get("variants") or {}
        locked = variants.get("F1+F2+F0") or variants.get("F1+F2") or {}
        if not locked.get("poc_pass_P_emit"):
            score -= 4.0
            notes.append("F1+F2(+F0) failed P_emit gate")
        if float(locked.get("clear_ball_R") or 0) < 0.80:
            score -= 3.0
            notes.append("F post clear_ball_R < 0.80")
    text = (ROOT / "src/mapping/match3_xy.py").read_text(encoding="utf-8")
    if "soft_dual_fallback" not in text or "solo_max_conf" not in text:
        score -= 3.0
        notes.append("fuse_balls missing F1/F2 flags")
    if "fuse_balls_with_hold" not in text or "HOLD_MAX_GAP" not in text:
        score -= 2.0
        notes.append("F0 hold helper missing")
    if "prune_ghost_maps" not in text or "GHOST_CONF" not in text:
        score -= 2.0
        notes.append("F3 ghost prune missing")
    if not notes:
        notes.append(f"F post winner={winner}")
    return clamp(score), notes


def score_c1_fn_audit() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    path = OUT / "c1_fn_audit.json"
    if not path.is_file():
        return 0.0, ["missing c1_fn_audit.json — run fn_audit_match3_quad.py"]
    data = json.loads(path.read_text(encoding="utf-8"))
    strips = data.get("strips") or []
    if len(strips) < 2:
        score -= 2.0
        notes.append("need P10 + P8 strip audits")
    for row in strips:
        pack = row.get("pack") or "?"
        r = float(row.get("clear_ball_R") or 0.0)
        if pack == "match3_quad_p8_87" and r < 0.80:
            score -= 3.0
            notes.append(f"P8 strip clear_R {r} < 0.80 after hull fix")
        if pack == "match3_quad_p10_31" and r < 0.80:
            score -= 2.0
            notes.append(f"P10 strip clear_R {r} < 0.80")
    p8 = next((s for s in strips if s.get("pack") == "match3_quad_p8_87"), {})
    map_fail = int((p8.get("fn_buckets") or {}).get("focus_map_fail") or 0)
    if map_fail > 10:
        score -= 2.0
        notes.append(f"P8 focus_map_fail still {map_fail}")
    if not notes:
        notes.append("C1 FN audit + hull ok")
    return clamp(score), notes


def score_c3_p6_hull() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    path = ROOT / "reports/eval_match3/match3_pitch_calib/P6_manual.json"
    if not path.is_file():
        return 0.0, ["P6 calib missing"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    hull = rec.get("hull_image_points") or []
    if len(hull) <= len(rec.get("image_points") or []):
        score -= 4.0
        notes.append("P6 missing near-touch hull expand")
    # Evidence: P9 t00559 proxy R after hull
    audit = OUT / "c1_fn_audit.json"
    if audit.is_file():
        data = json.loads(audit.read_text(encoding="utf-8"))
        row = next(
            (
                r
                for r in data.get("quad_caches") or []
                if "P9_t00559" in (r.get("cache") or "")
            ),
            None,
        )
        if row is not None:
            r = float(row.get("clear_ball_proxy_R") or 0)
            if r < 0.70:
                score -= 3.0
                notes.append(f"P9 t00559 proxy R {r} < 0.70 after P6 hull")
            elif r < 0.80:
                score -= 1.0
                notes.append(f"P9 t00559 proxy R {r} short of 0.80")
    else:
        score -= 1.0
        notes.append("missing c1_fn_audit for C3 check")
    if not notes:
        notes.append("C3 P6 near-touch hull ok")
    return clamp(score), notes


def score_c2_quad_det() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    path = OUT / "c2_quad_det_funnel.json"
    if not path.is_file():
        return 0.0, ["missing c2_quad_det_funnel.json — run funnel_match3_quad_det.py"]
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner")
    if winner not in ("v12_plain", "v12_sahi_fallback"):
        score -= 3.0
        notes.append(f"winner {winner} not v12")
    variants = {r["variant"]: r for r in data.get("variants") or []}
    base = variants.get("v10_plain", {}).get("totals", {})
    win = variants.get(winner or "", {}).get("totals", {})
    b_r = float(base.get("clear_ball_proxy_R") or 0)
    w_r = float(win.get("clear_ball_proxy_R") or 0)
    if w_r <= b_r:
        score -= 2.0
        notes.append(f"v12 did not beat v10 ({w_r} vs {b_r})")
    p9 = (win.get("per_stem") or {}).get("quad_P9_t00655.3s") or {}
    if float(p9.get("clear_ball_proxy_R") or 0) < 0.80:
        score -= 1.0
        notes.append("P9 t655 proxy R still below 0.80")
    if not notes:
        notes.append(f"C2 winner={winner} quad clear_R {b_r:.3f}→{w_r:.3f}")
    return clamp(score), notes


def score_product_goals() -> tuple[float, list[str]]:
    """Product P_emit + clear_ball_R evidence ≥ 9/10.

    Prefer F0 fuse A/B (`f_post_ab.json` winner with hold) — that is the product path.
    Fall back to m1 carry/ticks when A/B is missing.
    """
    notes = []
    score = 10.0
    ab_path = OUT / "f_post_ab.json"
    m1_path = OUT / "m1_provisional.json"
    if not m1_path.is_file() and not ab_path.is_file():
        return 0.0, ["missing m1_provisional.json / f_post_ab.json"]

    strips = {}
    if ab_path.is_file():
        ab = json.loads(ab_path.read_text(encoding="utf-8"))
        winner = ab.get("winner") or "F1+F2+F0+F3"
        for pack, block in (ab.get("strips") or {}).items():
            var = (block.get("variants") or {}).get(winner) or {}
            strips[pack] = {
                "P_emit": var.get("P_emit"),
                "clear_ball_R": var.get("clear_ball_R"),
                "poc_pass_P_emit": var.get("poc_pass_P_emit"),
                "source": f"f_post:{winner}",
            }
    if not strips and m1_path.is_file():
        data = json.loads(m1_path.read_text(encoding="utf-8"))
        raw = data.get("strips") or {}
        if not raw and data.get("strip"):
            raw = {"legacy": data["strip"]}
        for pack, row in raw.items():
            modes = row.get("modes") or {}
            ticks_r = (modes.get("detect_ticks_only") or {}).get("clear_ball_R")
            carry_r = (modes.get("carry_neighbor_tick") or {}).get("clear_ball_R")
            product_r = carry_r if carry_r is not None else ticks_r
            strips[pack] = {
                "P_emit": row.get("P_emit"),
                "clear_ball_R": product_r,
                "poc_pass_P_emit": row.get("poc_pass_P_emit"),
                "source": "m1_carry_or_ticks",
            }

    if len(strips) < 2:
        score -= 1.5
        notes.append(f"need ≥2 strips for goals≥9 (have {len(strips)})")
    r90_ok = 0
    for pack, row in strips.items():
        p = row.get("P_emit")
        r = row.get("clear_ball_R")
        if p is None or float(p) < 0.90:
            score -= 3.0
            notes.append(f"{pack} P_emit fail ({p})")
        if r is None or float(r) < 0.80:
            score -= 2.5
            notes.append(f"{pack} product clear_R fail ({r})")
        if r is not None and float(r) >= 0.90:
            r90_ok += 1
    if r90_ok < 1:
        score -= 1.0
        notes.append("no strip with product clear_R ≥ 0.90")
    text = (ROOT / "src/mapping/match3_xy.py").read_text(encoding="utf-8")
    if "HOLD_MAX_GAP = 4" not in text and "HOLD_MAX_GAP=4" not in text:
        score -= 1.0
        notes.append("HOLD_MAX_GAP not locked at 4")
    gallery = ROOT / "reports/eval_match3/pitchmap_gallery/manifest.json"
    if not gallery.is_file():
        score -= 1.0
        notes.append("random pitchmap gallery missing")
    if not notes:
        notes.append(f"goals ok on {len(strips)} strips (F0 clear_R, hold=4)")
    return clamp(score), notes


def score_product_post() -> tuple[float, list[str]]:
    """Post maturity ≥ 9/10: F0–F3 shipped, gallery reviewed, P gate held."""
    notes = []
    score = 10.0
    text = (ROOT / "src/mapping/match3_xy.py").read_text(encoding="utf-8")
    for needle, label in (
        ("fuse_balls_with_hold", "F0"),
        ("soft_dual_fallback", "F1"),
        ("solo_max_conf", "F2"),
        ("prune_ghost_maps", "F3"),
    ):
        if needle not in text:
            score -= 2.0
            notes.append(f"missing {label}")
    ab = OUT / "f_post_ab.json"
    if not ab.is_file():
        score -= 3.0
        notes.append("missing f_post_ab")
    else:
        data = json.loads(ab.read_text(encoding="utf-8"))
        winner = data.get("winner") or ""
        if "F0" not in winner:
            score -= 1.5
            notes.append("winner should include F0 hold")
        if "F3" not in winner and "prune_ghost_maps" in text:
            # F3 may tie; still require ghost prune on product path
            pass
        if not winner:
            score -= 3.0
            notes.append("no A/B winner")
    man = ROOT / "reports/eval_match3/pitchmap_gallery/manifest.json"
    if man.is_file():
        entries = json.loads(man.read_text(encoding="utf-8"))
        total_emit = sum(int(e.get("n_emit") or 0) for e in entries)
        if total_emit < 150:
            score -= 1.0
            notes.append(f"random gallery emit {total_emit} < 150")
    else:
        score -= 1.5
        notes.append("random gallery missing")
    if "EMIT_CONF = 0.80" not in text and "EMIT_CONF=0.80" not in text:
        score -= 2.0
        notes.append("EMIT_CONF not 0.80")
    if not notes:
        notes.append("post F0–F3 + gallery ok")
    return clamp(score), notes


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores, notes = {}, {}
    scores["plan"], notes["plan"] = score_plan()
    scores["t1_wire"], notes["t1_wire"] = score_t1_wire()
    scores["l1_calib"], notes["l1_calib"] = score_l1_calibs()
    scores["l2_overlap"], notes["l2_overlap"] = score_l2_overlap()
    scores["h1_support"], notes["h1_support"] = score_h1_support()
    scores["c1_fn"], notes["c1_fn"] = score_c1_fn_audit()
    scores["c2_det"], notes["c2_det"] = score_c2_quad_det()
    scores["c3_p6"], notes["c3_p6"] = score_c3_p6_hull()
    scores["f_post"], notes["f_post"] = score_f_post()
    scores["product_goals"], notes["product_goals"] = score_product_goals()
    scores["product_post"], notes["product_post"] = score_product_post()
    fails = [f"{k}={scores[k]} {notes[k]}" for k in scores if scores[k] < PASS]
    summary = {"scores": scores, "notes": notes, "pass": PASS, "fails": fails}
    (OUT / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(scores)
    if fails:
        print("BELOW 9")
        for f in fails:
            print(" ", f)
        return 1
    print("all subgoals >= 9/10")
    print(f"wrote {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
