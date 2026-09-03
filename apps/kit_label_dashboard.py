"""Streamlit — label pre-game team kits and save team_centroids.json.

Usage: streamlit run apps/kit_label_dashboard.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.raw_cam_id import cam_id_from_raw_name, load_match_raw  # noqa: E402
from src.mapping.match3_xy import load_calib_for_video  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    KIT_MODE_AUTO,
    KIT_MODE_MATCH3,
    backup_kit_ref,
    centroids_from_labeled,
    jersey_feature,
    kit_feat_preview_bgr,
    kit_samples_bank_path,
    load_centroids,
    load_kit_ref_meta,
    load_kit_samples_bank,
    merge_kit_sample_rows,
    save_kit_ref,
    save_kit_samples_bank,
    samples_to_bank_rows,
    torso_crop,
)
from src.review.cam_mosaic import undistort_bgr  # noqa: E402
from src.review.frame_sync import draw_det_boxes, guess_video_for_run, keep_top1_ball  # noqa: E402

DEFAULT_OUT = ROOT / "data/output/match_4_5min"
DEFAULT_RAW = ROOT / "data/raw/Match 3"
FISHEYE = frozenset({"P7", "P8", "P9", "P10"})
PLAYER_CKPT = ROOT / "models/people_after_100_epochs.pth"
BALL_CKPT = ROOT / "models/v13_residual_snaps/post_train/checkpoint.pth"


def list_raw_videos(raw_dir: Path) -> dict[str, Path]:
    if not raw_dir.is_dir():
        return {}
    try:
        return load_match_raw(raw_dir)
    except (FileNotFoundError, ValueError):
        out = {}
        for p in sorted(raw_dir.glob("*.mp4")):
            if p.name.startswith("._"):
                continue
            try:
                out[cam_id_from_raw_name(p.name)] = p
            except ValueError:
                continue
        return out


def list_batch_runs(out_root: Path) -> list[str]:
    if not out_root.is_dir():
        return []
    return sorted(p.name for p in out_root.iterdir() if p.is_dir())


def read_video_frame(video_path: Path, frame_id: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def prep_frame(frame_bgr: np.ndarray, cam: str, video_path: Path, defish: bool):
    calib = load_calib_for_video(str(video_path))
    if defish and calib and cam in FISHEYE:
        return undistort_bgr(frame_bgr, calib), calib
    return frame_bgr, calib


def player_dets(frame_bgr: np.ndarray, det_thr: float):
    from src.review.app import _load_verify_detector

    det = _load_verify_detector(
        str(PLAYER_CKPT), str(BALL_CKPT), float(det_thr), nms_ver="v4"
    )
    return keep_top1_ball(det.detect(frame_bgr))


def draw_numbered_players(
    frame_bgr: np.ndarray,
    dets: list,
    assignments: dict[int, int | None] | None = None,
) -> np.ndarray:
    vis = draw_det_boxes(frame_bgr, dets)
    idx = 0
    for det in dets:
        if getattr(det, "class_name", "") == "ball" or int(det.class_id) == 1:
            continue
        x, y, w, h = [int(v) for v in det.bbox]
        idx += 1
        team = (assignments or {}).get(idx - 1)
        box_color = (255, 80, 80) if team == 0 else (80, 80, 255) if team == 1 else (0, 220, 0)
        cv2.rectangle(vis, (x, y), (x + max(1, w), y + max(1, h)), box_color, 2)
        label = f"#{idx}"
        ty = max(28, y - 10)
        cv2.rectangle(vis, (x, ty - 22), (x + 36, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return vis


def player_only(dets: list) -> list:
    return [
        d
        for d in dets
        if getattr(d, "class_name", "") != "ball" and int(getattr(d, "class_id", -1)) != 1
    ]


def crop_player_rgb(frame_bgr: np.ndarray, det, cam: str) -> np.ndarray | None:
    wh = (frame_bgr.shape[1], frame_bgr.shape[0])
    crop = torso_crop(frame_bgr, det.bbox, cam=cam, frame_wh=wh)
    if crop is None:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def init_session():
    st.session_state.setdefault("kit_samples", [])
    st.session_state.setdefault("kit_assignments", {})
    st.session_state.setdefault("kit_team_names", ("Team 0", "Team 1"))


def slot_key(video_path: Path, frame_id: int, player_idx: int) -> str:
    return f"{video_path.name}:{int(frame_id)}:P{player_idx + 1}"


def player_assignment(video_path: Path, frame_id: int, player_idx: int) -> int | None:
    key = slot_key(video_path, frame_id, player_idx)
    val = st.session_state.kit_assignments.get(key)
    return int(val) if val is not None else None


def assign_player(
    video_path: Path,
    frame_id: int,
    player_idx: int,
    team_id: int,
    feat: np.ndarray,
    crop_rgb: np.ndarray,
    tag: str,
) -> None:
    key = slot_key(video_path, frame_id, player_idx)
    st.session_state.kit_assignments[key] = int(team_id)
    st.session_state.kit_samples = [
        s for s in st.session_state.kit_samples if s.get("slot_key") != key
    ]
    st.session_state.kit_samples.append(
        {
            "team": int(team_id),
            "feat": feat.astype(np.float32).tolist(),
            "crop_rgb": crop_rgb.tolist(),
            "frame_id": int(frame_id),
            "tag": tag,
            "slot_key": key,
        }
    )


def samples_by_team() -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for s in st.session_state.kit_samples:
        out[int(s["team"])].append(np.asarray(s["feat"], dtype=np.float32))
    return out


def bank_rows_to_session(rows: list[dict], *, merge: bool) -> None:
    """Load bank rows into session; synthesize preview swatches from features."""
    incoming = []
    for s in rows:
        feat = np.asarray(s["feat"], dtype=np.float32)
        preview = kit_feat_preview_bgr(feat)
        crop_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        incoming.append(
            {
                "team": int(s["team"]),
                "feat": feat.tolist(),
                "crop_rgb": crop_rgb.tolist(),
                "frame_id": int(s.get("frame_id") or 0),
                "tag": str(s.get("tag") or "bank"),
                "slot_key": str(s.get("slot_key") or ""),
            }
        )
    if merge:
        st.session_state.kit_samples = merge_kit_sample_rows(
            st.session_state.kit_samples, incoming
        )
    else:
        st.session_state.kit_samples = incoming
    st.session_state.kit_assignments = {
        s["slot_key"]: int(s["team"])
        for s in st.session_state.kit_samples
        if s.get("slot_key")
    }


def persist_kit_save(
    save_path: Path,
    out_root: Path,
    cents: np.ndarray,
    radius: float,
    *,
    team_names: tuple[str, str],
    kit_mode: str,
    samples: list[dict],
) -> Path | None:
    """Backup prior file, write centroids + sample bank (+ match-root copies)."""
    backup = backup_kit_ref(save_path)
    n0 = sum(1 for s in samples if int(s["team"]) == 0)
    n1 = sum(1 for s in samples if int(s["team"]) == 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_kit_ref(
        save_path,
        cents,
        radius,
        team_names=team_names,
        kit_mode=kit_mode,
        n_samples=(n0, n1),
    )
    save_kit_samples_bank(
        kit_samples_bank_path(save_path),
        samples,
        meta={"team_names": list(team_names), "kit_mode": kit_mode},
    )
    match_root = out_root / "team_centroids.json"
    if save_path.resolve() != match_root.resolve():
        backup_kit_ref(match_root)
        save_kit_ref(
            match_root,
            cents,
            radius,
            team_names=team_names,
            kit_mode=kit_mode,
            n_samples=(n0, n1),
        )
        save_kit_samples_bank(
            kit_samples_bank_path(match_root),
            samples,
            meta={"team_names": list(team_names), "kit_mode": kit_mode},
        )
    return backup


def merged_samples_for_save(save_path: Path) -> list[dict]:
    """Session samples unioned with on-disk bank (session wins on same slot_key)."""
    prior = load_kit_samples_bank(kit_samples_bank_path(save_path))
    return merge_kit_sample_rows(
        prior, samples_to_bank_rows(st.session_state.kit_samples)
    )


def save_kit_append(
    save_path: Path,
    out_root: Path,
    *,
    team_names: tuple[str, str],
    kit_mode: str,
) -> tuple[Path | None, int, int]:
    """Default save: merge with bank, refit, write. Never drops prior bank rows."""
    merged = merged_samples_for_save(save_path)
    bank_rows_to_session(merged, merge=False)
    fit = fit_centroids()
    if fit is None:
        raise ValueError("Need ≥1 sample per team after merge")
    cents, radius = fit
    backup = persist_kit_save(
        save_path,
        out_root,
        cents,
        radius,
        team_names=team_names,
        kit_mode=kit_mode,
        samples=st.session_state.kit_samples,
    )
    n0, n1 = len(samples_by_team()[0]), len(samples_by_team()[1])
    return backup, n0, n1


def fit_centroids():
    return centroids_from_labeled(samples_by_team())


def save_path_for_run(out_root: Path, run_name: str) -> Path:
    if run_name:
        return out_root / run_name / "team_centroids.json"
    return out_root / "team_centroids.json"


def render_sample_grid(team_id: int, team_name: str):
    rows = [s for s in st.session_state.kit_samples if int(s["team"]) == team_id]
    st.markdown(f"**{team_name}** — {len(rows)} sample(s)")
    if not rows:
        st.caption("No samples yet.")
        return
    cols = st.columns(min(6, len(rows)))
    for i, s in enumerate(rows[-12:]):
        crop = np.asarray(s["crop_rgb"], dtype=np.uint8)
        with cols[i % len(cols)]:
            st.image(crop, caption=f"f{s['frame_id']} {s.get('tag', '')}", use_container_width=True)


def main():
    st.set_page_config(page_title="Kit label", layout="wide")
    init_session()
    st.title("Team kit labeling")
    st.caption(
        "Pick pre-game frames, assign player crops to each kit, then save. "
        "**Save always merges** with `kit_samples_bank.json` (does not wipe prior labels)."
    )

    raw_videos = list_raw_videos(DEFAULT_RAW)
    batch_runs = list_batch_runs(DEFAULT_OUT)

    with st.sidebar:
        st.header("Match setup")
        out_root = Path(
            st.text_input("Batch output root", value=str(DEFAULT_OUT))
        )
        run_name = st.selectbox(
            "Batch run folder (save target)",
            options=[""] + batch_runs,
            index=(batch_runs.index("P10-match4") + 1) if "P10-match4" in batch_runs else 0,
        )
        cam = st.selectbox(
            "Camera",
            options=sorted(raw_videos.keys()) or ["P10"],
            index=sorted(raw_videos.keys()).index("P10") if "P10" in raw_videos else 0,
        )
        video_default = str(raw_videos.get(cam) or guess_video_for_run(run_name or cam, ROOT) or "")
        video_path = Path(st.text_input("Video path", value=video_default))
        team0_name = st.text_input("Team 0 label", value=st.session_state.kit_team_names[0])
        team1_name = st.text_input("Team 1 label", value=st.session_state.kit_team_names[1])
        st.session_state.kit_team_names = (team0_name, team1_name)
        kit_mode = st.selectbox("Kit mode", [KIT_MODE_AUTO, KIT_MODE_MATCH3], index=0)
        defish = st.checkbox("Defish P7–P10", value=True)
        frame_id = st.number_input("Frame index", min_value=0, value=0, step=30)
        det_thr = st.slider("Player det threshold", 0.05, 0.6, 0.25, 0.05)
        detect = st.button("Detect players on frame", type="primary")
        st.divider()
        st.subheader("Reset")
        reset_disk = st.checkbox(
            "Also delete sample bank + centroids on disk",
            value=False,
            help="Dangerous. Session-only reset keeps files; Save can rebuild from labels.",
        )
        if st.button("Reset labels", type="secondary"):
            st.session_state.kit_samples = []
            st.session_state.kit_assignments = {}
            st.session_state.pop("kit_loaded_centroids", None)
            st.session_state.pop("kit_dets", None)
            st.session_state.pop("kit_det_key", None)
            # Prevent auto-load from immediately restoring the bank.
            st.session_state[f"kit_autoload::{save_path_for_run(out_root, run_name)}"] = True
            if reset_disk:
                sp = save_path_for_run(out_root, run_name)
                backup_kit_ref(sp)
                for p in (
                    sp,
                    kit_samples_bank_path(sp),
                    out_root / "team_centroids.json",
                    kit_samples_bank_path(out_root / "team_centroids.json"),
                ):
                    if p.is_file():
                        p.unlink()
                st.warning("Session cleared and on-disk kit files removed (backup if prior existed).")
            else:
                st.info("Session labels cleared (disk bank untouched).")
            st.rerun()
        save_path = save_path_for_run(out_root, run_name)
        st.caption(f"Save → `{save_path}` (append/merge)")
        bank_path = kit_samples_bank_path(save_path)
        # Auto-load bank once per session when UI starts empty.
        autoload_key = f"kit_autoload::{save_path}"
        if (
            not st.session_state.kit_samples
            and bank_path.is_file()
            and not st.session_state.get(autoload_key)
        ):
            rows = load_kit_samples_bank(bank_path)
            if rows:
                bank_rows_to_session(rows, merge=False)
                st.session_state[autoload_key] = True
                st.info(f"Loaded {len(rows)} samples from existing bank.")
                st.rerun()
            st.session_state[autoload_key] = True

    if not video_path.is_file():
        st.error(f"Video not found: {video_path}")
        return

    frame_raw = read_video_frame(video_path, int(frame_id))
    if frame_raw is None:
        st.error("Could not read frame.")
        return

    frame_bgr, calib = prep_frame(frame_raw, cam, video_path, defish)
    if calib:
        st.sidebar.caption(f"Calib: {calib.get('camera', '?')}")
    else:
        st.sidebar.caption("No calib — raw pixels")

    cache_key = (str(video_path), int(frame_id), float(det_thr), bool(defish))
    if detect or st.session_state.get("kit_det_key") != cache_key:
        with st.spinner("Running RF-DETR…"):
            dets = player_dets(frame_bgr, det_thr)
            st.session_state.kit_dets = player_only(dets)
            st.session_state.kit_det_key = cache_key
    dets = st.session_state.get("kit_dets") or []

    frame_assignments = {
        i: player_assignment(video_path, int(frame_id), i) for i in range(len(dets))
    }
    vis = draw_numbered_players(frame_bgr, dets, frame_assignments)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_id}", use_container_width=True)

    st.subheader("Assign detected players")
    if not dets:
        st.info("No players detected — try another frame or lower threshold.")
    wh = (frame_bgr.shape[1], frame_bgr.shape[0])
    for i, det in enumerate(dets):
        crop = torso_crop(frame_bgr, det.bbox, cam=cam, frame_wh=wh)
        feat = jersey_feature(crop) if crop is not None else None
        selected = frame_assignments.get(i)
        c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
        with c1:
            if crop is not None:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=80)
            else:
                st.caption("bad crop")
        with c2:
            st.write(f"**Player #{i + 1}**")
            st.caption(f"conf {float(det.confidence):.2f}")
            if selected == 0:
                st.markdown(f":blue-badge[**{team0_name}**]")
            elif selected == 1:
                st.markdown(f":red-badge[**{team1_name}**]")
        with c3:
            btn_type = "primary" if selected == 0 else "secondary"
            label = f"✓ {team0_name}" if selected == 0 else f"→ {team0_name}"
            if st.button(label, key=f"t0_{frame_id}_{i}", type=btn_type):
                if feat is None:
                    st.warning("Jersey feature failed — try another crop.")
                else:
                    assign_player(
                        video_path,
                        int(frame_id),
                        i,
                        0,
                        feat,
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        f"P{i + 1}",
                    )
                    st.rerun()
        with c4:
            btn_type = "primary" if selected == 1 else "secondary"
            label = f"✓ {team1_name}" if selected == 1 else f"→ {team1_name}"
            if st.button(label, key=f"t1_{frame_id}_{i}", type=btn_type):
                if feat is None:
                    st.warning("Jersey feature failed — try another crop.")
                else:
                    assign_player(
                        video_path,
                        int(frame_id),
                        i,
                        1,
                        feat,
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        f"P{i + 1}",
                    )
                    st.rerun()

    st.subheader("Labeled kit samples")
    col0, col1 = st.columns(2)
    with col0:
        render_sample_grid(0, team0_name)
    with col1:
        render_sample_grid(1, team1_name)

    fit = fit_centroids()
    st.subheader("Centroid preview")
    if fit is None:
        n0 = len(samples_by_team()[0])
        n1 = len(samples_by_team()[1])
        st.warning(f"Need ≥1 sample per team (currently {n0} / {n1}).")
    else:
        cents, radius = fit
        sep = float(np.linalg.norm(cents[0, :3] - cents[1, :3]))
        p0, p1 = kit_feat_preview_bgr(cents[0]), kit_feat_preview_bgr(cents[1])
        m1, m2, m3 = st.columns(3)
        m1.image(cv2.cvtColor(p0, cv2.COLOR_BGR2RGB), caption=f"{team0_name} centroid", width=96)
        m2.image(cv2.cvtColor(p1, cv2.COLOR_BGR2RGB), caption=f"{team1_name} centroid", width=96)
        m3.metric("Kit separation", f"{sep:.3f}", help="Higher = easier to tell apart")

        if st.button("Save (merge with bank)", type="primary"):
            try:
                backup, n0, n1 = save_kit_append(
                    save_path,
                    out_root,
                    team_names=(team0_name, team1_name),
                    kit_mode=kit_mode,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved (merged) {save_path} — {n0} / {n1} samples")
                st.info(
                    f"Bank `{kit_samples_bank_path(save_path).name}` + "
                    f"match root `{out_root / 'team_centroids.json'}`"
                )
                if backup:
                    st.caption(f"Previous centroids backed up to `{backup.name}`")
                st.rerun()

        wipe = st.checkbox(
            "Allow replace-all save (wipes bank)",
            value=False,
            help="Only for starting over. Default Save always merges.",
        )
        if wipe and st.button("Replace all samples on disk"):
            backup = persist_kit_save(
                save_path,
                out_root,
                cents,
                radius,
                team_names=(team0_name, team1_name),
                kit_mode=kit_mode,
                samples=st.session_state.kit_samples,
            )
            n0, n1 = len(samples_by_team()[0]), len(samples_by_team()[1])
            st.warning(f"Replaced bank with session only ({n0} / {n1})")
            if backup:
                st.caption(f"Backup: `{backup.name}`")
            st.rerun()

    if save_path.is_file():
        loaded = load_centroids(save_path)
        meta = load_kit_ref_meta(save_path)
        bank_path = kit_samples_bank_path(save_path)
        bank_rows = load_kit_samples_bank(bank_path)
        st.divider()
        st.subheader("Existing file on disk")
        st.json(
            {
                "path": str(save_path),
                "meta": meta,
                "radius": loaded[1] if loaded else None,
                "bank_samples": len(bank_rows),
                "bank_path": str(bank_path) if bank_path.is_file() else None,
            }
        )
        if st.button("Reload bank into session"):
            if not bank_rows:
                st.error(
                    "No kit_samples_bank.json yet — Save once after labeling to create it."
                )
            else:
                bank_rows_to_session(bank_rows, merge=False)
                st.success(f"Loaded {len(bank_rows)} bank samples.")
                st.rerun()
        if st.button("Load centroids only (no samples)"):
            if loaded is None:
                st.error("Could not load centroids.")
            else:
                names = meta.get("team_names") or ["Team 0", "Team 1"]
                st.session_state.kit_team_names = (names[0], names[1])
                st.session_state.kit_loaded_centroids = {
                    "centroids": loaded[0].tolist(),
                    "radius": loaded[1],
                    "meta": meta,
                    "loaded_at": datetime.now(timezone.utc).isoformat(),
                }
                st.info("Centroids loaded — add samples then Save (merge).")



if __name__ == "__main__":
    main()