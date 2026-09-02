"""Match 3 camera projection: P matrix from landmarks, rays, 3D reproject."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.mapping.match3_xy import (
    CALIB_DIR,
    MATCH3_CAMS,
    calib_undistort_params,
    load_calib,
    undistort_px,
)

DEFAULT_FX_SCALE = 1.15
MIN_CAMERA_Z_M = 2.0


def _image_wh(calib: dict) -> tuple[int, int]:
    wh = calib.get("image_wh") or [1920, 1080]
    return int(wh[0]), int(wh[1])


def default_K(calib: dict) -> np.ndarray:
    w, h = _image_wh(calib)
    fx = fy = float(w) * DEFAULT_FX_SCALE
    return np.array(
        [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _dist_coeffs(calib: dict) -> np.ndarray:
    u = calib_undistort_params(calib)
    if u is None:
        return np.zeros(5, dtype=np.float64)
    return np.array([u["k1"], u["k2"], u["p1"], u["p2"], 0.0], dtype=np.float64)


def _landmark_reproj_err_p(calib: dict, proj: dict) -> float:
    rec = {**calib, "projection3d": proj}
    P = np.asarray(proj["P"], dtype=float)
    errs = []
    for ip, pp in zip(calib.get("image_points") or [], calib.get("pitch_points") or []):
        v = P @ np.array([float(pp[0]), float(pp[1]), 0.0, 1.0], dtype=float)
        if abs(v[2]) < 1e-8:
            errs.append(999.0)
        else:
            dx = float(ip[0]) - float(v[0] / v[2])
            dy = float(ip[1]) - float(v[1] / v[2])
            errs.append((dx * dx + dy * dy) ** 0.5)
    return float(max(errs)) if errs else 999.0


def fit_projection_from_homography(calib: dict) -> dict | None:
    """Derive P from inverse H (pitch→image) + focal-length search."""
    H = np.asarray(calib.get("H"), dtype=float)
    if H.shape != (3, 3):
        return None
    try:
        H_pi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    w, h = _image_wh(calib)
    best = None
    best_err = 1e9
    for scale in np.linspace(0.6, 2.5, 40):
        fx = fy = float(w) * float(scale)
        K = np.array(
            [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        try:
            _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_pi, K)
        except cv2.error:
            continue
        for R, t, _n in zip(Rs, Ts, Ns):
            R = np.asarray(R, dtype=float)
            t = np.asarray(t, dtype=float).reshape(3, 1)
            C = (-R.T @ t).reshape(3)
            if float(C[2]) < MIN_CAMERA_Z_M:
                continue
            P = K @ np.hstack([R, t])
            proj = {
                "K": K.tolist(),
                "R": R.tolist(),
                "t": t.reshape(3).tolist(),
                "C": C.tolist(),
                "P": P.tolist(),
            }
            err = _landmark_reproj_err_p(calib, proj)
            if err < best_err:
                best_err = err
                best = proj
    return best


def fit_projection_from_landmarks(calib: dict) -> dict | None:
    """Grid-search solvePnP focal length; fallback homography decompose."""
    imgs = calib.get("image_points") or []
    pitch = calib.get("pitch_points") or []
    if len(imgs) < 4 or len(imgs) != len(pitch):
        return fit_projection_from_homography(calib)
    obj = np.array([[float(p[0]), float(p[1]), 0.0] for p in pitch], dtype=np.float64)
    img = np.array([[float(p[0]), float(p[1])] for p in imgs], dtype=np.float64)
    w, h = _image_wh(calib)
    dist = _dist_coeffs(calib)
    best = None
    best_err = 1e9
    for scale in np.linspace(0.3, 3.0, 55):
        fx = fy = float(w) * float(scale)
        K = np.array(
            [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).reshape(3)
        if float(C[2]) < MIN_CAMERA_Z_M:
            continue
        P = K @ np.hstack([R, tvec.reshape(3, 1)])
        proj = {
            "K": K.tolist(),
            "R": R.tolist(),
            "t": tvec.reshape(3).tolist(),
            "C": C.tolist(),
            "P": P.tolist(),
        }
        err = _landmark_reproj_err_p(calib, proj)
        if err < best_err:
            best_err = err
            best = proj
    if best is not None:
        return best
    return fit_projection_from_homography(calib)


def ensure_projection(calib: dict) -> dict | None:
    """Return calib with projection3d attached (compute if missing)."""
    if calib.get("projection3d"):
        return calib
    proj = fit_projection_from_landmarks(calib)
    if proj is None:
        return None
    calib = dict(calib)
    calib["projection3d"] = proj
    return calib


def _P(calib: dict) -> np.ndarray | None:
    proj = calib.get("projection3d")
    if not proj:
        return None
    return np.asarray(proj["P"], dtype=float)


def _C(calib: dict) -> np.ndarray | None:
    proj = calib.get("projection3d")
    if not proj:
        return None
    return np.asarray(proj["C"], dtype=float)


def undistort_foot(calib: dict, u: float, v: float) -> tuple[float, float]:
    w, h = _image_wh(calib)
    params = calib_undistort_params(calib)
    if params is None:
        return float(u), float(v)
    return undistort_px(float(u), float(v), float(w), float(h), params)


def project_pitch_to_pixel(calib: dict, x: float, y: float) -> tuple[float, float] | None:
    """Pitch (x,y) on z=0 → pixel via inverse homography (product calib)."""
    H = np.asarray(calib.get("H"), dtype=float)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    v = H_inv @ np.array([float(x), float(y), 1.0], dtype=float)
    if abs(v[2]) < 1e-8:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def project_world(calib: dict, x: float, y: float, z: float = 0.0) -> tuple[float, float] | None:
    if abs(float(z)) < 1e-6:
        return project_pitch_to_pixel(calib, x, y)
    P = _P(calib)
    if P is None:
        return None
    v = P @ np.array([float(x), float(y), float(z), 1.0], dtype=float)
    if abs(v[2]) < 1e-8:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def pixel_to_ray(
    calib: dict, u: float, v: float, *, undistort: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame ray (origin, unit direction) from pixel."""
    proj = calib.get("projection3d")
    if not proj:
        return None
    if undistort:
        u, v = undistort_foot(calib, u, v)
    K = np.asarray(proj["K"], dtype=float)
    R = np.asarray(proj["R"], dtype=float)
    C = np.asarray(proj["C"], dtype=float)
    pix = np.array([u, v, 1.0], dtype=float)
    d_cam = np.linalg.inv(K) @ pix
    d_world = R.T @ d_cam
    n = np.linalg.norm(d_world)
    if n < 1e-8:
        return None
    return C, d_world / n


def ray_plane_intersect(
    origin: np.ndarray, direction: np.ndarray, z: float = 0.0,
) -> np.ndarray | None:
    if abs(float(direction[2])) < 1e-8:
        return None
    t = (float(z) - float(origin[2])) / float(direction[2])
    if t <= 0.0:
        return None
    return origin + t * direction


def reproj_err_px_3d(
    calib: dict, xyz, det_px, *, undistort: bool = True,
) -> float:
    if abs(float(xyz[2])) < 1e-6:
        rp = project_pitch_to_pixel(calib, float(xyz[0]), float(xyz[1]))
    else:
        P = _P(calib)
        if P is None:
            return float("inf")
        v = P @ np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=float)
        if abs(v[2]) < 1e-8:
            return float("inf")
        rp = float(v[0] / v[2]), float(v[1] / v[2])
    if rp is None or det_px is None:
        return float("inf")
    du, dv = float(det_px[0]), float(det_px[1])
    if undistort:
        du, dv = undistort_foot(calib, du, dv)
    dx = du - float(rp[0])
    dy = dv - float(rp[1])
    return float((dx * dx + dy * dy) ** 0.5)


def write_projection_to_calib(cam: str, dry_run: bool = False) -> dict:
    path = CALIB_DIR / f"{cam}_manual.json"
    if not path.is_file():
        return {"cam": cam, "ok": False, "reason": "missing calib"}
    rec = json.loads(path.read_text(encoding="utf-8"))
    H = rec.get("homography") or rec.get("H")
    if H is None:
        return {"cam": cam, "ok": False, "reason": "missing H"}
    rec["H"] = np.asarray(H, dtype=float)
    rec["camera"] = cam
    proj = fit_projection_from_landmarks(rec)
    if proj is None:
        return {"cam": cam, "ok": False, "reason": "solvePnP failed"}
    errs = []
    for ip, pp in zip(rec.get("image_points") or [], rec.get("pitch_points") or []):
        rp = project_pitch_to_pixel(rec, float(pp[0]), float(pp[1]))
        if rp is None:
            errs.append(999.0)
        else:
            dx = float(ip[0]) - rp[0]
            dy = float(ip[1]) - rp[1]
            errs.append((dx * dx + dy * dy) ** 0.5)
    out_path = path
    if not dry_run:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        payload["projection3d"] = proj
        payload["projection3d_max_reproj_px"] = float(max(errs)) if errs else None
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "cam": cam,
        "ok": True,
        "max_reproj_px": float(max(errs)) if errs else None,
        "C_z": float(proj["C"][2]),
    }


def enrich_all_calibs(cams: list[str] | None = None) -> list[dict]:
    cams = cams or list(MATCH3_CAMS)
    return [write_projection_to_calib(c) for c in cams]
