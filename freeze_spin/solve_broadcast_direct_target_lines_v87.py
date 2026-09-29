from __future__ import annotations

"""v87 direct Broadcast pinhole camera from floor paint + elevated target lines.

The fitted 3D evidence is deliberately limited to three source-visible white target
stripe line families. Orange rim pixels are excluded and reserved for the next
independent noncoplanar validation stage. This file is discovery-only even when
all internal stability gates pass.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

FT = 30.48
IN = 2.54
EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"
TARGET_HALF_W = 11.0 * IN
TARGET_BOTTOM_Z = 10.0 * FT + 1.0 * IN
TARGET_TOP_Z = 10.0 * FT + 17.0 * IN
TARGET_KEYS = ("target_top", "target_left", "target_right")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def camera_matrix(p: np.ndarray) -> np.ndarray:
    f = float(np.exp(p[6]))
    return np.array([[f, 0.0, p[7]], [0.0, f, p[8]], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation(p: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(p[:3], dtype=np.float64).reshape(3, 1))[0]


def camera_center(p: np.ndarray) -> np.ndarray:
    R = rotation(p)
    return -R.T @ np.asarray(p[3:6], dtype=np.float64)


def project3(p: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64)
    R = rotation(p)
    Q = (R @ P.T).T + np.asarray(p[3:6], dtype=np.float64)
    f = float(np.exp(p[6]))
    uv = np.column_stack([f * Q[:, 0] / Q[:, 2] + p[7], f * Q[:, 1] / Q[:, 2] + p[8]])
    return uv, Q


def floor_homography(p: np.ndarray) -> np.ndarray:
    R = rotation(p)
    return camera_matrix(p) @ np.column_stack([R[:, 0], R[:, 1], np.asarray(p[3:6], dtype=np.float64)])


def world_target_lines() -> dict[str, np.ndarray]:
    return {
        "target_top": np.array([[0.0, -TARGET_HALF_W, TARGET_TOP_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]], dtype=np.float64),
        "target_left": np.array([[0.0, -TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, -TARGET_HALF_W, TARGET_TOP_Z]], dtype=np.float64),
        "target_right": np.array([[0.0, TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]], dtype=np.float64),
    }


def signed_line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return (np.asarray(points, dtype=np.float64) - np.asarray(a, dtype=np.float64)) @ n


def target_residual(p: np.ndarray, target_obs: dict[str, np.ndarray]) -> np.ndarray:
    out = []
    for key, P in world_target_lines().items():
        uv, _ = project3(p, P)
        out.append(signed_line_distance(target_obs[key], uv[0], uv[1]))
    return np.concatenate(out)


def target_metrics(p: np.ndarray, target_obs: dict[str, np.ndarray]) -> dict:
    out = {}
    for key, P in world_target_lines().items():
        uv, _ = project3(p, P)
        d = np.abs(signed_line_distance(target_obs[key], uv[0], uv[1]))
        out[key] = {
            "count": int(len(d)),
            "median_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(np.max(d)),
        }
    return out


def data_residual(p: np.ndarray, floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray]) -> np.ndarray:
    H = floor_homography(p)
    rows = [v44.signed_pixel_residual(H, key, floor_train[key]) for key in v44.GROUPS]
    rows.append(target_residual(p, target_obs))
    # Enforce a forward-facing physical solution only; no focal/PP/camera-location prior.
    check = np.vstack([
        np.array([[v44.FT_X_CM, -v44.PAINT_HALF_CM, 0.0], [v44.FT_X_CM, v44.PAINT_HALF_CM, 0.0]], dtype=np.float64),
        np.vstack(list(world_target_lines().values())),
    ])
    _, q = project3(p, check)
    rows.append(np.minimum(q[:, 2] - 20.0, 0.0) / 5.0)
    return np.concatenate(rows)


def solve_warm(p0: np.ndarray, floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray], max_nfev: int = 5000) -> np.ndarray:
    lo = np.r_[[-10.0] * 3, [-20000.0] * 3, math.log(150.0), -2000.0, -2000.0]
    hi = np.r_[[10.0] * 3, [20000.0] * 3, math.log(8000.0), 3000.0, 3000.0]
    opt = least_squares(
        lambda p: data_residual(p, floor_train, target_obs),
        np.asarray(p0, dtype=np.float64),
        bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
    )
    return np.asarray(opt.x, dtype=np.float64)


def pnp_starts(spec: dict) -> list[np.ndarray]:
    c = spec["seed_only_corner_estimates_px"]
    uv_target = np.asarray([c["top_left"], c["top_right"], c["bottom_right"], c["bottom_left"]], dtype=np.float64)
    obj_target = np.asarray([
        [0.0, -TARGET_HALF_W, TARGET_TOP_Z],
        [0.0, TARGET_HALF_W, TARGET_TOP_Z],
        [0.0, TARGET_HALF_W, TARGET_BOTTOM_Z],
        [0.0, -TARGET_HALF_W, TARGET_BOTTOM_Z],
    ], dtype=np.float64)
    obj_floor = np.asarray([
        [v44.FT_X_CM, -v44.PAINT_HALF_CM, 0.0],
        [v44.FT_X_CM, v44.PAINT_HALF_CM, 0.0],
    ], dtype=np.float64)
    # Seed pixels only: excluded from residual and all metrics.
    uv_floor = np.asarray([[267.0, 297.0], [351.0, 390.0]], dtype=np.float64)
    obj = np.vstack([obj_target, obj_floor])
    uv = np.vstack([uv_target, uv_floor])

    starts = []
    for f0 in (450.0, 800.0, 1200.0, 2000.0, 3000.0):
        for pp in ((480.0, 270.0), (800.0, 540.0), (960.0, 540.0)):
            K = np.array([[f0, 0.0, pp[0]], [0.0, f0, pp[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
            ok, rv, tv = cv2.solvePnP(obj, uv, K, None, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                starts.append(np.r_[rv.ravel(), tv.ravel(), math.log(f0), pp])
    return starts


def dense_projection(p: np.ndarray) -> dict[str, np.ndarray]:
    dense = v44.dense_features()
    out = {f"floor_{k}": project3(p, np.column_stack([xy, np.zeros(len(xy))]))[0] for k, xy in dense.items()}
    t = np.linspace(0.0, 1.0, 401)
    for key, P in world_target_lines().items():
        pts = P[0][None, :] * (1.0 - t[:, None]) + P[1][None, :] * t[:, None]
        out[key] = project3(p, pts)[0]
    return out


def projection_shift(pa: np.ndarray, pb: np.ndarray) -> dict:
    a, b = dense_projection(pa), dense_projection(pb)
    out = {}
    for key in a:
        d = np.linalg.norm(a[key] - b[key], axis=1)
        out[key] = {"p95_px": float(np.percentile(d, 95)), "max_px": float(np.max(d))}
    return out


def max_p95(summary: dict) -> float:
    return float(max(v["p95_px"] for v in summary.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--target-lines", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual = sha256(args.frame)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Immutable Broadcast Frame C SHA changed: {actual}")

    v85.patch_line_aware_geometry()
    floor_spec = json.loads(args.floor.read_text())
    target_spec = json.loads(args.target_lines.read_text())
    if floor_spec.get("camera_label") != "Broadcast" or target_spec.get("camera_label") != "Broadcast":
        raise RuntimeError("Broadcast provenance changed")
    floor_train, floor_held = v44.split_groups(floor_spec["observations_px"], floor_spec["held_out_indices"])
    target_obs = {k: np.asarray(target_spec["observed_line_samples_px"][k], dtype=np.float64) for k in TARGET_KEYS}

    roots = []
    for start in pnp_starts(target_spec):
        try:
            p = solve_warm(start, floor_train, target_obs, max_nfev=7000)
        except Exception:
            continue
        r = data_residual(p, floor_train, target_obs)
        score = float(np.median(np.abs(r[:-8]))) if len(r) > 8 else float(np.median(np.abs(r)))
        C = camera_center(p)
        _, q = project3(p, np.vstack(list(world_target_lines().values())))
        if np.all(np.isfinite(p)) and np.all(q[:, 2] > 20.0) and np.isfinite(C).all():
            roots.append({"p": p, "score": score, "center_cm": C})
    if not roots:
        raise RuntimeError("No physically forward Broadcast v87 roots")
    roots.sort(key=lambda r: r["score"])
    best = roots[0]
    pb = best["p"]
    Cb = best["center_cm"]

    floor_H = floor_homography(pb)
    floor_held_metrics = v44.pixel_metrics(floor_H, floor_held, v44.dense_features())
    floor_held_max = float(max(v["p95_px"] for v in floor_held_metrics.values()))
    target_m = target_metrics(pb, target_obs)
    target_max = float(max(v["p95_px"] for v in target_m.values()))

    competitive = []
    best_score = best["score"]
    for row in roots:
        if row["score"] > best_score + 0.25:
            continue
        sh = projection_shift(pb, row["p"])
        competitive.append({
            "score": row["score"],
            "max_projection_p95_shift_px": max_p95(sh),
            "projection_shift": sh,
            "center_shift_cm": float(np.linalg.norm(row["center_cm"] - Cb)),
            "focal_px": float(np.exp(row["p"][6])),
            "principal_point_px": row["p"][7:9].tolist(),
            "center_cm": row["center_cm"].tolist(),
        })
    competitive_max = float(max(r["max_projection_p95_shift_px"] for r in competitive))
    competitive_center_max = float(max(r["center_shift_cm"] for r in competitive))

    rng = np.random.default_rng(870903)
    pert = []
    for trial in range(args.perturbation_trials):
        fg = {k: floor_train[k] + rng.uniform(-0.5, 0.5, size=floor_train[k].shape) for k in v44.GROUPS}
        tg = {k: target_obs[k] + rng.uniform(-0.5, 0.5, size=target_obs[k].shape) for k in TARGET_KEYS}
        try:
            p = solve_warm(pb, fg, tg, max_nfev=4500)
        except Exception:
            pert.append({"trial": trial, "failed": True})
            continue
        C = camera_center(p)
        sh = projection_shift(pb, p)
        H = floor_homography(p)
        hm = v44.pixel_metrics(H, floor_held, v44.dense_features())
        fm = float(max(v["p95_px"] for v in hm.values()))
        tm = target_metrics(p, target_obs)
        tmax = float(max(v["p95_px"] for v in tm.values()))
        pert.append({
            "trial": trial,
            "failed": False,
            "max_projection_p95_shift_px": max_p95(sh),
            "center_shift_cm": float(np.linalg.norm(C - Cb)),
            "heldout_floor_max_p95_px": fm,
            "target_line_max_p95_px": tmax,
            "focal_fraction_shift": float(abs(np.exp(p[6]) - np.exp(pb[6])) / np.exp(pb[6])),
            "principal_point_shift_px": float(np.linalg.norm(p[7:9] - pb[7:9])),
        })
    good = [r for r in pert if not r.get("failed")]
    failures = len(pert) - len(good)
    max_pert_proj = float(max((r["max_projection_p95_shift_px"] for r in good), default=float("inf")))
    max_pert_center = float(max((r["center_shift_cm"] for r in good), default=float("inf")))
    max_pert_floor = float(max((r["heldout_floor_max_p95_px"] for r in good), default=float("inf")))
    max_pert_target = float(max((r["target_line_max_p95_px"] for r in good), default=float("inf")))

    gates = {
        "nominal_heldout_floor_p95_at_most_2px": floor_held_max <= 2.0,
        "nominal_target_line_p95_at_most_1_5px": target_max <= 1.5,
        "at_least_3_competitive_roots": len(competitive) >= 3,
        "competitive_projection_p95_at_most_0_5px": competitive_max <= 0.5,
        "competitive_center_spread_at_most_75cm": competitive_center_max <= 75.0,
        "all_64_half_pixel_trials_converged": failures == 0 and len(good) == args.perturbation_trials,
        "half_pixel_projection_p95_at_most_2px": max_pert_proj <= 2.0,
        "half_pixel_center_shift_at_most_75cm": max_pert_center <= 75.0,
        "half_pixel_heldout_floor_p95_at_most_2_5px": max_pert_floor <= 2.5,
        "half_pixel_target_line_p95_at_most_2px": max_pert_target <= 2.0,
        "pinhole_only_no_distortion_parameters": True,
    }
    internal_pass = bool(all(gates.values()))

    report = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_BROADCAST_DIRECT_TARGET_LINES_V87",
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Broadcast",
        "immutable_frame_sha256": actual,
        "method": "direct undistorted pinhole camera from v86 line-aware floor paint plus source-visible centreline of three elevated 2-inch white target-stripe line families",
        "rim_policy": "orange rim excluded from fit and reserved as independent held-out 3D evidence for the next stage",
        "camera_candidate": {
            "rvec": pb[:3].tolist(),
            "tvec_cm": pb[3:6].tolist(),
            "focal_px": float(np.exp(pb[6])),
            "principal_point_px": pb[7:9].tolist(),
            "physical_center_cm": Cb.tolist(),
            "physical_center_ft": (Cb / FT).tolist(),
        },
        "nominal": {
            "heldout_floor_max_p95_px": floor_held_max,
            "heldout_floor": floor_held_metrics,
            "target_line_max_p95_px": target_max,
            "target_lines": target_m,
        },
        "multistart": {
            "total_forward_roots": len(roots),
            "competitive_root_count": len(competitive),
            "max_competitive_projection_p95_shift_px": competitive_max,
            "max_competitive_center_shift_cm": competitive_center_max,
            "roots": competitive,
        },
        "half_pixel_perturbation": {
            "trials": len(pert),
            "failed_trials": failures,
            "max_projection_p95_shift_px": max_pert_proj,
            "max_center_shift_cm": max_pert_center,
            "max_heldout_floor_p95_px": max_pert_floor,
            "max_target_line_p95_px": max_pert_target,
            "all_trials": pert,
        },
        "gates": gates,
        "candidate_passes_internal_gates": internal_pass,
        "permissions": {
            "broadcast_floor_homography_allowed": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_gate": "If internal gates pass, validate the regulation orange rim as entirely independent held-out noncoplanar evidence before promoting any physical camera permission.",
    }
    path = args.out / "broadcast_direct_target_lines_v87.json"
    path.write_text(json.dumps(v44.json_safe(report), indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "internal_pass": internal_pass,
        "heldout_floor_max_p95_px": floor_held_max,
        "target_line_max_p95_px": target_max,
        "focal_px": report["camera_candidate"]["focal_px"],
        "principal_point_px": report["camera_candidate"]["principal_point_px"],
        "physical_center_ft": report["camera_candidate"]["physical_center_ft"],
        "competitive_roots": len(competitive),
        "competitive_projection_max_p95_px": competitive_max,
        "half_pixel_projection_max_p95_px": max_pert_proj,
        "half_pixel_center_max_cm": max_pert_center,
        "half_pixel_floor_max_p95_px": max_pert_floor,
        "half_pixel_target_max_p95_px": max_pert_target,
        "failed_trials": failures,
    }, indent=2))


if __name__ == "__main__":
    main()
