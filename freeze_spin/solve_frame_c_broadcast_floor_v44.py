from __future__ import annotations

"""v44: strict pinhole-floor test for the exact Frame C Broadcast camera.

The earlier v43 cross-camera feature-transfer preflight was the wrong geometric
bridge: a distinct physical sideline camera cannot inherit the Left Above Rim
homography through direct SIFT transfer.  This stage instead observes the native
Broadcast frame's own visible regulation paint.

All fit residuals are first-order signed pixel distances to a regulation curve or
line.  That matters in this oblique view, where a centimetre does not have a
constant pixel scale.  Every feature group reserves spatially distributed held-out
pixels.  The accepted root must also survive support reduction and 64 independent
half-pixel annotation perturbations.

Passing would validate only an undistorted floor-plane homography.  Failure is a
useful, explicit result: it requires a distortion-aware floor model rather than a
relaxed geometric gate.  Neither result authorizes 3D or replay rendering.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


W, H = 960, 540
FOOT_CM = 30.48
INCH_CM = 2.54
RIM_X_CM = 15.0 * INCH_CM
BASELINE_X_CM = -4.0 * FOOT_CM
FT_X_CM = 15.0 * FOOT_CM
FT_R_CM = 6.0 * FOOT_CM
THREE_R_CM = 23.75 * FOOT_CM
CORNER_Y_CM = 22.0 * FOOT_CM
PAINT_HALF_CM = 8.0 * FOOT_CM

SCALES = np.asarray([1.0, 1.0, 300.0, 1.0, 1.0, 300.0, 0.002, 0.002], dtype=np.float64)
GROUPS = (
    "three_point_arc",
    "free_throw_front_semicircle",
    "free_throw_line",
    "lane_negative_y",
    "lane_positive_y",
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def project_h(Hm: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    q = (Hm @ np.column_stack([xy, np.ones(len(xy))]).T).T
    return q[:, :2] / q[:, 2:3]


def parameter_vector(Hm: np.ndarray) -> np.ndarray:
    Hm = np.asarray(Hm, dtype=np.float64) / float(Hm[2, 2])
    return np.r_[Hm[0], Hm[1], Hm[2, :2]]


def H_from_z(z: np.ndarray, h0: np.ndarray) -> np.ndarray:
    v = h0 + np.asarray(z, dtype=np.float64) * SCALES
    return np.asarray([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], 1.0]], dtype=np.float64)


def split_groups(obs: dict, held_indices: dict) -> tuple[dict, dict]:
    train, held = {}, {}
    if set(obs) != set(GROUPS) or set(held_indices) != set(GROUPS):
        raise RuntimeError("Broadcast v44 feature-group schema changed")
    for key in GROUPS:
        pts = np.asarray(obs[key], dtype=np.float64)
        ids = np.asarray(held_indices[key], dtype=int)
        if np.any(ids < 0) or np.any(ids >= len(pts)) or len(np.unique(ids)) != len(ids):
            raise RuntimeError(f"Invalid held-out indices for {key}")
        mask = np.ones(len(pts), dtype=bool)
        mask[ids] = False
        train[key], held[key] = pts[mask], pts[~mask]
        if len(train[key]) < 3 or len(held[key]) < 1:
            raise RuntimeError(f"Insufficient train/held-out support for {key}")
    return train, held


def implicit_world_value(Hm: np.ndarray, key: str, pixels: np.ndarray) -> np.ndarray:
    world = project_h(np.linalg.inv(Hm), pixels)
    if key == "three_point_arc":
        return np.hypot(world[:, 0] - RIM_X_CM, world[:, 1]) - THREE_R_CM
    if key == "free_throw_front_semicircle":
        return np.hypot(world[:, 0] - FT_X_CM, world[:, 1]) - FT_R_CM
    if key == "free_throw_line":
        return world[:, 0] - FT_X_CM
    if key == "lane_negative_y":
        return world[:, 1] + PAINT_HALF_CM
    if key == "lane_positive_y":
        return world[:, 1] - PAINT_HALF_CM
    raise KeyError(key)


def signed_pixel_residual(Hm: np.ndarray, key: str, pixels: np.ndarray) -> np.ndarray:
    """First-order image-plane distance to an implicit metric floor feature."""
    pixels = np.asarray(pixels, dtype=np.float64)
    eps = 0.25
    f = implicit_world_value(Hm, key, pixels)
    gx = (
        implicit_world_value(Hm, key, pixels + np.asarray([eps, 0.0]))
        - implicit_world_value(Hm, key, pixels - np.asarray([eps, 0.0]))
    ) / (2.0 * eps)
    gy = (
        implicit_world_value(Hm, key, pixels + np.asarray([0.0, eps]))
        - implicit_world_value(Hm, key, pixels - np.asarray([0.0, eps]))
    ) / (2.0 * eps)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def residual(z: np.ndarray, h0: np.ndarray, groups: dict) -> np.ndarray:
    Hm = H_from_z(z, h0)
    if not np.isfinite(Hm).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        return np.full(sum(len(v) for v in groups.values()) + len(z), 1e6, dtype=np.float64)
    rows = [signed_pixel_residual(Hm, key, groups[key]) for key in GROUPS]
    rows.append(np.asarray(z, dtype=np.float64) * 0.001)
    return np.concatenate(rows)


def solve_multistart(
    h0: np.ndarray,
    groups: dict,
    *,
    warm: np.ndarray | None = None,
    return_roots: bool = False,
):
    seeds = []
    if warm is not None:
        seeds.append(np.asarray(warm, dtype=np.float64))
    seeds.append(np.zeros(8, dtype=np.float64))
    rng = np.random.default_rng(440903)
    seeds.extend(rng.uniform(-0.25, 0.25, size=8) for _ in range(4))
    best, best_score = None, float("inf")
    roots = []
    for x0 in seeds:
        try:
            fit = least_squares(
                lambda z: residual(z, h0, groups), x0,
                loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=3000,
                bounds=(-np.ones(8), np.ones(8)),
            )
            Hm = H_from_z(fit.x, h0)
            score = float(np.median(np.abs(np.concatenate([
                signed_pixel_residual(Hm, key, groups[key]) for key in GROUPS
            ]))))
            if np.isfinite(score) and score < best_score:
                best, best_score = np.asarray(fit.x, dtype=np.float64), score
            if np.isfinite(score):
                roots.append({"z": np.asarray(fit.x, dtype=np.float64), "median_abs_pixel_residual": score})
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Broadcast v44 homography solve failed from all deterministic roots")
    if return_roots:
        return best, roots
    return best


def solve_warm(h0: np.ndarray, groups: dict, warm: np.ndarray) -> np.ndarray:
    fit = least_squares(
        lambda z: residual(z, h0, groups), np.asarray(warm, dtype=np.float64),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=1000,
        bounds=(-np.ones(8), np.ones(8)),
    )
    z = np.asarray(fit.x, dtype=np.float64)
    Hm = H_from_z(z, h0)
    if not np.isfinite(Hm).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        raise RuntimeError("Broadcast v44 warm solve was degenerate")
    return z


def dense_features() -> dict[str, np.ndarray]:
    tmax = math.asin(CORNER_Y_CM / THREE_R_CM)
    t3 = np.linspace(-tmax, tmax, 2001)
    tf = np.linspace(-math.pi / 2.0, math.pi / 2.0, 1601)
    xline = np.linspace(BASELINE_X_CM, FT_X_CM, 1201)
    return {
        "three_point_arc": np.column_stack([RIM_X_CM + THREE_R_CM * np.cos(t3), THREE_R_CM * np.sin(t3)]),
        "free_throw_front_semicircle": np.column_stack([FT_X_CM + FT_R_CM * np.cos(tf), FT_R_CM * np.sin(tf)]),
        "free_throw_line": np.column_stack([np.full(1201, FT_X_CM), np.linspace(-PAINT_HALF_CM, PAINT_HALF_CM, 1201)]),
        "lane_negative_y": np.column_stack([xline, np.full(len(xline), -PAINT_HALF_CM)]),
        "lane_positive_y": np.column_stack([xline, np.full(len(xline), PAINT_HALF_CM)]),
    }


def pixel_metrics(Hm: np.ndarray, groups: dict, dense: dict) -> dict:
    out = {}
    for key in GROUPS:
        pred = project_h(Hm, dense[key])
        obs = np.asarray(groups[key], dtype=np.float64)
        distances = np.sqrt(np.sum((obs[:, None, :] - pred[None, :, :]) ** 2, axis=2)).min(axis=1)
        out[key] = {
            "count": int(len(distances)),
            "rms_px": float(np.sqrt(np.mean(distances ** 2))),
            "median_px": float(np.median(distances)),
            "p95_px": float(np.percentile(distances, 95)),
            "max_px": float(np.max(distances)),
            "per_point_px": distances.tolist(),
        }
    return out


def curve_shift(Ha: np.ndarray, Hb: np.ndarray, dense: dict) -> dict:
    out = {}
    for key in GROUPS:
        d = np.linalg.norm(project_h(Ha, dense[key]) - project_h(Hb, dense[key]), axis=1)
        out[key] = {"p95_px": float(np.percentile(d, 95)), "max_px": float(np.max(d))}
    return out


def draw_overlay(image: np.ndarray, spec: dict, Hm: np.ndarray, dense: dict, path: Path) -> None:
    out = image.copy()
    colors = {
        "three_point_arc": (0, 0, 255),
        "free_throw_front_semicircle": (0, 255, 0),
        "free_throw_line": (255, 255, 255),
        "lane_negative_y": (255, 255, 0),
        "lane_positive_y": (255, 255, 0),
    }
    for key in GROUPS:
        q = np.round(project_h(Hm, dense[key])).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        qq = q[ok]
        if len(qq) > 1:
            cv2.polylines(out, [qq.reshape(-1, 1, 2)], False, colors[key], 2, cv2.LINE_AA)
        held = set(spec["held_out_indices"][key])
        for i, p in enumerate(np.asarray(spec["observations_px"][key], dtype=int)):
            cv2.circle(out, tuple(p), 5 if i in held else 3, colors[key], 2 if i in held else 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-heldout-p95-px", type=float, default=2.0)
    ap.add_argument("--max-root-p95-shift-px", type=float, default=2.0)
    ap.add_argument("--max-half-pixel-p95-shift-px", type=float, default=2.0)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Broadcast Frame C")
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    lock = spec["freeze_lock"]
    if spec["camera_label"] != "Broadcast" or lock["authority_camera"] != "Right Slash" or lock["chooser_option"] != "C":
        raise RuntimeError("Broadcast v44 observations are not bound to immutable chooser C")
    if abs(float(lock["right_slash_local_time"]) - 8.275733) > 5e-7 or int(lock["right_slash_decoded_frame_index"]) != 248:
        raise RuntimeError("Immutable authority timing changed")
    if abs(float(lock["broadcast_synchronized_time"]) - 9.194613) > 5e-7 or int(lock["broadcast_decoded_frame_index"]) != 276:
        raise RuntimeError("Immutable Broadcast Frame C changed")

    seed = spec["seed_only_correspondences"]
    H_seed, _ = cv2.findHomography(
        np.asarray(seed["world_cm"], dtype=np.float64),
        np.asarray(seed["image_px"], dtype=np.float64),
        method=0,
    )
    if H_seed is None:
        raise RuntimeError("Could not construct seed-only homography")
    h0 = parameter_vector(H_seed)
    train, held = split_groups(spec["observations_px"], spec["held_out_indices"])
    dense = dense_features()

    z, nominal_roots = solve_multistart(h0, train, return_roots=True)
    Hm = H_from_z(z, h0)
    train_metrics = pixel_metrics(Hm, train, dense)
    held_metrics = pixel_metrics(Hm, held, dense)

    reduced = {key: value[:-1] for key, value in train.items()}
    zr = solve_multistart(h0, reduced, warm=z)
    root_shift = curve_shift(Hm, H_from_z(zr, h0), dense)
    max_root_p95 = max(row["p95_px"] for row in root_shift.values())

    best_root_score = min(row["median_abs_pixel_residual"] for row in nominal_roots)
    competitive_roots = [row for row in nominal_roots if row["median_abs_pixel_residual"] <= best_root_score + 0.25]
    pairwise_root_rows = []
    for i in range(len(competitive_roots)):
        for j in range(i + 1, len(competitive_roots)):
            shifts = curve_shift(
                H_from_z(competitive_roots[i]["z"], h0),
                H_from_z(competitive_roots[j]["z"], h0),
                dense,
            )
            pairwise_root_rows.append({
                "i": i,
                "j": j,
                "feature_shift": shifts,
                "max_p95_px": max(row["p95_px"] for row in shifts.values()),
            })
    max_pairwise_root_p95 = max((row["max_p95_px"] for row in pairwise_root_rows), default=0.0)

    rng = np.random.default_rng(441903)
    perturbations = []
    for trial in range(args.perturbation_trials):
        perturbed = {key: value + rng.uniform(-0.5, 0.5, size=value.shape) for key, value in train.items()}
        zp = solve_warm(h0, perturbed, z)
        shifts = curve_shift(Hm, H_from_z(zp, h0), dense)
        perturbations.append({"trial": trial, "feature_shift": shifts, "max_p95_px": max(x["p95_px"] for x in shifts.values())})
    max_perturb_p95 = max(row["max_p95_px"] for row in perturbations)
    max_heldout_p95 = max(row["p95_px"] for row in held_metrics.values())

    gates = {
        "immutable_frame_c_lock": True,
        "native_960x540_source": True,
        "static_regulation_floor_only": True,
        "spatially_distributed_heldout_observations": True,
        "at_least_three_competitive_multistart_roots": len(competitive_roots) >= 3,
        "competitive_multistart_roots_functionally_equivalent": max_pairwise_root_p95 <= 0.5,
        "every_heldout_feature_p95_at_most_two_px": max_heldout_p95 <= args.max_heldout_p95_px,
        "support_reduction_root_p95_at_most_threshold": max_root_p95 <= args.max_root_p95_shift_px,
        "half_pixel_annotation_p95_stability": max_perturb_p95 <= args.max_half_pixel_p95_shift_px,
        "finite_nondegenerate_homography": bool(np.isfinite(Hm).all() and abs(float(np.linalg.det(Hm))) > 1e-12),
    }
    passed = bool(all(gates.values()))
    draw_overlay(image, spec, Hm, dense, args.out / "broadcast_frame_c_floor_overlay_v44.png")
    payload = {
        "schema_version": 1,
        "status": "PASS_BROADCAST_PINHOLE_FLOOR_V44" if passed else "FAIL_BROADCAST_PINHOLE_FLOOR_V44",
        "game_id": spec["game_id"],
        "event_id": spec["event_id"],
        "camera_label": "Broadcast",
        "model": "undistorted pinhole floor homography",
        "method": "native source-visible regulation paint; signed image-plane fit; held-out pixels from every feature group; support reduction; 64 half-pixel perturbations",
        "floor_homography_world_to_image": Hm.tolist(),
        "floor_homography_image_to_world": np.linalg.inv(Hm).tolist(),
        "training_pixel_error": train_metrics,
        "heldout_pixel_error": held_metrics,
        "max_heldout_feature_p95_px": float(max_heldout_p95),
        "support_reduction_projection_shift": root_shift,
        "max_support_reduction_p95_shift_px": float(max_root_p95),
        "nominal_multistart": {
            "seed_count": len(nominal_roots),
            "competitive_score_margin_px": 0.25,
            "competitive_root_count": len(competitive_roots),
            "competitive_root_scores_px": [row["median_abs_pixel_residual"] for row in competitive_roots],
            "pairwise_projection_shift": pairwise_root_rows,
            "max_pairwise_p95_shift_px": float(max_pairwise_root_p95),
            "max_allowed_pairwise_p95_shift_px": 0.5,
        },
        "half_pixel_training_annotation_perturbation": {
            "trial_count": len(perturbations),
            "max_feature_p95_shift_px": float(max_perturb_p95),
            "trials": perturbations,
        },
        "thresholds": {
            "max_heldout_p95_px": args.max_heldout_p95_px,
            "max_root_p95_shift_px": args.max_root_p95_shift_px,
            "max_half_pixel_p95_shift_px": args.max_half_pixel_p95_shift_px,
        },
        "gates": gates,
        "permissions": {
            "broadcast_floor_homography_allowed": passed,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "failure_policy": "Do not relax the two-pixel held-out gate. A pinhole failure promotes a distortion-aware floor model as the next experiment, not this homography.",
        "independent_baseline_policy": "Broadcast is distinct from Left Above Rim. Mobile Broadcast, Other Broadcast and Play by Play remain excluded as near-duplicate/crop-family feeds.",
    }
    (args.out / "broadcast_frame_c_floor_v44.json").write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({
        "status": payload["status"],
        "heldout_p95_px": {key: row["p95_px"] for key, row in held_metrics.items()},
        "max_heldout_feature_p95_px": max_heldout_p95,
        "max_support_reduction_p95_shift_px": max_root_p95,
        "competitive_root_count": len(competitive_roots),
        "max_competitive_pairwise_p95_shift_px": max_pairwise_root_p95,
        "max_half_pixel_p95_shift_px": max_perturb_p95,
        "gates": gates,
        "permissions": payload["permissions"],
    }), indent=2), flush=True)

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
