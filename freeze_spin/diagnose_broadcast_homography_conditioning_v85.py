from __future__ import annotations

"""v85: diagnose the robustness failure of the v84 Broadcast floor homography.

This is discovery-only.  It does not weaken or reinterpret any v84 gate.

v84 established an accurate and uniquely converged line-aware nominal homography,
but failed the inherited v44 support-reduction and half-pixel perturbation gates.
This diagnostic separates those failure modes by:

1. repeating the exact v44 simultaneous-last-training-point reduction;
2. exhaustively removing ONE training observation at a time and refitting;
3. removing the last training observation from ONE feature family at a time;
4. repeating the exact 64 deterministic +/-0.5 px perturbation trials while
   recording the worst feature, held-out error, parameter change and input noise;
5. measuring the numerical conditioning of the data-only residual Jacobian.

The metric loci use the v83/v84 NBA 2-inch painted-line centre convention.
No pixel is moved, added, solver-corrected, or transferred from another camera.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44


FOOT_CM = 30.48
INCH_CM = 2.54
LINE_CENTER_OFFSET_CM = INCH_CM
EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_line_aware_geometry() -> None:
    v44.FT_X_CM = 15.0 * FOOT_CM - LINE_CENTER_OFFSET_CM
    v44.FT_R_CM = 6.0 * FOOT_CM - LINE_CENTER_OFFSET_CM
    v44.THREE_R_CM = 23.75 * FOOT_CM - LINE_CENTER_OFFSET_CM
    v44.PAINT_HALF_CM = 8.0 * FOOT_CM - LINE_CENTER_OFFSET_CM


def h_seed_from_spec(spec: dict) -> np.ndarray:
    seed = spec["seed_only_correspondences"]
    H, _ = cv2.findHomography(
        np.asarray(seed["world_cm"], dtype=np.float64),
        np.asarray(seed["image_px"], dtype=np.float64),
        method=0,
    )
    if H is None:
        raise RuntimeError("Could not build v85 seed homography")
    return v44.parameter_vector(H)


def copied(groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in groups.items()}


def max_shift(summary: dict) -> float:
    return float(max(v["p95_px"] for v in summary.values()))


def max_heldout(H: np.ndarray, held: dict, dense: dict) -> tuple[float, dict]:
    m = v44.pixel_metrics(H, held, dense)
    return float(max(x["p95_px"] for x in m.values())), m


def train_original_indices(spec: dict) -> dict[str, list[int]]:
    out = {}
    for key in v44.GROUPS:
        held = set(int(i) for i in spec["held_out_indices"][key])
        out[key] = [i for i in range(len(spec["observations_px"][key])) if i not in held]
    return out


def data_residual(z: np.ndarray, h0: np.ndarray, groups: dict) -> np.ndarray:
    H = v44.H_from_z(z, h0)
    return np.concatenate([v44.signed_pixel_residual(H, key, groups[key]) for key in v44.GROUPS])


def jacobian_conditioning(z: np.ndarray, h0: np.ndarray, groups: dict) -> dict:
    z = np.asarray(z, dtype=np.float64)
    f0 = data_residual(z, h0, groups)
    J = np.zeros((len(f0), len(z)), dtype=np.float64)
    eps = 1e-5
    for j in range(len(z)):
        d = np.zeros_like(z)
        d[j] = eps
        J[:, j] = (data_residual(z + d, h0, groups) - data_residual(z - d, h0, groups)) / (2.0 * eps)
    s = np.linalg.svd(J, compute_uv=False)
    tol = float(max(J.shape) * np.finfo(float).eps * s[0]) if len(s) else 0.0
    rank = int(np.sum(s > tol))
    cond = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else float("inf")
    return {
        "shape": list(J.shape),
        "rank": rank,
        "singular_values": s.tolist(),
        "condition_number": cond,
        "smallest_to_largest_singular_ratio": float(s[-1] / s[0]) if len(s) and s[0] > 0 else 0.0,
    }


def solve_variant(h0: np.ndarray, nominal_z: np.ndarray, groups: dict, held: dict, dense: dict) -> dict:
    z = v44.solve_multistart(h0, groups, warm=nominal_z)
    H = v44.H_from_z(z, h0)
    nominal_H = v44.H_from_z(nominal_z, h0)
    shift = v44.curve_shift(nominal_H, H, dense)
    held_max, held_metrics = max_heldout(H, held, dense)
    return {
        "max_curve_p95_shift_px": max_shift(shift),
        "curve_shift": shift,
        "max_heldout_feature_p95_px": held_max,
        "heldout_pixel_error": held_metrics,
        "z_delta": (z - nominal_z).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual_sha = sha256(args.frame)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"Immutable Broadcast Frame C SHA changed: {actual_sha}")

    patch_line_aware_geometry()
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    if spec.get("camera_label") != "Broadcast" or spec.get("event_id") != 489:
        raise RuntimeError("v85 observation provenance changed")

    h0 = h_seed_from_spec(spec)
    train, held = v44.split_groups(spec["observations_px"], spec["held_out_indices"])
    dense = v44.dense_features()
    original_train_indices = train_original_indices(spec)

    nominal_z, roots = v44.solve_multistart(h0, train, return_roots=True)
    nominal_H = v44.H_from_z(nominal_z, h0)
    nominal_held_max, nominal_held = max_heldout(nominal_H, held, dense)

    best_score = min(r["median_abs_pixel_residual"] for r in roots)
    competitive = [r for r in roots if r["median_abs_pixel_residual"] <= best_score + 0.25]
    pairwise = []
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            sh = v44.curve_shift(v44.H_from_z(competitive[i]["z"], h0), v44.H_from_z(competitive[j]["z"], h0), dense)
            pairwise.append({"i": i, "j": j, "max_p95_px": max_shift(sh), "curve_shift": sh})

    # Exact inherited v44 support-reduction test: remove the final training
    # observation from EVERY family simultaneously.
    simultaneous = {k: v[:-1] for k, v in train.items()}
    exact_v44_reduction = solve_variant(h0, nominal_z, simultaneous, held, dense)
    exact_v44_reduction["removed"] = {
        k: {
            "train_index": len(train[k]) - 1,
            "original_observation_index": original_train_indices[k][-1],
            "pixel": train[k][-1].tolist(),
        }
        for k in v44.GROUPS
    }

    # Remove the same terminal support point, but from only one family at a time.
    one_family_last = []
    for key in v44.GROUPS:
        g = copied(train)
        removed = g[key][-1].copy()
        g[key] = g[key][:-1]
        row = solve_variant(h0, nominal_z, g, held, dense)
        row.update({
            "feature": key,
            "train_index": len(train[key]) - 1,
            "original_observation_index": original_train_indices[key][-1],
            "removed_pixel": removed.tolist(),
        })
        one_family_last.append(row)
    one_family_last.sort(key=lambda r: r["max_curve_p95_shift_px"], reverse=True)

    # Exhaustive single-observation leave-one-out over all training evidence.
    loo = []
    for key in v44.GROUPS:
        for ti in range(len(train[key])):
            g = copied(train)
            removed = g[key][ti].copy()
            g[key] = np.delete(g[key], ti, axis=0)
            row = solve_variant(h0, nominal_z, g, held, dense)
            row.update({
                "feature": key,
                "train_index": ti,
                "original_observation_index": original_train_indices[key][ti],
                "removed_pixel": removed.tolist(),
            })
            loo.append(row)
    loo.sort(key=lambda r: r["max_curve_p95_shift_px"], reverse=True)

    # Exact deterministic v44 perturbation schedule, with full diagnostics.
    rng = np.random.default_rng(441903)
    perturb = []
    for trial in range(64):
        noise = {k: rng.uniform(-0.5, 0.5, size=v.shape) for k, v in train.items()}
        g = {k: train[k] + noise[k] for k in v44.GROUPS}
        zp = v44.solve_warm(h0, g, nominal_z)
        Hp = v44.H_from_z(zp, h0)
        shift = v44.curve_shift(nominal_H, Hp, dense)
        held_max, held_metrics = max_heldout(Hp, held, dense)
        feature_max = {k: float(v["p95_px"]) for k, v in shift.items()}
        worst_feature = max(feature_max, key=feature_max.get)
        perturb.append({
            "trial": trial,
            "max_curve_p95_shift_px": max(feature_max.values()),
            "worst_feature": worst_feature,
            "curve_p95_shift_px": feature_max,
            "curve_shift": shift,
            "max_heldout_feature_p95_px": held_max,
            "heldout_pixel_error": held_metrics,
            "z_delta": (zp - nominal_z).tolist(),
            "noise_by_feature_px": {k: noise[k].tolist() for k in v44.GROUPS},
            "max_input_point_displacement_px": float(max(np.linalg.norm(noise[k], axis=1).max() for k in v44.GROUPS)),
        })
    perturb.sort(key=lambda r: r["max_curve_p95_shift_px"], reverse=True)

    cond = jacobian_conditioning(nominal_z, h0, train)

    loo_max = float(loo[0]["max_curve_p95_shift_px"])
    loo_over_2 = int(sum(r["max_curve_p95_shift_px"] > 2.0 for r in loo))
    family_last_max = float(one_family_last[0]["max_curve_p95_shift_px"])
    perturb_max = float(perturb[0]["max_curve_p95_shift_px"])
    perturb_over_2 = int(sum(r["max_curve_p95_shift_px"] > 2.0 for r in perturb))

    # Diagnostic classification only; never grants permission.
    if loo_max <= 2.0 and family_last_max <= 2.0 and exact_v44_reduction["max_curve_p95_shift_px"] > 2.0:
        support_diagnosis = "SIMULTANEOUS_MULTI_FAMILY_REDUCTION_IS_DOMINANT_SUPPORT_FAILURE"
    elif loo_max > 2.0:
        support_diagnosis = "SINGLE_OBSERVATION_REMOVAL_CAN_DESTABILIZE_HOMOGRAPHY"
    else:
        support_diagnosis = "SUPPORT_FAILURE_NOT_ISOLATED_BY_SINGLE_POINT_TESTS"

    if perturb_max > 2.0:
        perturb_diagnosis = "FREE_HOMOGRAPHY_REMAINS_UNSTABLE_TO_HALF_PIXEL_TRAINING_NOISE"
    else:
        perturb_diagnosis = "HALF_PIXEL_INSTABILITY_NOT_REPRODUCED"

    report = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_BROADCAST_HOMOGRAPHY_CONDITIONING_V85",
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Broadcast",
        "immutable_frame_sha256": actual_sha,
        "metric_geometry": {
            "painted_line_width_in": 2.0,
            "center_offset_from_outside_dimension_in": 1.0,
            "free_throw_line_x_from_backboard_face_ft": v44.FT_X_CM / FOOT_CM,
            "free_throw_circle_centerline_radius_ft": v44.FT_R_CM / FOOT_CM,
            "three_point_arc_centerline_radius_ft": v44.THREE_R_CM / FOOT_CM,
            "lane_boundary_centerline_abs_y_ft": v44.PAINT_HALF_CM / FOOT_CM,
        },
        "nominal": {
            "max_heldout_feature_p95_px": nominal_held_max,
            "heldout_pixel_error": nominal_held,
            "competitive_root_count": len(competitive),
            "max_competitive_pairwise_p95_shift_px": float(max((r["max_p95_px"] for r in pairwise), default=0.0)),
            "homography_world_to_image": nominal_H.tolist(),
            "z": nominal_z.tolist(),
        },
        "jacobian_conditioning": cond,
        "exact_v44_simultaneous_last_point_reduction": exact_v44_reduction,
        "one_family_last_point_reduction": one_family_last,
        "single_observation_leave_one_out": {
            "trial_count": len(loo),
            "max_curve_p95_shift_px": loo_max,
            "count_over_2px": loo_over_2,
            "worst_10": loo[:10],
            "all_trials": loo,
        },
        "half_pixel_perturbation_64": {
            "trial_count": len(perturb),
            "max_curve_p95_shift_px": perturb_max,
            "count_over_2px": perturb_over_2,
            "worst_10": perturb[:10],
            "all_trials": perturb,
        },
        "diagnosis": {
            "support": support_diagnosis,
            "perturbation": perturb_diagnosis,
            "nominal_geometry_is_accurate": nominal_held_max <= 2.0,
            "nominal_optimizer_root_is_unique_in_projection": float(max((r["max_p95_px"] for r in pairwise), default=0.0)) <= 0.5,
        },
        "permissions": {
            "broadcast_floor_homography_allowed": False,
            "broadcast_shared_optical_center_prior_allowed": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "guardrail": "Do not relax v84 thresholds from this diagnostic. If instability is intrinsic to the free single-frame homography, add independent same-game/non-coplanar constraints or a physically constrained camera model rather than accepting the nominal fit alone.",
    }
    out_json = args.out / "broadcast_homography_conditioning_v85.json"
    out_json.write_text(json.dumps(v44.json_safe(report), indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": report["status"],
        "nominal_heldout_max_p95_px": nominal_held_max,
        "competitive_root_count": len(competitive),
        "competitive_pairwise_max_p95_px": report["nominal"]["max_competitive_pairwise_p95_shift_px"],
        "v44_simultaneous_reduction_max_p95_px": exact_v44_reduction["max_curve_p95_shift_px"],
        "single_family_last_reduction_max_p95_px": family_last_max,
        "single_observation_loo_max_p95_px": loo_max,
        "single_observation_loo_count_over_2px": loo_over_2,
        "worst_loo": {k: loo[0][k] for k in ["feature", "original_observation_index", "removed_pixel", "max_curve_p95_shift_px", "max_heldout_feature_p95_px"]},
        "half_pixel_max_p95_px": perturb_max,
        "half_pixel_count_over_2px": perturb_over_2,
        "worst_half_pixel_trial": perturb[0]["trial"],
        "worst_half_pixel_feature": perturb[0]["worst_feature"],
        "worst_half_pixel_heldout_max_p95_px": perturb[0]["max_heldout_feature_p95_px"],
        "jacobian_condition_number": cond["condition_number"],
        "jacobian_singular_values": cond["singular_values"],
        "diagnosis": report["diagnosis"],
        "permissions": report["permissions"],
    }
    print(json.dumps(v44.json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
