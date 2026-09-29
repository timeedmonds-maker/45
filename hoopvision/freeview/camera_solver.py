from __future__ import annotations

"""Metric sports-camera calibration for HoopVision free-view replay.

Clean-room implementation inspired by the *ideas* in TVCalib/PnLCalib:
known metric sports geometry + point/segment reprojection + robust nonlinear
refinement.  It does not copy either project's implementation.

Coordinate convention used by the free-view tool is basket-local centimetres:
  +X: backboard plane toward the court
  +Y: across the court, positive to camera-independent court right
  +Z: upward
The near backboard plane is X=0, rim centre is (38.1, 0, 304.8).
"""

from dataclasses import dataclass, field
import itertools
import math
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


RIM_CENTER_CM = np.array([38.1, 0.0, 304.8], dtype=np.float64)
RIM_RADIUS_CM = 9.0 * 2.54
BOARD_X_CM = 0.0
# 20 x 14 inch clear opening inside the 24 x 18 inch target rectangle stripe.
TARGET_INNER_HALF_W_CM = 10.0 * 2.54
TARGET_INNER_BOTTOM_Z_CM = 10.0 * 30.48 + 2.0 * 2.54
TARGET_INNER_TOP_Z_CM = 10.0 * 30.48 + 16.0 * 2.54


def rim_world_points(n: int = 240) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.column_stack([
        RIM_CENTER_CM[0] + RIM_RADIUS_CM * np.cos(th),
        RIM_CENTER_CM[1] + RIM_RADIUS_CM * np.sin(th),
        np.full_like(th, RIM_CENTER_CM[2]),
    ]).astype(np.float64)


def target_inner_corners(mirror_y: bool = False) -> dict[str, np.ndarray]:
    s = -1.0 if mirror_y else 1.0
    return {
        "target_inner_top_left": np.array([BOARD_X_CM, s * -TARGET_INNER_HALF_W_CM, TARGET_INNER_TOP_Z_CM]),
        "target_inner_top_right": np.array([BOARD_X_CM, s * TARGET_INNER_HALF_W_CM, TARGET_INNER_TOP_Z_CM]),
        "target_inner_bottom_right": np.array([BOARD_X_CM, s * TARGET_INNER_HALF_W_CM, TARGET_INNER_BOTTOM_Z_CM]),
        "target_inner_bottom_left": np.array([BOARD_X_CM, s * -TARGET_INNER_HALF_W_CM, TARGET_INNER_BOTTOM_Z_CM]),
    }


def target_inner_lines(mirror_y: bool = False) -> dict[str, np.ndarray]:
    c = target_inner_corners(mirror_y)
    return {
        "target_top": np.stack([c["target_inner_top_left"], c["target_inner_top_right"]]),
        "target_left": np.stack([c["target_inner_top_left"], c["target_inner_bottom_left"]]),
        "target_right": np.stack([c["target_inner_top_right"], c["target_inner_bottom_right"]]),
        "target_bottom": np.stack([c["target_inner_bottom_left"], c["target_inner_bottom_right"]]),
    }


def target_stripe_center_lines(mirror_y: bool = False) -> dict[str, np.ndarray]:
    """Centreline of the 2-inch white target stripe (22 x 16 inches)."""
    s = -1.0 if mirror_y else 1.0
    hw = 11.0 * 2.54
    z0 = 10.0 * 30.48 + 1.0 * 2.54
    z1 = 10.0 * 30.48 + 17.0 * 2.54
    tl=np.array([BOARD_X_CM,s*-hw,z1],float); tr=np.array([BOARD_X_CM,s*hw,z1],float)
    bl=np.array([BOARD_X_CM,s*-hw,z0],float); br=np.array([BOARD_X_CM,s*hw,z0],float)
    return {"target_top":np.stack([tl,tr]),"target_left":np.stack([tl,bl]),"target_right":np.stack([tr,br]),"target_bottom":np.stack([bl,br])}


@dataclass
class PointObservation:
    world: np.ndarray
    image: np.ndarray
    weight: float = 1.0
    family: str = "point"


@dataclass
class CurveObservation:
    world_curve: np.ndarray
    image_points: np.ndarray
    weight: float = 1.0
    family: str = "curve"


@dataclass
class LineObservation:
    world_segment: np.ndarray
    image_points: np.ndarray
    weight: float = 1.0
    family: str = "line"


@dataclass
class CameraObservations:
    label: str
    width: int
    height: int
    points: list[PointObservation] = field(default_factory=list)
    curves: list[CurveObservation] = field(default_factory=list)
    lines: list[LineObservation] = field(default_factory=list)
    center_prior_cm: np.ndarray | None = None
    center_prior_sigma_cm: float | None = None
    focal_prior_px: float | None = None
    focal_prior_sigma_log: float = 1.5
    pp_prior_px: np.ndarray | None = None
    pp_prior_sigma_px: float = 180.0


@dataclass
class CameraState:
    label: str
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray
    distortion: np.ndarray
    cost: float
    metrics: dict

    def as_json(self) -> dict:
        rv, _ = cv2.Rodrigues(self.R)
        return {
            "label": self.label,
            "K": self.K.tolist(),
            "R_world_to_camera": self.R.tolist(),
            "center_cm": self.C.tolist(),
            "rvec": rv.reshape(-1).tolist(),
            "distortion_k1_k2": self.distortion.tolist(),
            "focal_px": float(self.K[0, 0]),
            "principal_point_px": [float(self.K[0, 2]), float(self.K[1, 2])],
            "cost": float(self.cost),
            "metrics": self.metrics,
        }


def rodrigues_to_R(rv: Sequence[float]) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))[0]


def project_points(
    world: np.ndarray,
    rv: Sequence[float],
    C: Sequence[float],
    logf: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    R = rodrigues_to_R(rv)
    C = np.asarray(C, dtype=np.float64).reshape(3)
    Xc = (R @ (P - C).T).T
    z = Xc[:, 2]
    den = np.where(np.abs(z) < 1e-9, np.sign(z) * 1e-9 + (z == 0) * 1e-9, z)
    x = Xc[:, 0] / den
    y = Xc[:, 1] / den
    r2 = x * x + y * y
    radial = 1.0 + float(k1) * r2 + float(k2) * r2 * r2
    x *= radial
    y *= radial
    f = float(np.exp(logf))
    uv = np.column_stack([f * x + float(cx), f * y + float(cy)])
    return uv, z


def _line_signed_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    n = np.array([-d[1], d[0]], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return np.full(len(p), 1e3)
    n /= norm
    return (p - a) @ n


def _nearest_curve_distance(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    if len(obs) == 0 or len(pred) == 0:
        return np.empty(0, dtype=float)
    # Tiny curves (rim etc.): explicit dense nearest neighbour is deterministic
    # and avoids a heavy dependency.
    d2 = ((obs[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(np.min(d2, axis=1))


def _state_from_vec(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    rv = z[0:3]
    C = z[3:6]
    logf, cx, cy, k1, k2 = [float(v) for v in z[6:11]]
    return rv, C, logf, cx, cy, k1, k2


def residual_vector(z: np.ndarray, obs: CameraObservations, families: set[str] | None = None) -> np.ndarray:
    rv, C, logf, cx, cy, k1, k2 = _state_from_vec(np.asarray(z, float))
    rows: list[np.ndarray] = []

    for o in obs.points:
        if families is not None and o.family not in families:
            continue
        uv, depth = project_points(o.world[None, :], rv, C, logf, cx, cy, k1, k2)
        r = (uv[0] - o.image) * float(o.weight)
        if depth[0] <= 1.0:
            r = r + np.sign(r + 1e-6) * min(500.0, 1.0 - float(depth[0]))
        rows.append(r)

    for o in obs.lines:
        if families is not None and o.family not in families:
            continue
        uv, depth = project_points(o.world_segment, rv, C, logf, cx, cy, k1, k2)
        if np.all(depth > 1.0):
            r = _line_signed_distance(o.image_points, uv[0], uv[-1]) * float(o.weight)
        else:
            r = np.full(len(o.image_points), 250.0 * float(o.weight))
        rows.append(r)

    for o in obs.curves:
        if families is not None and o.family not in families:
            continue
        uv, depth = project_points(o.world_curve, rv, C, logf, cx, cy, k1, k2)
        if np.mean(depth > 1.0) > 0.95:
            r = _nearest_curve_distance(o.image_points, uv) * float(o.weight)
        else:
            r = np.full(len(o.image_points), 250.0 * float(o.weight))
        rows.append(r)

    # Soft physical priors. They stabilize broadcast crops but are deliberately
    # much weaker than pixel geometry.
    if obs.center_prior_cm is not None and obs.center_prior_sigma_cm:
        rows.append((C - np.asarray(obs.center_prior_cm, float)) / float(obs.center_prior_sigma_cm))
    if obs.focal_prior_px is not None and obs.focal_prior_px > 0:
        rows.append(np.array([(logf - math.log(float(obs.focal_prior_px))) / float(obs.focal_prior_sigma_log)]))
    pp0 = np.asarray(obs.pp_prior_px if obs.pp_prior_px is not None else [obs.width / 2, obs.height / 2], float)
    rows.append((np.array([cx, cy]) - pp0) / float(obs.pp_prior_sigma_px))
    # Distortion is allowed, but unexplained large warps are discouraged.
    rows.append(np.array([k1 / 0.08, k2 / 0.04]) * 0.20)

    if not rows:
        return np.array([1e6], dtype=float)
    return np.concatenate([np.asarray(r, float).reshape(-1) for r in rows])


def _bounds(obs: CameraObservations) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([
        -10, -10, -10,
        -40000, -40000, 50,
        math.log(120.0), -400.0, -400.0, -0.20, -0.10,
    ], dtype=float)
    hi = np.array([
        10, 10, 10,
        40000, 40000, 6000,
        math.log(8000.0), obs.width + 400.0, obs.height + 400.0, 0.20, 0.10,
    ], dtype=float)
    return lo, hi


def homography_initializations(obs: CameraObservations) -> list[np.ndarray]:
    floor = [o for o in obs.points if abs(float(o.world[2])) < 1e-6]
    if len(floor) < 4:
        return []
    world_xy = np.asarray([[o.world[0], o.world[1]] for o in floor], np.float64)
    image_xy = np.asarray([o.image for o in floor], np.float64)
    H, mask = cv2.findHomography(world_xy, image_xy, cv2.RANSAC, 4.0)
    if H is None:
        return []
    seeds: list[np.ndarray] = []
    f0s = [500.0, 750.0, 1000.0, 1400.0, 2000.0, 3000.0]
    pp0s = [
        (obs.width / 2, obs.height / 2),
        (obs.width / 2, obs.height * 0.65),
        (obs.width * 0.42, obs.height * 0.62),
        (obs.width * 0.58, obs.height * 0.62),
    ]
    for f0, (cx, cy) in itertools.product(f0s, pp0s):
        K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1.0]], dtype=float)
        B = np.linalg.inv(K) @ H
        n1, n2 = np.linalg.norm(B[:, 0]), np.linalg.norm(B[:, 1])
        if min(n1, n2) < 1e-12:
            continue
        lam0 = 2.0 / (n1 + n2)
        for sign in (1.0, -1.0):
            lam = sign * lam0
            r1, r2 = lam * B[:, 0], lam * B[:, 1]
            r3 = np.cross(r1, r2)
            R0 = np.column_stack([r1, r2, r3])
            U, _, Vt = np.linalg.svd(R0)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            t = lam * B[:, 2]
            C = -R.T @ t
            rv, _ = cv2.Rodrigues(R)
            z = np.r_[rv.reshape(-1), C, math.log(f0), cx, cy, 0.0, 0.0]
            if np.isfinite(z).all() and 20.0 < C[2] < 6000.0:
                seeds.append(z)
    return seeds


def pnp_initializations(obs: CameraObservations) -> list[np.ndarray]:
    if len(obs.points) < 4:
        return []
    obj = np.asarray([o.world for o in obs.points], np.float64)
    img = np.asarray([o.image for o in obs.points], np.float64)
    # PnP is used only as another hypothesis generator. Unknown intrinsics are
    # scanned coarsely; final intrinsics/extrinsics are jointly refined below.
    seeds: list[np.ndarray] = []
    for f0 in (600.0, 900.0, 1300.0, 2000.0):
        for cx, cy in ((obs.width/2, obs.height/2), (obs.width/2, obs.height*0.65)):
            K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]], float)
            try:
                out = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                continue
            if not out or not bool(out[0]):
                continue
            rvecs, tvecs = out[1], out[2]
            for rv, tv in zip(rvecs, tvecs):
                R = rodrigues_to_R(np.asarray(rv).reshape(-1))
                C = -R.T @ np.asarray(tv, float).reshape(3)
                z = np.r_[np.asarray(rv).reshape(-1), C, math.log(f0), cx, cy, 0.0, 0.0]
                if np.isfinite(z).all() and 20.0 < C[2] < 6000.0:
                    seeds.append(z)
    return seeds


def camera_metrics(z: np.ndarray, obs: CameraObservations) -> dict:
    rv, C, logf, cx, cy, k1, k2 = _state_from_vec(z)
    fam: dict[str, list[float]] = {}
    positive = []
    for o in obs.points:
        uv, d = project_points(o.world[None, :], rv, C, logf, cx, cy, k1, k2)
        fam.setdefault(o.family, []).extend(np.linalg.norm(uv - o.image[None, :], axis=1).tolist())
        positive.extend((d > 1.0).tolist())
    for o in obs.lines:
        uv, d = project_points(o.world_segment, rv, C, logf, cx, cy, k1, k2)
        fam.setdefault(o.family, []).extend(np.abs(_line_signed_distance(o.image_points, uv[0], uv[-1])).tolist())
        positive.extend((d > 1.0).tolist())
    for o in obs.curves:
        uv, d = project_points(o.world_curve, rv, C, logf, cx, cy, k1, k2)
        fam.setdefault(o.family, []).extend(_nearest_curve_distance(o.image_points, uv).tolist())
        positive.extend((d > 1.0).tolist())
    def stats(v: list[float]):
        a = np.asarray(v, float)
        return {
            "count": int(len(a)),
            "median_px": float(np.median(a)) if len(a) else None,
            "p95_px": float(np.percentile(a, 95)) if len(a) else None,
            "max_px": float(np.max(a)) if len(a) else None,
        }
    allv = [x for v in fam.values() for x in v]
    return {
        "families": {k: stats(v) for k, v in fam.items()},
        "all": stats(allv),
        "positive_depth_fraction": float(np.mean(positive)) if positive else 0.0,
        "camera_height_cm": float(C[2]),
        "camera_distance_to_rim_cm": float(np.linalg.norm(C - RIM_CENTER_CM)),
        "focal_px": float(np.exp(logf)),
        "principal_point_px": [float(cx), float(cy)],
        "distortion": [float(k1), float(k2)],
    }


def _fit_seed(seed: np.ndarray, obs: CameraObservations, max_nfev: int = 2500) -> tuple[np.ndarray, float]:
    lo, hi = _bounds(obs)
    seed = np.minimum(np.maximum(np.asarray(seed, float), lo + 1e-6), hi - 1e-6)
    q = least_squares(
        lambda z: residual_vector(z, obs), seed,
        bounds=(lo, hi), loss="soft_l1", f_scale=1.5,
        x_scale="jac", max_nfev=int(max_nfev),
    )
    return np.asarray(q.x, float), float(q.cost)


def solve_camera(
    obs: CameraObservations,
    extra_seeds: Sequence[np.ndarray] | None = None,
    max_seed_count: int = 10,
) -> CameraState:
    seeds = homography_initializations(obs) + pnp_initializations(obs)
    if extra_seeds:
        seeds.extend([np.asarray(s, float) for s in extra_seeds])
    if not seeds:
        raise RuntimeError(f"{obs.label}: no valid camera initialization")

    # Rank cheap residual at seed and refine only the strongest hypotheses.
    scored = sorted((float(np.mean(np.square(residual_vector(s, obs)))), s) for s in seeds)
    best: tuple[np.ndarray, float] | None = None
    for _, s in scored[: int(max_seed_count)]:
        try:
            z, cost = _fit_seed(s, obs)
        except Exception:
            continue
        m = camera_metrics(z, obs)
        C = z[3:6]
        plausible = (
            m["positive_depth_fraction"] >= 0.98
            and 50.0 < C[2] < 6000.0
            and 120.0 <= m["focal_px"] <= 8000.0
            and m["camera_distance_to_rim_cm"] < 50000.0
        )
        if plausible and (best is None or cost < best[1]):
            best = (z, cost)
    if best is None:
        raise RuntimeError(f"{obs.label}: optimizer found no physically plausible camera")
    z, cost = best
    rv, C, logf, cx, cy, k1, k2 = _state_from_vec(z)
    f = float(np.exp(logf))
    state = CameraState(
        label=obs.label,
        K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float),
        R=rodrigues_to_R(rv),
        C=np.asarray(C, float),
        distortion=np.asarray([k1, k2], float),
        cost=cost,
        metrics=camera_metrics(z, obs),
    )
    return state


def perturbation_stability(
    obs: CameraObservations,
    nominal: CameraState,
    trials: int = 12,
    pixel_jitter: float = 0.5,
    seed: int = 130489,
) -> dict:
    rv, _ = cv2.Rodrigues(nominal.R)
    z0 = np.r_[
        rv.reshape(-1), nominal.C,
        math.log(float(nominal.K[0, 0])), nominal.K[0, 2], nominal.K[1, 2],
        nominal.distortion,
    ]
    shifts = []
    rng = np.random.default_rng(int(seed))
    for _ in range(int(trials)):
        pobs = CameraObservations(
            label=obs.label, width=obs.width, height=obs.height,
            center_prior_cm=obs.center_prior_cm,
            center_prior_sigma_cm=obs.center_prior_sigma_cm,
            focal_prior_px=obs.focal_prior_px,
            focal_prior_sigma_log=obs.focal_prior_sigma_log,
            pp_prior_px=obs.pp_prior_px,
            pp_prior_sigma_px=obs.pp_prior_sigma_px,
        )
        for o in obs.points:
            pobs.points.append(PointObservation(o.world.copy(), o.image + rng.choice([-pixel_jitter, pixel_jitter], 2), o.weight, o.family))
        for o in obs.lines:
            noise = rng.choice([-pixel_jitter, pixel_jitter], np.asarray(o.image_points).shape)
            pobs.lines.append(LineObservation(o.world_segment.copy(), o.image_points + noise, o.weight, o.family))
        for o in obs.curves:
            noise = rng.choice([-pixel_jitter, pixel_jitter], np.asarray(o.image_points).shape)
            pobs.curves.append(CurveObservation(o.world_curve.copy(), o.image_points + noise, o.weight, o.family))
        try:
            z, _ = _fit_seed(z0, pobs, max_nfev=1800)
            shifts.append(float(np.linalg.norm(z[3:6] - nominal.C)))
        except Exception:
            shifts.append(float("inf"))
    a = np.asarray(shifts, float)
    return {
        "trials": int(trials),
        "pixel_jitter": float(pixel_jitter),
        "center_shift_cm_median": float(np.median(a)),
        "center_shift_cm_p95": float(np.percentile(a, 95)),
        "center_shift_cm_max": float(np.max(a)),
        "all_finite": bool(np.isfinite(a).all()),
    }


def multistart_stability(obs: CameraObservations, nominal: CameraState, trials: int = 10, seed: int = 130490) -> dict:
    rv, _ = cv2.Rodrigues(nominal.R)
    z0 = np.r_[rv.reshape(-1), nominal.C, math.log(float(nominal.K[0, 0])), nominal.K[0,2], nominal.K[1,2], nominal.distortion]
    rng = np.random.default_rng(int(seed))
    shifts = []
    costs = []
    for _ in range(int(trials)):
        s = z0.copy()
        s[:3] += rng.normal(0, 0.08, 3)
        s[3:6] += rng.normal(0, 150.0, 3)
        s[6] += rng.normal(0, 0.18)
        s[7:9] += rng.normal(0, 60.0, 2)
        s[9:11] += rng.normal(0, 0.01, 2)
        try:
            z, cost = _fit_seed(s, obs, max_nfev=2200)
            shifts.append(float(np.linalg.norm(z[3:6] - nominal.C)))
            costs.append(float(cost))
        except Exception:
            shifts.append(float("inf")); costs.append(float("inf"))
    a = np.asarray(shifts, float)
    return {
        "trials": int(trials),
        "center_shift_cm_median": float(np.median(a)),
        "center_shift_cm_p95": float(np.percentile(a, 95)),
        "center_shift_cm_max": float(np.max(a)),
        "all_finite": bool(np.isfinite(a).all()),
        "costs": costs,
    }


def acceptance_report(state: CameraState, perturb: dict, multi: dict) -> dict:
    m = state.metrics
    all_p95 = m["all"]["p95_px"] if m["all"]["p95_px"] is not None else 1e9
    family_p95 = [v["p95_px"] for v in m["families"].values() if v["p95_px"] is not None]
    max_family = max(family_p95) if family_p95 else 1e9
    gates = {
        "all_reprojection_p95_le_6px": bool(all_p95 <= 6.0),
        "family_reprojection_p95_le_8px": bool(max_family <= 8.0),
        "positive_depth_ge_0_98": bool(m["positive_depth_fraction"] >= 0.98),
        "half_pixel_center_shift_le_75cm": bool(perturb["all_finite"] and perturb["center_shift_cm_max"] <= 75.0),
        "multistart_center_shift_le_75cm": bool(multi["all_finite"] and multi["center_shift_cm_max"] <= 75.0),
        "physical_height": bool(50.0 < m["camera_height_cm"] < 6000.0),
    }
    return {
        "accepted": bool(all(gates.values())),
        "gates": gates,
        "metrics": m,
        "perturbation": perturb,
        "multistart": multi,
    }


def _shared_unpack(z: np.ndarray, n: int):
    z = np.asarray(z, float)
    C = z[:3]
    states = []
    o = 3
    for _ in range(int(n)):
        rv = z[o:o+3]; logf, cx, cy, k1, k2 = z[o+3:o+8]
        states.append((rv, C, float(logf), float(cx), float(cy), float(k1), float(k2)))
        o += 8
    return C, states


def solve_shared_center_states(observations: Sequence[CameraObservations]) -> list[CameraState]:
    """Jointly solve PTZ/crop states that share one physical optical centre.

    Each state has independent rotation/focal/principal-point/distortion while C
    is shared. This directly targets operated broadcast cameras whose pan/tilt/
    zoom changes but mount location remains fixed.
    """
    observations = list(observations)
    if len(observations) < 2:
        raise ValueError("shared-centre solve requires >=2 states")
    independents = [solve_camera(o, max_seed_count=20) for o in observations]
    C0 = np.median(np.stack([s.C for s in independents]), axis=0)
    blocks = []
    for s in independents:
        rv, _ = cv2.Rodrigues(s.R)
        blocks.extend([*rv.reshape(-1), math.log(float(s.K[0,0])), float(s.K[0,2]), float(s.K[1,2]), *s.distortion])
    z0 = np.r_[C0, np.asarray(blocks, float)]

    lo = [-40000, -40000, 50]
    hi = [40000, 40000, 6000]
    for o in observations:
        lo.extend([-10,-10,-10, math.log(120), -400,-400,-0.2,-0.1])
        hi.extend([10,10,10, math.log(8000), o.width+400,o.height+400,0.2,0.1])
    lo, hi = np.asarray(lo,float), np.asarray(hi,float)

    def fun(z):
        C, states = _shared_unpack(z, len(observations))
        rows=[]
        for o, st in zip(observations, states):
            rv, _, logf, cx, cy, k1, k2 = st
            zz=np.r_[rv,C,logf,cx,cy,k1,k2]
            rows.append(residual_vector(zz,o))
        return np.concatenate(rows)

    z0=np.minimum(np.maximum(z0,lo+1e-6),hi-1e-6)
    q=least_squares(fun,z0,bounds=(lo,hi),loss='soft_l1',f_scale=1.5,x_scale='jac',max_nfev=7000)
    C, sts=_shared_unpack(q.x,len(observations))
    out=[]
    for o, st in zip(observations, sts):
        rv, _, logf, cx, cy, k1, k2=st
        zz=np.r_[rv,C,logf,cx,cy,k1,k2]
        f=float(np.exp(logf)); R=rodrigues_to_R(rv)
        out.append(CameraState(
            label=o.label,K=np.array([[f,0,cx],[0,f,cy],[0,0,1]],float),R=R,C=C.copy(),
            distortion=np.array([k1,k2],float),cost=float(q.cost),metrics=camera_metrics(zz,o)))
    return out