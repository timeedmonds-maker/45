from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from ultralytics import YOLO

W, H = 960, 540
COURT_L = 2865.0
COURT_W = 1524.0
RIM_Z = 304.8
RIM_RADIUS = 23.0
RIM_XS = (160.0, COURT_L - 160.0)
RIM_Y = COURT_W / 2.0

PAINT_W = 488.0
PAINT_L = 579.0
SIDELINE_3 = 91.0
STRAIGHT_3 = 424.0
FT_X = 835.0
RIM_BASELINE = 160.0
THREE_R = 724.0
PAINT_START = (COURT_W - PAINT_W) / 2.0

# Exact 33-landmark NBA world model used by frozen HoopVision v1.3.
COURT_VERTICES = np.asarray([
    (0.0, 0.0),
    (0.0, SIDELINE_3),
    (0.0, PAINT_START),
    (0.0, PAINT_START + PAINT_W),
    (0.0, COURT_W - SIDELINE_3),
    (0.0, COURT_W),
    (RIM_BASELINE, COURT_W / 2.0),
    (STRAIGHT_3, SIDELINE_3),
    (STRAIGHT_3, COURT_W - SIDELINE_3),
    (PAINT_L, PAINT_START),
    (PAINT_L, PAINT_START + PAINT_W / 2.0),
    (PAINT_L, PAINT_START + PAINT_W),
    (FT_X, 0.0),
    (RIM_BASELINE + THREE_R, COURT_W / 2.0),
    (FT_X, COURT_W),
    (COURT_L / 2.0, 0.0),
    (COURT_L / 2.0, COURT_W / 2.0),
    (COURT_L / 2.0, COURT_W),
    (COURT_L - FT_X, 0.0),
    (COURT_L - RIM_BASELINE - THREE_R, COURT_W / 2.0),
    (COURT_L - FT_X, COURT_W),
    (COURT_L - PAINT_L, PAINT_START),
    (COURT_L - PAINT_L, PAINT_START + PAINT_W / 2.0),
    (COURT_L - PAINT_L, PAINT_START + PAINT_W),
    (COURT_L - STRAIGHT_3, SIDELINE_3),
    (COURT_L - STRAIGHT_3, COURT_W - SIDELINE_3),
    (COURT_L - RIM_BASELINE, COURT_W / 2.0),
    (COURT_L, 0.0),
    (COURT_L, SIDELINE_3),
    (COURT_L, PAINT_START),
    (COURT_L, PAINT_START + PAINT_W),
    (COURT_L, COURT_W - SIDELINE_3),
    (COURT_L, COURT_W),
], dtype=np.float64)

# Previously hard-gated Adams/Jazz game-level metric centres. These are used
# only as centre priors for the three already-proven physical cameras; v1.3
# still solves the event optical state (orientation/focal/crop/distortion).
_ACCEPTED_BASKET_LOCAL_CENTRES = {
    "Left Above Rim": np.asarray([1954.0944213029006, -20.657870280048282, 370.3129555117168], dtype=np.float64),
    "Right Above Rim": np.asarray([-10.706171965705735, -0.2250433572710421, 566.0472369503494], dtype=np.float64),
    "Broadcast": np.asarray([39.513155402289954 * 30.48, 96.76737963404346 * 30.48, 33.09376291010151 * 30.48], dtype=np.float64),
}
_BOARD_TO_NEAR_BASELINE_CM = RIM_BASELINE - 15.0 * 2.54
_BOARD_TO_FAR_BASELINE_CM = COURT_L - _BOARD_TO_NEAR_BASELINE_CM


def accepted_center_candidates(label: str, rim_x: float) -> list[np.ndarray]:
    local = _ACCEPTED_BASKET_LOCAL_CENTRES.get(label)
    if local is None:
        return []
    rows = []
    for y_sign in (1.0, -1.0):
        if abs(rim_x - RIM_XS[0]) < 1.0:
            x = _BOARD_TO_NEAR_BASELINE_CM + local[0]
        else:
            x = _BOARD_TO_FAR_BASELINE_CM - local[0]
        rows.append(np.asarray([x, RIM_Y + y_sign * local[1], local[2]], dtype=np.float64))
    return rows


def K_matrix(f: float, cx: float, cy: float) -> np.ndarray:
    return np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def court_infer(model: YOLO, frame: np.ndarray) -> dict | None:
    best = None
    for size in (640, 960, 1280):
        r = model.predict(frame, imgsz=size, device="cpu", verbose=False, conf=0.3)[0]
        if len(r.boxes) == 0 or r.keypoints is None:
            continue
        bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
        xy = r.keypoints.xy.cpu().numpy()[bi].astype(np.float64)
        cf = (
            r.keypoints.conf.cpu().numpy()[bi].astype(np.float64)
            if r.keypoints.conf is not None
            else np.ones(len(xy), dtype=np.float64)
        )
        n = int(((cf >= 0.45) & (xy[:, 0] > 1) & (xy[:, 1] > 1)).sum())
        cand = (n, size, xy, cf)
        if best is None or n > best[0]:
            best = cand
        if n >= 10:
            break
    if best is None:
        return None
    return {"visible": best[0], "imgsz": best[1], "xy": best[2], "conf": best[3]}


def make_detector(detector_onnx: Path, object_eval_src: Path):
    sys.path.insert(0, str(object_eval_src))
    from object_detection_eval.inference.detectors.rfdetr import RFDETRDetector

    label_map = {
        0: "basketball-unused",
        1: "ball",
        2: "ball-in-basket",
        3: "number",
        4: "player",
        5: "player-in-possession",
        6: "player-jump-shot",
        7: "player-layup-dunk",
        8: "player-shot-block",
        9: "referee",
        10: "rim",
    }
    return RFDETRDetector(
        detector_onnx,
        label_map,
        confidence_threshold=0.025,
        num_select=300,
        input_height=640,
        input_width=640,
        providers=["CPUExecutionProvider"],
    )


def detections(detector, frame: np.ndarray) -> list[dict]:
    rows = []
    for d in detector.predict(frame):
        b = d.bbox
        rows.append({
            "class_id": int(d.class_id),
            "confidence": float(d.confidence),
            "xyxy": np.asarray([
                b.x * W,
                b.y * H,
                (b.x + b.w) * W,
                (b.y + b.h) * H,
            ], dtype=np.float64),
        })
    return rows


def best_rim(rows: list[dict]) -> dict | None:
    cand = [r for r in rows if r["class_id"] == 10 and r["confidence"] >= 0.05]
    if not cand:
        return None
    cand.sort(key=lambda r: r["confidence"], reverse=True)
    return cand[0]


def best_ball(rows: list[dict], rim_box: np.ndarray | None = None) -> dict | None:
    cand = [r for r in rows if r["class_id"] in {1, 2} and r["confidence"] >= 0.035]
    if not cand:
        return None
    if rim_box is None:
        return max(cand, key=lambda r: r["confidence"])
    rc = 0.5 * (rim_box[:2] + rim_box[2:])
    scale = max(30.0, float(np.linalg.norm(rim_box[2:] - rim_box[:2])))
    return min(
        cand,
        key=lambda r: (
            np.linalg.norm(0.5 * (r["xyxy"][:2] + r["xyxy"][2:]) - rc) / scale
            - 0.20 * r["confidence"]
        ),
    )


def rim_world(xc: float, n: int = 240) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([
        xc + RIM_RADIUS * np.cos(t),
        RIM_Y + RIM_RADIUS * np.sin(t),
        np.full_like(t, RIM_Z),
    ]).astype(np.float64)


def project_points(x: np.ndarray, pts: np.ndarray):
    rv = np.asarray(x[:3], dtype=np.float64)
    C = np.asarray(x[3:6], dtype=np.float64)
    f = math.exp(float(x[6]))
    cx, cy = float(x[7]), float(x[8])
    dist = np.asarray([x[9], x[10], 0.0, 0.0, 0.0], dtype=np.float64)
    R, _ = cv2.Rodrigues(rv.reshape(3, 1))
    t = -R @ C
    uv, _ = cv2.projectPoints(
        np.asarray(pts, dtype=np.float64),
        rv.reshape(3, 1),
        t.reshape(3, 1),
        K_matrix(f, cx, cy),
        dist,
    )
    z = (R @ (np.asarray(pts, dtype=np.float64) - C).T).T[:, 2]
    return uv.reshape(-1, 2), z, R, t


def homography_seeds(world_xy: np.ndarray, image_xy: np.ndarray) -> list[np.ndarray]:
    Hwi, _ = cv2.findHomography(
        world_xy.astype(np.float64),
        image_xy.astype(np.float64),
        cv2.RANSAC,
        5.0,
    )
    if Hwi is None:
        return []
    out = []
    for f in (400.0, 700.0, 1200.0, 2200.0):
        K = K_matrix(f, W / 2.0, H / 2.0)
        B = np.linalg.inv(K) @ Hwi
        for sign in (1.0, -1.0):
            b1, b2, b3 = sign * B[:, 0], sign * B[:, 1], sign * B[:, 2]
            s = 2.0 / max(np.linalg.norm(b1) + np.linalg.norm(b2), 1e-12)
            r1, r2, t = s * b1, s * b2, s * b3
            r3 = np.cross(r1, r2)
            R0 = np.column_stack([r1, r2, r3])
            u, _, vt = np.linalg.svd(R0)
            R = u @ vt
            if np.linalg.det(R) < 0:
                u[:, -1] *= -1
                R = u @ vt
            C = -R.T @ t
            if not np.isfinite(C).all():
                continue
            rv, _ = cv2.Rodrigues(R)
            out.append(np.r_[
                rv.ravel(),
                C,
                math.log(f),
                W / 2.0,
                H / 2.0,
                0.0,
                0.0,
            ])
    return out


def _camera_residual(
    x: np.ndarray,
    floor_world: np.ndarray,
    floor_image: np.ndarray,
    rim_box: np.ndarray,
    rim_x: float,
) -> np.ndarray:
    rows = []
    uv, z, _, _ = project_points(x, floor_world)
    rows.append(((uv - floor_image) / 2.0).ravel())

    ring = rim_world(rim_x, 240)
    ru, rz, _, _ = project_points(x, ring)
    if np.isfinite(ru).all():
        pred = np.asarray([
            ru[:, 0].min(),
            ru[:, 1].min(),
            ru[:, 0].max(),
            ru[:, 1].max(),
        ])
        rows.append((pred - rim_box) / 2.0)
        rw = float(ru[:, 0].max() - ru[:, 0].min())
        rh = float(ru[:, 1].max() - ru[:, 1].min())
    else:
        rows.append(np.full(4, 500.0))
        rw = rh = 0.0

    bw = float(rim_box[2] - rim_box[0])
    bh = float(rim_box[3] - rim_box[1])
    rows.append(np.asarray([(rw - bw) / 3.0, (rh - bh) / 3.0]))

    C = np.asarray(x[3:6], dtype=np.float64)
    f = math.exp(float(x[6]))
    cx, cy = float(x[7]), float(x[8])
    rows.append(np.asarray([
        (cx - W / 2.0) / 180.0,
        (cy - H / 2.0) / 150.0,
        math.log(f / 900.0) / 1.6,
        x[9] / 0.25,
        x[10] / 0.15,
    ]))

    floor_depth_bad = max(0.0, 25.0 - float(np.percentile(z, 10))) / 10.0
    rim_depth_bad = max(0.0, 25.0 - float(np.percentile(rz, 10))) / 10.0
    low_height = max(0.0, 120.0 - C[2]) / 40.0
    high_height = max(0.0, C[2] - 7000.0) / 500.0
    rows.append(np.asarray([
        floor_depth_bad,
        rim_depth_bad,
        low_height,
        high_height,
    ]))
    return np.concatenate(rows)



def decompose_homography_state(
    H_w2i: np.ndarray,
    logf: float,
    cx: float,
    cy: float,
    sign: float,
) -> tuple[np.ndarray, dict]:
    f = math.exp(float(logf))
    K = K_matrix(f, float(cx), float(cy))
    A = np.linalg.inv(K) @ (float(sign) * H_w2i)
    a1, a2, a3 = A[:, 0], A[:, 1], A[:, 2]
    scale = 2.0 / max(np.linalg.norm(a1) + np.linalg.norm(a2), 1e-12)
    r1 = scale * a1
    r2 = scale * a2
    t = scale * a3
    r3 = np.cross(r1, r2)
    R0 = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(R0)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    C = -R.T @ t
    rv, _ = cv2.Rodrigues(R)
    x = np.r_[rv.ravel(), C, float(logf), float(cx), float(cy), 0.0, 0.0]
    qa = {
        "orthogonality": float(np.dot(r1, r2)),
        "norm_delta": float(np.linalg.norm(r1) - np.linalg.norm(r2)),
        "scale": float(scale),
    }
    return x, qa


def homography_manifold_roots(
    H_w2i: np.ndarray,
    floor_world: np.ndarray,
    floor_image: np.ndarray,
    rim_box: np.ndarray,
    rim_x: float,
) -> list[dict]:
    qlo = np.asarray([math.log(140.0), -500.0, -500.0], dtype=np.float64)
    qhi = np.asarray([math.log(9000.0), 1460.0, 1040.0], dtype=np.float64)
    starts = []
    for f0 in (300.0, 500.0, 800.0, 1200.0, 2000.0, 3500.0):
        for dx, dy in ((0,0), (-160,0), (160,0), (0,-120), (0,120)):
            starts.append(np.asarray([math.log(f0), W/2.0+dx, H/2.0+dy], dtype=np.float64))

    rows = []
    for sign in (1.0, -1.0):
        def residual(q):
            x, hqa = decompose_homography_state(H_w2i, q[0], q[1], q[2], sign)
            uv, z, _, _ = project_points(x, floor_world)
            ru, rz, _, _ = project_points(x, rim_world(rim_x, 360))
            if not np.isfinite(uv).all() or not np.isfinite(ru).all():
                return np.full(floor_world.shape[0]*2 + 12, 1e4, dtype=np.float64)
            pred_box = np.asarray([
                ru[:,0].min(), ru[:,1].min(), ru[:,0].max(), ru[:,1].max()
            ], dtype=np.float64)
            C = x[3:6]
            return np.r_[
                ((uv-floor_image)/1.5).ravel(),
                (pred_box-rim_box)/1.5,
                25.0*hqa["orthogonality"],
                25.0*hqa["norm_delta"],
                max(0.0, 20.0-float(np.percentile(z,10)))/4.0,
                max(0.0, 20.0-float(np.percentile(rz,10)))/4.0,
                max(0.0, 100.0-float(C[2]))/30.0,
                max(0.0, float(C[2])-9000.0)/300.0,
                (q[1]-W/2.0)/450.0,
                (q[2]-H/2.0)/350.0,
            ]

        for q0 in starts:
            try:
                fit=least_squares(
                    residual,
                    np.minimum(np.maximum(q0,qlo+1e-7),qhi-1e-7),
                    bounds=(qlo,qhi),
                    loss="soft_l1",
                    f_scale=1.0,
                    x_scale="jac",
                    max_nfev=800,
                )
            except Exception:
                continue
            x,hqa=decompose_homography_state(H_w2i,fit.x[0],fit.x[1],fit.x[2],sign)
            uv,z,R,_=project_points(x,floor_world)
            ru,rz,_,_=project_points(x,rim_world(rim_x,720))
            floor_e=np.linalg.norm(uv-floor_image,axis=1)
            pred_box=np.asarray([ru[:,0].min(),ru[:,1].min(),ru[:,0].max(),ru[:,1].max()])
            rim_e=np.abs(pred_box-rim_box)
            C=x[3:6]
            valid=bool(
                np.isfinite(C).all()
                and np.percentile(z,10)>10
                and np.percentile(rz,10)>10
                and 100<C[2]<9000
            )
            rows.append({
                "rim_x":float(rim_x),
                "x":x.copy(),
                "cost":float(fit.cost),
                "valid":valid,
                "floor_median_px":float(np.median(floor_e)),
                "floor_p95_px":float(np.percentile(floor_e,95)),
                "rim_bbox_p95_px":float(np.percentile(rim_e,95)),
                "R":R.copy(),
                "C":C.copy(),
                "K":K_matrix(math.exp(x[6]),x[7],x[8]),
                "dist":np.zeros(5,dtype=np.float64),
                "center_source":"homography_manifold_full_rim",
                "homography_qa":hqa,
            })
    return rows

def solve_camera(
    label: str,
    court: dict,
    rim_box: np.ndarray,
    H_i2w_override: np.ndarray | None = None,
):
    xy, cf = court["xy"], court["conf"]
    raw_sel = (cf >= 0.45) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if int(raw_sel.sum()) < 6:
        raise RuntimeError(f"{label}: only {int(raw_sel.sum())} reliable v1.3 court landmarks")

    sel = raw_sel.copy()

    if H_i2w_override is not None:
        # The frozen v1.3 calibrator is authoritative for the floor plane.  Do
        # not re-fit that accepted homography from one noisy freeze frame.
        # Instead sample the accepted metric plane and ask the 3-D solver only
        # to recover a physical camera that reproduces it while also explaining
        # the non-coplanar regulation rim.
        H_i2w = np.asarray(H_i2w_override, dtype=np.float64).reshape(3, 3)
        H_w2i = np.linalg.inv(H_i2w)
        H_w2i = H_w2i / H_w2i[2, 2]
        projected = cv2.perspectiveTransform(
            COURT_VERTICES.astype(np.float64).reshape(-1, 1, 2),
            H_w2i,
        ).reshape(-1, 2)
        visible = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= -40.0)
            & (projected[:, 0] <= W + 40.0)
            & (projected[:, 1] >= -40.0)
            & (projected[:, 1] <= H + 40.0)
        )
        if int(visible.sum()) < 6:
            raise RuntimeError(
                f"{label}: frozen v1.3 homography exposes only "
                f"{int(visible.sum())} usable metric court anchors"
            )
        seed_world_xy = COURT_VERTICES[visible].astype(np.float64)
        floor_image = projected[visible].astype(np.float64)
        floor_world = np.column_stack([
            seed_world_xy,
            np.zeros(int(visible.sum())),
        ]).astype(np.float64)
        seeds = homography_seeds(seed_world_xy, floor_image)
    else:
        # Diagnostic fallback for callers without a frozen v1.3 homography.
        src = xy[raw_sel].astype(np.float64)
        dst = COURT_VERTICES[raw_sel].astype(np.float64)
        H_i2w, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
        if H_i2w is None or inlier is None:
            raise RuntimeError(f"{label}: v1.3 court RANSAC failed")
        keep_local = inlier.ravel().astype(bool)
        raw_ids = np.where(raw_sel)[0]
        keep_ids = raw_ids[keep_local]
        if len(keep_ids) < 6:
            raise RuntimeError(f"{label}: only {len(keep_ids)} v1.3 court RANSAC inliers")
        sel = np.zeros(len(xy), dtype=bool)
        sel[keep_ids] = True
        seed_world_xy = COURT_VERTICES[sel].astype(np.float64)
        floor_world = np.column_stack([
            seed_world_xy,
            np.zeros(int(sel.sum())),
        ]).astype(np.float64)
        floor_image = xy[sel].astype(np.float64)
        H_w2i, _ = cv2.findHomography(seed_world_xy, floor_image, 0)
        if H_w2i is None:
            raise RuntimeError(f"{label}: v1.3 world-to-image homography failed")
        H_w2i = H_w2i / H_w2i[2, 2]
        seeds = homography_seeds(seed_world_xy, floor_image)

    if not seeds:
        raise RuntimeError(f"{label}: could not seed projective camera from v1.3 court")

    lo = np.r_[
        [-10.0] * 3,
        [-20000.0, -20000.0, 100.0],
        math.log(140.0),
        -800.0,
        -800.0,
        -0.55,
        -0.35,
    ]
    hi = np.r_[
        [10.0] * 3,
        [22000.0, 22000.0, 9000.0],
        math.log(9000.0),
        1760.0,
        1340.0,
        0.55,
        0.35,
    ]

    roots = []

    # The primary solver lives on the exact court-homography manifold: the
    # planar v1.3 solution supplies the projective state, while the regulation
    # 3-D rim selects the remaining intrinsic/gauge degree of freedom. This is
    # substantially better conditioned than allowing R,t,K to drift
    # independently while trying to preserve the same plane.
    for rim_x in RIM_XS:
        for row in homography_manifold_roots(
            H_w2i, floor_world, floor_image, rim_box, rim_x
        ):
            row["visible_landmarks"] = int(sel.sum())
            row["raw_visible_landmarks"] = int(raw_sel.sum())
            row["court_ransac_inliers"] = int(sel.sum())
            roots.append(row)

    # For the three cameras already proved metrically in the earlier Adams/Jazz
    # work, do not throw away that evidence. Lock only the physical optical
    # centre and re-solve the current v1.3 event optical state. This is exactly
    # the PTZ/focal/crop hierarchy established by the old hard-gated proofs.
    accepted = label in _ACCEPTED_BASKET_LOCAL_CENTRES
    if accepted:
        for rim_x in RIM_XS:
            for C_fixed in accepted_center_candidates(label, rim_x):
                for seed in seeds:
                    q0 = np.r_[seed[:3], seed[6:]]
                    qlo = np.r_[lo[:3], lo[6:]]
                    qhi = np.r_[hi[:3], hi[6:]]

                    def expand(q):
                        return np.r_[q[:3], C_fixed, q[3:]]

                    try:
                        fit = least_squares(
                            lambda q: _camera_residual(
                                expand(q), floor_world, floor_image, rim_box, rim_x
                            ),
                            np.minimum(np.maximum(q0, qlo + 1e-7), qhi - 1e-7),
                            bounds=(qlo, qhi),
                            loss="soft_l1",
                            f_scale=1.0,
                            x_scale="jac",
                            max_nfev=1200,
                        )
                    except Exception:
                        continue
                    xfit = expand(fit.x)
                    uv, z, R, _ = project_points(xfit, floor_world)
                    ru, rz, _, _ = project_points(xfit, rim_world(rim_x, 720))
                    floor_e = np.linalg.norm(uv - floor_image, axis=1)
                    pred_box = np.asarray([
                        ru[:, 0].min(), ru[:, 1].min(),
                        ru[:, 0].max(), ru[:, 1].max(),
                    ])
                    rim_e = np.abs(pred_box - rim_box)
                    valid = bool(
                        np.percentile(z, 10) > 10
                        and np.percentile(rz, 10) > 10
                        and 100 < C_fixed[2] < 9000
                    )
                    roots.append({
                        "rim_x": float(rim_x),
                        "x": xfit.copy(),
                        "cost": float(fit.cost),
                        "valid": valid,
                        "floor_median_px": float(np.median(floor_e)),
                        "floor_p95_px": float(np.percentile(floor_e, 95)),
                        "rim_bbox_p95_px": float(np.percentile(rim_e, 95)),
                        "visible_landmarks": int(sel.sum()),
                        "raw_visible_landmarks": int(raw_sel.sum()),
                        "court_ransac_inliers": int(sel.sum()),
                        "R": R.copy(),
                        "C": C_fixed.copy(),
                        "K": K_matrix(math.exp(xfit[6]), xfit[7], xfit[8]),
                        "dist": np.asarray([xfit[9], xfit[10], 0.0, 0.0, 0.0], dtype=np.float64),
                        "center_source": "accepted_game_metric_center",
                    })

    # New camera families still get a true free-centre solve.
    if not accepted:
      for rim_x in RIM_XS:
        for seed in seeds:
            s = np.minimum(np.maximum(seed, lo + 1e-7), hi - 1e-7)
            try:
                fit = least_squares(
                    lambda q: _camera_residual(q, floor_world, floor_image, rim_box, rim_x),
                    s,
                    bounds=(lo, hi),
                    loss="soft_l1",
                    f_scale=1.0,
                    x_scale="jac",
                    max_nfev=1200,
                )
            except Exception:
                continue

            uv, z, R, _ = project_points(fit.x, floor_world)
            ru, rz, _, _ = project_points(fit.x, rim_world(rim_x, 720))
            floor_e = np.linalg.norm(uv - floor_image, axis=1)
            pred_box = np.asarray([
                ru[:, 0].min(),
                ru[:, 1].min(),
                ru[:, 0].max(),
                ru[:, 1].max(),
            ])
            rim_e = np.abs(pred_box - rim_box)
            C = fit.x[3:6]
            valid = bool(
                np.percentile(z, 10) > 10
                and np.percentile(rz, 10) > 10
                and 100 < C[2] < 9000
            )
            roots.append({
                "rim_x": float(rim_x),
                "x": fit.x.copy(),
                "cost": float(fit.cost),
                "valid": valid,
                "floor_median_px": float(np.median(floor_e)),
                "floor_p95_px": float(np.percentile(floor_e, 95)),
                "rim_bbox_p95_px": float(np.percentile(rim_e, 95)),
                "visible_landmarks": int(sel.sum()),
                "raw_visible_landmarks": int(raw_sel.sum()),
                "court_ransac_inliers": int(sel.sum()),
                "R": R.copy(),
                "C": C.copy(),
                "K": K_matrix(math.exp(fit.x[6]), fit.x[7], fit.x[8]),
                "dist": np.asarray([fit.x[9], fit.x[10], 0.0, 0.0, 0.0], dtype=np.float64),
                "center_source": "v1.3_free_center_solve",
            })

    roots = [r for r in roots if r["valid"]]
    if not roots:
        raise RuntimeError(f"{label}: no physical camera root")

    roots.sort(key=lambda r: (
        r["floor_p95_px"] + 0.70 * r["rim_bbox_p95_px"],
        0 if r.get("center_source") == "accepted_game_metric_center" else 1,
        r["cost"],
    ))
    best = roots[0]
    if best["floor_p95_px"] > 18.0 or best["rim_bbox_p95_px"] > 18.0:
        raise RuntimeError(
            f"{label}: camera gate failed "
            f"floor={best['floor_p95_px']:.2f}px rim={best['rim_bbox_p95_px']:.2f}px"
        )
    return best, roots[:8]


def draw_camera_qa(
    image: np.ndarray,
    court: dict,
    rim_box: np.ndarray,
    cam: dict,
    out: Path,
) -> None:
    im = image.copy()
    xy, cf = court["xy"], court["conf"]
    sel = (cf >= 0.45) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    P = np.column_stack([COURT_VERTICES[sel], np.zeros(int(sel.sum()))])
    uv, _, _, _ = project_points(cam["x"], P)
    for observed, predicted in zip(xy[sel], uv):
        a = tuple(np.rint(observed).astype(int))
        b = tuple(np.rint(predicted).astype(int))
        cv2.circle(im, a, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(im, b, 4, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(im, a, b, (255, 255, 0), 1, cv2.LINE_AA)

    ru, _, _, _ = project_points(cam["x"], rim_world(cam["rim_x"], 720))
    cv2.polylines(
        im,
        [np.rint(ru).astype(np.int32)],
        True,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    x1, y1, x2, y2 = np.rint(rim_box).astype(int)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(out), im)
