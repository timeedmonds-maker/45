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


def solve_camera(label: str, court: dict, rim_box: np.ndarray):
    xy, cf = court["xy"], court["conf"]
    sel = (cf >= 0.45) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if int(sel.sum()) < 6:
        raise RuntimeError(f"{label}: only {int(sel.sum())} reliable v1.3 court landmarks")

    floor_world = np.column_stack([
        COURT_VERTICES[sel],
        np.zeros(int(sel.sum())),
    ]).astype(np.float64)
    floor_image = xy[sel].astype(np.float64)
    seeds = homography_seeds(COURT_VERTICES[sel], floor_image)
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
                    max_nfev=900,
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
                "R": R.copy(),
                "C": C.copy(),
                "K": K_matrix(math.exp(fit.x[6]), fit.x[7], fit.x[8]),
                "dist": np.asarray([fit.x[9], fit.x[10], 0.0, 0.0, 0.0], dtype=np.float64),
            })

    roots = [r for r in roots if r["valid"]]
    if not roots:
        raise RuntimeError(f"{label}: no physical camera root")

    roots.sort(key=lambda r: (
        r["floor_p95_px"] + 0.70 * r["rim_bbox_p95_px"],
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
