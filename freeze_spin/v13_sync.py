from __future__ import annotations

import itertools
import math

import cv2
import numpy as np
from ultralytics import YOLO

from freeze_spin.v13_camera import W, H, best_ball, best_rim

ACTION_KP = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def registration_h(candidate: np.ndarray, nominal: np.ndarray):
    if np.array_equal(candidate, nominal):
        return np.eye(3, dtype=np.float64), {"inliers": 999, "p95_px": 0.0}

    sift = cv2.SIFT_create(
        nfeatures=7000,
        contrastThreshold=0.02,
        edgeThreshold=14,
        sigma=1.3,
    )
    kc, dc = sift.detectAndCompute(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY), None)
    kn, dn = sift.detectAndCompute(cv2.cvtColor(nominal, cv2.COLOR_BGR2GRAY), None)
    if dc is None or dn is None:
        return None, {"reason": "no_descriptors"}

    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dc, dn, k=2)
    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    if len(good) < 25:
        return None, {"reason": "too_few_matches", "matches": len(good)}

    p = np.float32([kc[m.queryIdx].pt for m in good])
    q = np.float32([kn[m.trainIdx].pt for m in good])
    Hm, mask = cv2.findHomography(
        p,
        q,
        cv2.RANSAC,
        2.5,
        maxIters=10000,
        confidence=0.999,
    )
    if Hm is None or mask is None or int(mask.sum()) < 20:
        return None, {
            "reason": "homography_fail",
            "matches": len(good),
            "inliers": 0 if mask is None else int(mask.sum()),
        }

    ids = mask.ravel().astype(bool)
    pred = cv2.perspectiveTransform(p[ids, None, :], Hm)[:, 0]
    err = np.linalg.norm(pred - q[ids], axis=1)
    return Hm / Hm[2, 2], {
        "matches": len(good),
        "inliers": int(ids.sum()),
        "p95_px": float(np.percentile(err, 95)),
    }


def map_points(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(
        np.asarray(pts, np.float64).reshape(-1, 1, 2),
        Hm,
    ).reshape(-1, 2)


def pose_observation(
    pose_model: YOLO,
    frame: np.ndarray,
    detector_rows: list[dict],
    rim_box: np.ndarray,
    H_to_nominal: np.ndarray,
    cam: dict,
) -> dict:
    result = pose_model.predict(
        frame,
        imgsz=640,
        device="cpu",
        verbose=False,
        conf=0.16,
    )[0]
    rim_center = 0.5 * (rim_box[:2] + rim_box[2:])
    candidates = []

    if result.boxes is not None and result.keypoints is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        data = result.keypoints.data.cpu().numpy()
        for i, kp in enumerate(data):
            xy = kp[:, :2].astype(np.float64)
            cf = kp[:, 2].astype(np.float64)
            good = cf[list(ACTION_KP)] >= 0.18
            if good.any():
                d = np.linalg.norm(xy[list(ACTION_KP)][good] - rim_center, axis=1)
                joint_near = float(np.min(d))
            else:
                joint_near = 9999.0
            box = boxes[i]
            box_center = 0.5 * (box[:2] + box[2:])
            box_h = max(1.0, box[3] - box[1])
            score = (
                joint_near / max(50.0, box_h)
                + 0.25 * np.linalg.norm(box_center - rim_center) / max(50.0, box_h)
                - 0.15 * float(scores[i])
            )
            candidates.append((score, xy, cf, box, float(scores[i])))

    if not candidates:
        return {
            "xy": np.full((17, 2), np.nan),
            "conf": np.zeros(17),
            "ball": None,
            "person_score": 999.0,
        }

    candidates.sort(key=lambda z: z[0])
    score, xy, cf, box, pconf = candidates[0]
    xy_nom = map_points(H_to_nominal, xy)
    K = cam["K"]
    dist = cam["dist"]
    xy_un = cv2.undistortPoints(
        xy_nom.reshape(-1, 1, 2),
        K,
        dist,
        P=K,
    ).reshape(-1, 2)

    ball = best_ball(detector_rows, rim_box)
    ball_un = None
    if ball is not None:
        center = 0.5 * (ball["xyxy"][:2] + ball["xyxy"][2:])
        center_nom = map_points(H_to_nominal, center.reshape(1, 2))[0]
        ball_un = cv2.undistortPoints(
            center_nom.reshape(1, 1, 2),
            K,
            dist,
            P=K,
        ).reshape(2)

    return {
        "xy": xy_un,
        "conf": cf,
        "ball": ball_un,
        "person_score": float(score),
        "person_box": box.tolist(),
        "person_conf": pconf,
    }


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = map(float, v)
    return np.asarray([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def fundamental(a: dict, b: dict) -> np.ndarray:
    R1, C1, K1 = a["R"], a["C"], a["K"]
    R2, C2, K2 = b["R"], b["C"], b["K"]
    R = R2 @ R1.T
    t = R2 @ (C1 - C2)
    E = skew(t) @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def epi_dist(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    x1 = np.r_[p1, 1.0]
    x2 = np.r_[p2, 1.0]
    l2 = F @ x1
    l1 = F.T @ x2
    d2 = abs(float(x2 @ l2)) / max(math.hypot(l2[0], l2[1]), 1e-9)
    d1 = abs(float(x1 @ l1)) / max(math.hypot(l1[0], l1[1]), 1e-9)
    return 0.5 * (d1 + d2)


def pair_score(cam_a: dict, obs_a: dict, cam_b: dict, obs_b: dict) -> float:
    F = fundamental(cam_a, cam_b)
    ds = []

    for j in ACTION_KP:
        if (
            obs_a["conf"][j] >= 0.22
            and obs_b["conf"][j] >= 0.22
            and np.isfinite(obs_a["xy"][j]).all()
            and np.isfinite(obs_b["xy"][j]).all()
        ):
            ds.append(epi_dist(F, obs_a["xy"][j], obs_b["xy"][j]))

    if obs_a["ball"] is not None and obs_b["ball"] is not None:
        d = epi_dist(F, obs_a["ball"], obs_b["ball"])
        ds.extend([d, d])

    if len(ds) < 5:
        return 150.0 + 15.0 * (5 - len(ds))

    a = np.asarray(ds, dtype=np.float64)
    return float(
        np.median(a)
        + 0.25 * np.percentile(a, 90)
        + 5.0 * max(0, 7 - len(ds))
    )


def choose_exact_state(
    camera_labels: tuple[str, ...],
    offsets: tuple[int, ...],
    cameras: dict[str, dict],
    observations: dict[str, dict[int, dict]],
):
    pair_tables = {}
    for i, a in enumerate(camera_labels):
        for b in camera_labels[i + 1:]:
            table = {}
            for oa in offsets:
                for ob in offsets:
                    table[(oa, ob)] = pair_score(
                        cameras[a],
                        observations[a][oa],
                        cameras[b],
                        observations[b][ob],
                    )
            pair_tables[(a, b)] = table

    best = None
    for combo in itertools.product(offsets, repeat=len(camera_labels)):
        selected = dict(zip(camera_labels, combo))
        score = 0.35 * sum(abs(x) for x in combo)
        for i, a in enumerate(camera_labels):
            for b in camera_labels[i + 1:]:
                score += pair_tables[(a, b)][(selected[a], selected[b])]
        if best is None or score < best[0]:
            best = (score, selected)

    return best, pair_tables


def analyze_candidates(
    camera_labels: tuple[str, ...],
    offsets: tuple[int, ...],
    nominal_frames: dict[str, int],
    decoded: dict[str, dict[int, np.ndarray]],
    nominal_images: dict[str, np.ndarray],
    cameras: dict[str, dict],
    detector,
    detection_fn,
):
    pose_model = YOLO("yolo11n-pose.pt")
    observations = {label: {} for label in camera_labels}
    registrations = {label: {} for label in camera_labels}

    for label in camera_labels:
        for off in offsets:
            frame = decoded[label][nominal_frames[label] + off]
            Hm, qa = registration_h(frame, nominal_images[label])
            registrations[label][off] = (Hm, qa)
            if Hm is None:
                observations[label][off] = {
                    "xy": np.full((17, 2), np.nan),
                    "conf": np.zeros(17),
                    "ball": None,
                    "person_score": 999.0,
                }
                continue

            rows = detection_fn(detector, frame)
            rim = best_rim(rows)
            rim_box = rim["xyxy"] if rim is not None else cameras[label]["rim_box"]
            observations[label][off] = pose_observation(
                pose_model,
                frame,
                rows,
                rim_box,
                Hm,
                cameras[label],
            )

    best, pair_tables = choose_exact_state(
        camera_labels,
        offsets,
        cameras,
        observations,
    )
    return best, observations, registrations, pair_tables
