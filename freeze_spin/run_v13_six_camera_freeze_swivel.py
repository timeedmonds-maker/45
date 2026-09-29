from __future__ import annotations

"""HoopVision v1.3 downstream six-camera Adams dunk freeze-swivel.

Pipeline:
official event-489 feeds -> v1.3 court/rim perception -> joint metric camera
states -> exact physical-state search -> metric multi-view depth -> source-pixel
free-view orbit -> deterministic UHD delivery.

This file never modifies the frozen HoopVision v1.3 release.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from freeze_spin.prepare_v32_layered_scene import decode_indices, find_clip, safe_label
from freeze_spin.v13_camera import (
    RIM_Y,
    RIM_Z,
    COURT_VERTICES,
    best_rim,
    court_infer,
    detections,
    draw_camera_qa,
    make_detector,
    solve_camera,
)
from freeze_spin.v13_render import (
    build_metric_clouds,
    encode_product,
    render_orbit,
)
from freeze_spin.v13_sync import analyze_candidates

CAMERAS = (
    "Broadcast",
    "In Arena",
    "Left Slash",
    "Right Slash",
    "Left Above Rim",
    "Right Above Rim",
)
NOMINAL = {
    "Broadcast": 276,
    "In Arena": 272,
    "Left Slash": 266,
    "Right Slash": 248,
    "Left Above Rim": 259,
    "Right Above Rim": 257,
}
OFFSETS = tuple(range(-3, 4))


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--detector-onnx", type=Path, required=True)
    ap.add_argument("--object-eval-src", type=Path, required=True)
    ap.add_argument("--court-model", type=Path, required=True)
    ap.add_argument("--nbacv-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-degree", type=float, default=25.0)
    ap.add_argument("--orbit-frames", type=int, default=61)
    ap.add_argument("--tokens", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    court_model = YOLO(str(args.court_model))
    detector = make_detector(args.detector_onnx, args.object_eval_src)

    # Use the exact frozen v1.3 court calibration implementation rather than a
    # one-frame approximation.  This preserves its temporal pooling, RANSAC,
    # plausibility gates and scale-outlier rejection.
    sys.path.insert(0, str(args.nbacv_src))
    import nbacv.court as nbacv_court
    nbacv_court.VERTICES_CM = COURT_VERTICES.copy()

    clips = {}
    decoded = {}
    nominal_images = {}
    court_obs = {}
    nominal_detections = {}
    cameras = {}
    camera_roots = {}

    for label in CAMERAS:
        clip = find_clip(args.clips_dir, label)
        clips[label] = clip
        indices = [NOMINAL[label] + off for off in OFFSETS]
        decoded[label] = decode_indices(clip, indices)
        nominal = decoded[label][NOMINAL[label]]
        nominal_images[label] = nominal

        # Run the genuine frozen v1.3 calibration on a local real-frame
        # temporal window.  The accepted floor homography is then promoted to
        # a full 3-D camera using the regulation rim as non-coplanar evidence.
        kp_window = {}
        good_size = 640
        for frame_index in indices:
            fr = decoded[label][frame_index]
            kp = None
            for size in [good_size] + [s for s in (640, 960, 1280) if s != good_size]:
                cand = nbacv_court._court_infer(court_model, fr, size, "cpu")
                if cand is not None and int((cand[1] >= 0.5).sum()) >= 6:
                    kp = cand
                    good_size = size
                    break
                if kp is None:
                    kp = cand
            kp_window[frame_index] = kp

        calibration = nbacv_court.calibrate_video(
            kp_window,
            window=2,
            frame_hw=(540, 960),
        )
        floor_rec = calibration.get(NOMINAL[label], {})
        if floor_rec.get("H") is None:
            raise RuntimeError(
                f"{label}: frozen v1.3 floor calibration failed at nominal "
                f"frame {NOMINAL[label]}: {floor_rec}"
            )

        court = court_infer(court_model, nominal)
        if court is None:
            raise RuntimeError(f"{label}: no v1.3 court detection")
        court["v13_floor_calibration"] = floor_rec
        court_obs[label] = court

        det_rows = detections(detector, nominal)
        nominal_detections[label] = det_rows
        rim = best_rim(det_rows)
        if rim is None:
            raise RuntimeError(f"{label}: v1.3 RF-DETR did not detect rim")

        cam, roots = solve_camera(
            label,
            court,
            rim["xyxy"],
            H_i2w_override=np.asarray(floor_rec["H"], dtype=np.float64),
        )
        cam["label"] = label
        cam["rim_box"] = rim["xyxy"]
        cameras[label] = cam
        camera_roots[label] = roots

        draw_camera_qa(
            nominal,
            court,
            rim["xyxy"],
            cam,
            args.out / f"camera_qa_{safe_label(label)}.png",
        )
        print(
            "V13_CAMERA",
            label,
            json.dumps({
                "court_landmarks": court["visible"],
                "floor_p95_px": cam["floor_p95_px"],
                "rim_bbox_p95_px": cam["rim_bbox_p95_px"],
                "camera_center_cm": cam["C"].tolist(),
                "rim_x_world_cm": cam["rim_x"],
            }, sort_keys=True),
            flush=True,
        )

    # All six views must describe the same physical basket.  Individual planar
    # roots can mirror to the opposite end; use six-view consensus to select the
    # shared basket before any dynamic evidence enters the reconstruction.
    basket_counts = {}
    for cam in cameras.values():
        basket_counts[cam["rim_x"]] = basket_counts.get(cam["rim_x"], 0) + 1
    majority_basket = max(basket_counts, key=basket_counts.get)

    for label in CAMERAS:
        if cameras[label]["rim_x"] == majority_basket:
            continue
        alternatives = [
            root for root in camera_roots[label]
            if root["rim_x"] == majority_basket
        ]
        if not alternatives:
            raise RuntimeError(
                f"{label}: no physical root on six-view basket consensus "
                f"{majority_basket}"
            )
        cameras[label] = alternatives[0]
        cameras[label]["label"] = label
        cameras[label]["rim_box"] = best_rim(
            nominal_detections[label]
        )["xyxy"]

    # Search only real native frames.  Each candidate is scene-registered back
    # to its nominal optical state, so moving PTZ/zoom does not masquerade as a
    # different basketball instant.  Pose + ball epipolar consistency chooses
    # one physical state across the six views.
    best_state, observations, registrations, pair_tables = analyze_candidates(
        CAMERAS,
        OFFSETS,
        NOMINAL,
        decoded,
        nominal_images,
        cameras,
        detector,
        detections,
    )
    sync_score, offsets = best_state
    print(
        "V13_EXACT_STATE",
        json.dumps({"score": sync_score, "offsets": offsets}, sort_keys=True),
        flush=True,
    )

    exact_undistorted = {}
    for label in CAMERAS:
        off = offsets[label]
        raw = decoded[label][NOMINAL[label] + off]
        Hm = registrations[label][off][0]
        if Hm is None:
            raise RuntimeError(f"{label}: chosen exact-state frame has no static registration")
        registered = (
            cv2.warpPerspective(
                raw,
                Hm,
                (960, 540),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if off != 0
            else raw.copy()
        )
        cam = cameras[label]
        exact = cv2.undistort(
            registered,
            cam["K"],
            cam["dist"],
            None,
            cam["K"],
        )
        exact_undistorted[label] = exact
        cv2.imwrite(
            str(args.out / f"exact_{safe_label(label)}.png"),
            exact,
        )

    clouds, depth_qa = build_metric_clouds(
        CAMERAS,
        exact_undistorted,
        cameras,
        args.out,
        args.tokens,
    )

    anchor = "Broadcast"
    pivot = np.asarray(
        [majority_basket, RIM_Y, RIM_Z],
        dtype=np.float64,
    )
    orbit_qa = render_orbit(
        CAMERAS,
        cameras,
        clouds,
        exact_undistorted,
        anchor,
        pivot,
        args.out,
        args.max_degree,
        args.orbit_frames,
    )

    freeze_index = NOMINAL[anchor] + offsets[anchor]
    native, uhd, probe = encode_product(
        clips[anchor],
        freeze_index,
        cameras[anchor],
        args.out / "orbit",
        args.orbit_frames,
        args.out,
    )

    report = {
        "status": "V13_SIX_CAMERA_FREEZE_SWIVEL_RENDERED",
        "game_id": "0022500301",
        "event_id": 489,
        "player": "Steven Adams",
        "basketball_moment": "Steven Adams dunk vs Utah immediately after his block",
        "frozen_engine": "HoopVision release/v1.3 downstream consumer; upstream v1.3 unchanged",
        "camera_set": list(CAMERAS),
        "camera_count": len(CAMERAS),
        "shared_basket_x_cm": float(majority_basket),
        "cameras": {
            label: {
                "nominal_frame": NOMINAL[label],
                "selected_offset": offsets[label],
                "selected_frame": NOMINAL[label] + offsets[label],
                "court_visible": court_obs[label]["visible"],
                "court_imgsz": court_obs[label]["imgsz"],
                "floor_median_px": cameras[label]["floor_median_px"],
                "floor_p95_px": cameras[label]["floor_p95_px"],
                "rim_bbox_p95_px": cameras[label]["rim_bbox_p95_px"],
                "rim_x_world_cm": cameras[label]["rim_x"],
                "C_world_cm": cameras[label]["C"],
                "K_px": cameras[label]["K"],
                "dist_k1_k2": cameras[label]["dist"][:2],
                "registration_qa": registrations[label][offsets[label]][1],
                "depth_qa": depth_qa[label],
                "action_person_score": observations[label][offsets[label]].get(
                    "person_score"
                ),
            }
            for label in CAMERAS
        },
        "exact_state": {
            "score": float(sync_score),
            "offsets": offsets,
            "search_offsets": list(OFFSETS),
            "policy": (
                "one real native decoded frame per camera; candidate PTZ/zoom "
                "registered to nominal optical state; six-view pose/ball "
                "epipolar consistency chooses the physical instant"
            ),
        },
        "render": {
            "native_resolution": [960, 540],
            "uhd_resolution": [3840, 2160],
            "anchor": anchor,
            "pivot_world_cm": pivot,
            "max_degree": args.max_degree,
            "orbit_frames": args.orbit_frames,
            "path": "0 -> +max -> 0",
            "orbit_qa": orbit_qa,
        },
        "appearance_policy": (
            "official NBA source pixels only; no generated RGB, optical-flow "
            "morph or crossfade; secondary views fill unresolved pixels in "
            "virtual-view angular order"
        ),
        "geometry_policy": (
            "frozen v1.3 33-landmark NBA court + RF-DETR rim -> full metric "
            "camera with focal/principal/distortion refinement; MoGe supplies "
            "shape only and is metrically aligned to the solved NBA floor"
        ),
        "uhd_policy": (
            "all calibration/reconstruction/render QA occurs at native "
            "960x540; deterministic Lanczos presentation master is created "
            "only after native render"
        ),
        "native_video": native.name,
        "uhd_video": uhd.name,
        "probe": probe,
    }

    report_path = args.out / "v13_six_camera_freeze_swivel_report.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps({
            "status": report["status"],
            "offsets": offsets,
            "camera_floor_p95_px": {
                label: report["cameras"][label]["floor_p95_px"]
                for label in CAMERAS
            },
            "depth_heldout_p95_cm": {
                label: depth_qa[label]["heldout_p95_cm"]
                for label in CAMERAS
            },
            "uhd": str(uhd),
            "probe": probe,
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

# Runner trigger: public Actions execute the frozen v1.3 downstream proof.
# Re-triggered from ChatGPT on 2026-09-30 to use the public HoopVision runner.
