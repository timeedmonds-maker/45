from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from scipy.optimize import least_squares

from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer
from freeze_spin.v13_camera import W, H, COURT_L, COURT_W


def floor_grid() -> np.ndarray:
    xs = np.linspace(0.0, COURT_L, 75)
    ys = np.linspace(0.0, COURT_W, 45)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([
        gx.ravel(),
        gy.ravel(),
        np.zeros(gx.size),
    ])


def depth_mapping(depth: np.ndarray, valid: np.ndarray, cam: dict):
    P = floor_grid()
    K, R, C = cam["K"], cam["R"], cam["C"]
    Xc = (R @ (P - C).T).T
    z = Xc[:, 2]
    q = (K @ Xc.T).T
    uv = q[:, :2] / q[:, 2:3]
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    good = (
        (z > 20.0)
        & np.isfinite(uv).all(axis=1)
        & (x >= 3)
        & (x < W - 3)
        & (y >= 3)
        & (y < H - 3)
    )
    x, y, z = x[good], y[good], z[good]
    d = depth[y, x].astype(np.float64)
    ok = valid[y, x] & np.isfinite(d) & (d > 0.02)
    d, z = d[ok], z[ok]
    if len(d) < 30:
        raise RuntimeError(f"{cam['label']}: only {len(d)} metric floor depth anchors")

    ids = np.arange(len(d))
    hold = (ids % 7) == 0
    train = ~hold
    scale = float(np.median(z[train] / np.maximum(d[train], 1e-6)))
    fit = least_squares(
        lambda p: p[0] * d[train] + p[1] - z[train],
        [scale, 0.0],
        loss="soft_l1",
        f_scale=40.0,
        max_nfev=4000,
    )
    pred = fit.x[0] * d + fit.x[1]
    err = np.abs(pred - z)
    qa = {
        "anchors": int(len(d)),
        "scale": float(fit.x[0]),
        "offset_cm": float(fit.x[1]),
        "heldout_median_cm": float(np.median(err[hold])),
        "heldout_p95_cm": float(np.percentile(err[hold], 95)),
        "all_p95_cm": float(np.percentile(err, 95)),
    }
    return fit.x, qa


def build_cloud(
    image: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    mapping: np.ndarray,
    cam: dict,
    stride: int,
):
    z = mapping[0] * depth.astype(np.float64) + mapping[1]
    yy, xx = np.indices((H, W))
    sample = ((xx % stride) == 0) & ((yy % stride) == 0)
    ok = sample & valid & np.isfinite(z) & (z > 20.0) & (z < 15000.0)
    ys, xs = np.where(ok)
    zz = z[ys, xs]

    K, R, C = cam["K"], cam["R"], cam["C"]
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = np.column_stack([xn * zz, yn * zz, zz])
    Xw = (R.T @ Xc.T).T + C
    return Xw.astype(np.float32), image[ys, xs].copy()


def orbit_pose(
    C0: np.ndarray,
    R0: np.ndarray,
    pivot: np.ndarray,
    degree: float,
):
    t = math.radians(float(degree))
    Q = np.asarray([
        [math.cos(t), -math.sin(t), 0.0],
        [math.sin(t),  math.cos(t), 0.0],
        [0.0,          0.0,         1.0],
    ], dtype=np.float64)
    C = pivot + Q @ (C0 - pivot)
    R = R0 @ Q.T
    return R, C


def raster_cloud(cloud, K: np.ndarray, R: np.ndarray, C: np.ndarray, radius: int = 1):
    X, colours = cloud
    Xf = X.astype(np.float64)
    Xc = (R @ (Xf - C).T).T
    z = Xc[:, 2]
    q = (K @ Xc.T).T
    uv = q[:, :2] / q[:, 2:3]
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)

    ok = (
        np.isfinite(uv).all(axis=1)
        & (z > 20.0)
        & (u >= 0)
        & (u < W)
        & (v >= 0)
        & (v < H)
    )
    ids = np.where(ok)[0]
    image = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    zbuf = np.full(H * W, np.inf, np.float32)

    if len(ids):
        pix = v[ids] * W + u[ids]
        np.minimum.at(zbuf, pix, z[ids].astype(np.float32))
        winners = ids[z[ids] <= zbuf[pix] + 1e-4]
        image[v[winners], u[winners]] = colours[winners]
        mask[v[winners], u[winners]] = 255

    for _ in range(int(radius)):
        base_i, base_m = image.copy(), mask.copy()
        holes = mask == 0
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            shifted_i = np.roll(np.roll(base_i, dy, 0), dx, 1)
            shifted_m = np.roll(np.roll(base_m, dy, 0), dx, 1)
            take = holes & (mask == 0) & (shifted_m > 0)
            image[take] = shifted_i[take]
            mask[take] = 255

    return image, mask


def angular_order(cameras: dict[str, dict], pivot: np.ndarray, target_C: np.ndarray):
    tv = target_C - pivot
    tv = tv / max(np.linalg.norm(tv), 1e-9)
    rows = []
    for label, cam in cameras.items():
        sv = cam["C"] - pivot
        sv = sv / max(np.linalg.norm(sv), 1e-9)
        angle = math.degrees(
            math.acos(float(np.clip(np.dot(tv, sv), -1.0, 1.0)))
        )
        rows.append((angle, label))
    return [x[1] for x in sorted(rows)]


def composite(rendered: dict, order: list[str]):
    first = order[0]
    out = rendered[first][0].copy()
    mask = rendered[first][1] > 0
    for label in order[1:]:
        image, m = rendered[label]
        take = (~mask) & (m > 0)
        out[take] = image[take]
        mask |= m > 0
    return out, mask


def build_metric_clouds(
    camera_labels: tuple[str, ...],
    exact_images: dict[str, np.ndarray],
    cameras: dict[str, dict],
    out: Path,
    tokens: int,
):
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    clouds = {}
    qa = {}

    for label in camera_labels:
        depth, _, valid, _, _ = moge_infer(model, exact_images[label], tokens)
        mapping, depth_qa = depth_mapping(depth, valid, cameras[label])
        qa[label] = depth_qa
        stride = 1 if label == "Broadcast" else 2
        clouds[label] = build_cloud(
            exact_images[label],
            depth,
            valid,
            mapping,
            cameras[label],
            stride,
        )

        vis = np.zeros((H, W), np.uint8)
        good = valid & np.isfinite(depth) & (depth > 0)
        if good.any():
            lo, hi = np.percentile(depth[good], [2, 98])
            vis[good] = np.clip(
                (depth[good] - lo) * 255.0 / max(float(hi - lo), 1e-6),
                0,
                255,
            ).astype(np.uint8)
        cv2.imwrite(str(out / f"depth_{_safe(label)}.png"), vis)
        print("DEPTH", label, json.dumps(depth_qa, sort_keys=True), flush=True)

    return clouds, qa


def render_orbit(
    camera_labels: tuple[str, ...],
    cameras: dict[str, dict],
    clouds: dict,
    exact_images: dict[str, np.ndarray],
    anchor: str,
    pivot: np.ndarray,
    out: Path,
    max_degree: float,
    frames: int,
):
    stage = out / "orbit"
    stage.mkdir(parents=True, exist_ok=True)
    A = cameras[anchor]
    Kout = A["K"]
    rows = []

    for i in range(frames):
        phase = i / max(frames - 1, 1)
        degree = float(max_degree * math.sin(math.pi * phase))
        Rn, Cn = orbit_pose(A["C"], A["R"], pivot, degree)
        rendered = {
            label: raster_cloud(clouds[label], Kout, Rn, Cn, radius=1)
            for label in camera_labels
        }
        order = angular_order(cameras, pivot, Cn)
        image, mask = composite(rendered, order)

        if i == 0 or i == frames - 1:
            image = exact_images[anchor].copy()
            mask = np.ones((H, W), dtype=bool)

        cv2.imwrite(str(stage / f"orbit_{i:04d}.png"), image)
        rows.append({
            "frame": int(i),
            "degree": degree,
            "source_order": order,
            "resolved_fraction": float(np.mean(mask)),
        })

    return rows


def encode_product(
    anchor_clip: Path,
    freeze_index: int,
    anchor_camera: dict,
    orbit_dir: Path,
    orbit_frames: int,
    out: Path,
):
    from freeze_spin.prepare_v32_layered_scene import decode_indices

    pre = list(range(max(0, freeze_index - 24), freeze_index))
    post = list(range(freeze_index + 1, freeze_index + 25))
    extra = decode_indices(anchor_clip, pre + post)

    seq = out / "sequence"
    seq.mkdir(parents=True, exist_ok=True)
    n = 0

    for idx in pre:
        frame = cv2.undistort(
            extra[idx],
            anchor_camera["K"],
            anchor_camera["dist"],
            None,
            anchor_camera["K"],
        )
        cv2.imwrite(str(seq / f"frame_{n:04d}.png"), frame)
        n += 1

    for i in range(orbit_frames):
        frame = cv2.imread(str(orbit_dir / f"orbit_{i:04d}.png"))
        if frame is None:
            raise RuntimeError(f"missing orbit frame {i}")
        cv2.imwrite(str(seq / f"frame_{n:04d}.png"), frame)
        n += 1

    for idx in post:
        frame = cv2.undistort(
            extra[idx],
            anchor_camera["K"],
            anchor_camera["dist"],
            None,
            anchor_camera["K"],
        )
        cv2.imwrite(str(seq / f"frame_{n:04d}.png"), frame)
        n += 1

    native = out / "adams_dunk_v13_six_camera_freeze_swivel_native.mp4"
    uhd = out / "adams_dunk_v13_six_camera_freeze_swivel_UHD.mp4"

    _run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(seq / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "15",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(native),
    ])
    _run([
        "ffmpeg", "-y",
        "-i", str(native),
        "-vf", "scale=3840:2160:flags=lanczos,unsharp=5:5:0.35:3:3:0.0,fps=30",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "5.1",
        "-preset", "medium",
        "-crf", "15",
        "-maxrate", "38M",
        "-bufsize", "76M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(uhd),
    ])

    probe = json.loads(
        subprocess.check_output([
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(uhd),
        ], text=True)
    )
    return native, uhd, probe


def _run(cmd: list[str], timeout: int = 1800):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")
