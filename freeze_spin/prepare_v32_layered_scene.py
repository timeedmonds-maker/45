from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np

W, H = 960, 540

def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")

def find_clip(clips_dir: Path, label: str) -> Path:
    token = safe_label(label)
    rows = sorted(Path(clips_dir).glob(f"*_{token}_SOURCE.mp4"))
    if label == "Broadcast":
        rows = [p for p in rows if "Other_Broadcast" not in p.name and "Mobile_Broadcast" not in p.name]
    if len(rows) != 1:
        raise RuntimeError(f"expected one native clip for {label}; got {rows}")
    return rows[0]

def decode_indices(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    need = sorted(set(int(x) for x in indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    for idx in need:
        if idx < 0 or idx >= n:
            raise RuntimeError(f"frame {idx} outside {path.name} frame_count={n}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"decode failed {path.name} frame={idx}")
        if frame.shape[:2] != (H, W):
            raise RuntimeError(f"source resolution changed for {path.name}: {frame.shape[:2]}")
        out[idx] = frame
    cap.release()
    return out
