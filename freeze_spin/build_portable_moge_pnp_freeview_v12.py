from __future__ import annotations
import cv2
import numpy as np
import torch

def normalized_to_pixel_K(Kn: np.ndarray, w: int, h: int) -> np.ndarray:
    K = np.asarray(Kn, dtype=np.float64).copy()
    K[0, 0] *= w
    K[0, 2] *= w
    K[1, 1] *= h
    K[1, 2] *= h
    return K

def moge_infer(model, image: np.ndarray, tokens: int):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    with torch.inference_mode():
        out = model.infer(t, num_tokens=tokens, use_fp16=False, apply_mask=False)
    depth = out["depth"].cpu().numpy().astype(np.float32)
    points = out["points"].cpu().numpy().astype(np.float32)
    valid = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
    Kn = out["intrinsics"].cpu().numpy().astype(np.float64)
    K = normalized_to_pixel_K(Kn, image.shape[1], image.shape[0])
    return depth, points, valid, K, Kn
