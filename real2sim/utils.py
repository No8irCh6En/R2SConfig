"""Utility functions for mask operations, visualization, and I/O."""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


def dilate_mask(mask: torch.Tensor, kernel_size: int = 15, iters: int = 3) -> torch.Tensor:
    """Iteratively dilate a binary mask."""
    if iters <= 0 or kernel_size <= 0:
        return mask
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    m = mask[None, None].float()
    for _ in range(iters):
        m = F.conv2d(F.pad(m, (kernel_size // 2,) * 4, mode="replicate"), kernel)
        m = (m > 0.5).float()
    return m[0, 0].bool()


def mask_to_bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) from binary mask."""
    ys, xs = torch.where(mask)
    if len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Intersection-over-Union for two binary masks."""
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return intersection / union.clamp(min=1e-6)


def depth_to_3d(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Back-project depth map to 3D point cloud.

    Args:
        depth: (H, W) depth values
        K: (3, 3) intrinsics

    Returns:
        (H, W, 3) 3D point cloud in camera space
    """
    H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    return torch.stack([X, Y, depth], dim=-1)


def save_scene_json(path: str, results: dict):
    """Export optimization results to JSON for downstream use."""
    import json

    out = {
        "camera": {
            "fx": results.get("fx", None),
            "fy": results.get("fy", None),
            "cx": results.get("cx", None),
            "cy": results.get("cy", None),
            "width": results.get("width", None),
            "height": results.get("height", None),
        },
        "objects": [],
    }
    for obj in results.get("objects", []):
        out["objects"].append(
            {
                "name": obj.get("name", "object"),
                "rotation": obj.get("R", None).tolist() if isinstance(obj.get("R"), torch.Tensor) else obj.get("R"),
                "translation": obj.get("t", None).tolist() if isinstance(obj.get("t"), torch.Tensor) else obj.get("t"),
                "scale": float(obj.get("scale", 1.0)),
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
