"""Mask-asymmetry-driven yaw initialization for multi-start pose optimization.

For objects with an asymmetric feature (mug handle, teapot spout, kettle handle,
shovel head, etc.), uniform yaw multi-start is wasteful AND can miss the right
basin entirely when the silhouette IoU loss has a degenerate min that prefers
"feature hidden behind body".

This module solves it by READING THE ASYMMETRY OUT OF THE MASKS directly:

  1. Mesh-local handle direction (h_mesh): 2D PCA of the mesh's horizontal
     (xy) projection. For mug-shaped objects, the cup body is rotationally
     symmetric so eigenvalues of the projected cov are equal — the handle is
     the only thing that breaks the symmetry. Major eigenvector → handle axis.
     180° flip resolved by mass asymmetry (which side has more vertices).

  2. Mask handle direction (d_mask): same 2D PCA on mask pixels. Major axis
     of mask = elongation direction = handle direction (cup body contributes
     near-zero eigenvalue, handle dominates).

  3. Per-view yaw candidate: search θ ∈ [0, 360°) for the yaw that, after
     applying to h_mesh and projecting through this view's camera, produces
     a 2D image direction parallel to d_mask. Each view gives 1-2 best yaws
     (plus optional 180° flip).

  4. Pool across views, dedupe (within 15°), return as multi-start seeds.

Completely symmetric objects (sphere, bottle without handle) produce noise-
dominated h_mesh and unstable d_mask. Use uniform multi-start for those.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch


def extract_mesh_handle_dir_xy(verts: torch.Tensor) -> np.ndarray:
    """Mesh-local handle direction in xy plane (mesh assumed z-up after M_LOAD).

    Returns (dx, dy) unit vector pointing TOWARDS handle (180° flip resolved
    by mass asymmetry — the side with more vertices beyond the mean).
    """
    v = verts.detach().cpu().numpy()
    xy = v[:, :2]
    centered = xy - xy.mean(0)
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)              # ascending
    major = eigvecs[:, -1]                              # (2,) unit vector
    proj = centered @ major
    if float((proj > 0).sum()) < float((proj < 0).sum()):
        major = -major
    return major


def extract_mask_handle_dir_2d(mask: np.ndarray) -> Optional[np.ndarray]:
    """Per-mask 2D handle direction in image pixel space (u right, v down).

    Returns (du, dv) unit vector or None if mask is too small.
    """
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - pts.mean(0)
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # If eigenvalues are nearly equal, mask is rotationally symmetric → unreliable
    if eigvals[0] / max(eigvals[1], 1e-9) > 0.85:
        return None
    major = eigvecs[:, -1]
    proj = centered @ major
    if float((proj > 0).sum()) < float((proj < 0).sum()):
        major = -major
    return major


def _project_world_dir_to_image(h_world: np.ndarray, t_world: np.ndarray,
                                R_view: np.ndarray, T_view: np.ndarray,
                                fx: float, fy: float, cx: float, cy: float,
                                eps: float = 1e-3) -> Optional[np.ndarray]:
    """Finite-difference of PT3D pinhole projection along a world-frame direction.

    PT3D: P_view = P_world @ R_view + T_view;  u = -fx · P_view[0]/P_view[2] + cx
    Returns 2D unit vector in image (image x right, image y down).
    """
    P0 = t_world @ R_view + T_view
    P1 = (t_world + eps * h_world) @ R_view + T_view
    if P0[2] <= 1e-6 or P1[2] <= 1e-6:
        return None
    u0 = -fx * P0[0] / P0[2] + cx
    v0 = -fy * P0[1] / P0[2] + cy
    u1 = -fx * P1[0] / P1[2] + cx
    v1 = -fy * P1[1] / P1[2] + cy
    d = np.array([u1 - u0, v1 - v0])
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return d / n


def _yaw_handle_in_image(theta_rad: float, h_mesh_xy: np.ndarray,
                         t_world: np.ndarray, R_view: np.ndarray, T_view: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float):
    """Mesh handle direction in image after world yaw θ around +z.

    PT3D row-vec yaw: world = mesh @ R_yaw, R_yaw = [[c,s,0],[-s,c,0],[0,0,1]].
    For h_mesh = (a, b, 0):  h_world = (a·c - b·s, a·s + b·c, 0).
    """
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    a, b = float(h_mesh_xy[0]), float(h_mesh_xy[1])
    h_world = np.array([a * c - b * s, a * s + b * c, 0.0])
    return _project_world_dir_to_image(h_world, t_world, R_view, T_view, fx, fy, cx, cy)


def propose_yaw_candidates(
    views: List[Dict],
    mesh_verts: torch.Tensor,
    init_t_world: torch.Tensor,
    n_grid: int = 360,
    include_flip: bool = True,
    dedupe_deg: float = 15.0,
    verbose: bool = True,
) -> List[Dict]:
    """Returns list of {yaw_deg, score, source} dicts. Use as multi-start seeds.

    `include_flip`: also add 180°-flip of each candidate (PCA 180° ambig insurance).
    `dedupe_deg`: yaws within this circular distance are clustered, keep best.
    """
    h_mesh = extract_mesh_handle_dir_xy(mesh_verts)
    t_world_np = init_t_world.detach().cpu().numpy()

    if verbose:
        print(f"[yaw_init] mesh handle direction (xy) = "
              f"({h_mesh[0]:+.3f}, {h_mesh[1]:+.3f})")

    yaws_rad = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    raw_candidates: List[Dict] = []

    for view in views:
        mask = view["mask"].detach().cpu().numpy() > 0.5
        d_mask = extract_mask_handle_dir_2d(mask)
        if d_mask is None:
            if verbose:
                print(f"[yaw_init] {view['name']}: mask asymmetry too weak; skip")
            continue

        R_view = view["R_view"].detach().cpu().numpy().astype(np.float64)
        T_view = view["T_view"].detach().cpu().numpy().astype(np.float64)
        fx, fy = float(view["fx"]), float(view["fy"])
        cx, cy = float(view["cx"]), float(view["cy"])

        scores = np.full(n_grid, -np.inf, dtype=np.float64)
        for i, theta in enumerate(yaws_rad):
            d_img = _yaw_handle_in_image(theta, h_mesh, t_world_np,
                                         R_view, T_view, fx, fy, cx, cy)
            if d_img is None:
                continue
            scores[i] = float(np.dot(d_img, d_mask))  # cos(angle); higher = better

        if not np.isfinite(scores).any():
            continue
        best_i = int(np.argmax(scores))
        best_yaw_deg = float(np.degrees(yaws_rad[best_i]))
        best_score = float(scores[best_i])
        mask_px = int(mask.sum())
        raw_candidates.append({
            "yaw_deg": best_yaw_deg, "score": best_score,
            "weight": mask_px, "source": view["name"],
        })
        if verbose:
            print(f"[yaw_init] {view['name']}: best yaw = {best_yaw_deg:+.1f}°  "
                  f"(cos-align={best_score:+.3f}, mask_px={mask_px})")

    if include_flip:
        flipped = [{**c, "yaw_deg": (c["yaw_deg"] + 180.0) % 360.0,
                    "source": c["source"] + "+180", "score": -c["score"]}
                   for c in raw_candidates]
        raw_candidates = raw_candidates + flipped

    if not raw_candidates:
        if verbose:
            print("[yaw_init] no candidates produced; "
                  "fall back to uniform multi-start")
        return []

    # Circular dedupe within `dedupe_deg`: keep highest weight×score per cluster.
    def _circ_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    deduped: List[Dict] = []
    # sort by weight × score descending so the first kept in each cluster is the best
    raw_candidates.sort(key=lambda c: -c["weight"] * c["score"])
    for c in raw_candidates:
        if any(_circ_dist(c["yaw_deg"], k["yaw_deg"]) < dedupe_deg for k in deduped):
            continue
        deduped.append(c)
    # canonical order by yaw
    deduped.sort(key=lambda c: c["yaw_deg"])
    return deduped
