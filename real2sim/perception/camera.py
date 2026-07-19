"""
Camera utilities: intrinsics from (H,W) heuristics, and a PyTorch3D camera wrapper.
"""

import torch
import torch.nn.functional as F


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float = 60.0) -> tuple[float, float, float, float]:
    """Derive fx, fy, cx, cy from image size and horizontal FOV."""
    fx = (width / 2.0) / torch.tan(torch.deg2rad(torch.tensor(hfov_deg / 2.0))).item()
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """3x3 intrinsic matrix."""
    K = torch.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


@torch.jit.script
def project_points(points_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project 3D points to 2D.

    Args:
        points_3d: (N, 3) world points
        K: (3, 3) intrinsics
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        (N, 2) pixel coordinates
    """
    cam = points_3d @ R.T + t[None, :]
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3].clamp(min=1e-8)
    return uv


def estimate_initial_translation(
    mask: torch.Tensor, mesh_verts: torch.Tensor, K: torch.Tensor, scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heuristic initial pose from mask bounding box and mesh.

    Assumes the object is roughly centered and at a canonical distance
    where its projected size matches the mask extent.

    Args:
        mask: (H, W) binary object mask
        mesh_verts: (V, 3) canonical mesh vertices
        K: (3, 3) camera intrinsics
        scale: initial scale guess

    Returns:
        R: (3, 3) rotation (identity)
        t: (3,) translation
    """
    device = mask.device
    H, W = mask.shape

    # Bounding box center
    ys, xs = torch.where(mask > 0.5)
    if len(ys) == 0:
        return torch.eye(3, device=device), torch.tensor([0.0, 0.0, 2.0], device=device)

    cx_mask = xs.float().mean()
    cy_mask = ys.float().mean()
    bbox_diag = torch.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Mesh diameter
    mesh_diag = (mesh_verts.max(dim=0).values - mesh_verts.min(dim=0).values).norm()

    # Estimate z so that projection size matches mask size
    z_est = (fx + fy) / 2.0 * mesh_diag * scale / bbox_diag.clamp(min=1.0)

    # Back-project to estimate tx, ty
    tx = (cx_mask - cx) / fx * z_est
    ty = (cy_mask - cy) / fy * z_est

    R = torch.eye(3, device=device)
    t = torch.tensor([tx, ty, z_est], device=device)
    return R, t


def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        r6d: (..., 6) first two columns of rotation matrix, flattened

    Returns:
        (..., 3, 3) rotation matrix
    """
    *batch, _ = r6d.shape
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation.

    R 是 column-major（rotation_6d_to_matrix 用 stack([b1,b2,b3], dim=-1)，
    所以 b1,b2,b3 是 R 的列）。要还原必须取**列**然后 cat，不是 row-major reshape。

    Bug 修复 2026-05-17: 之前用 R[...,:2].reshape(6) 是 row-major，
    输出 [b1_x, b2_x, b1_y, b2_y, b1_z, b2_z]，
    送进 rotation_6d_to_matrix 的 a1=[b1_x, b2_x, b1_y] 完全错位，
    identity matrix round-trip 都出 NaN，refinement 起点就乱。
    """
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)
