"""Scene-level geometric helpers (world-frame mode only).

step3c 走 world-frame 路径, 只需要:
- pt3d_world_camera_from_genesis: 从 Genesis 相机配置算 PT3D 外参 + 内参 + plane_z
- yaw_rotation_matrix: step3c 生成 init_R yaw candidates 用

之前老路径用的 fit_dominant_plane_ransac / build_scene_R_from_gravity /
gravity_dir_from_genesis_camera / intrinsics_from_genesis_camera /
compute_align_long_axis_rotation 全部移除了 (2026-06 清理).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def pt3d_world_camera_from_genesis(template_path: Path, H: int, W: int, device):
    """从 Genesis 相机配置算 PT3D 的 world→view 外参 + 内参 + plane_z.

    "world" = Genesis world (z-up). 相机摆在 single_pos 看 single_lookat, up = single_up.
    PT3D 的 look_at_view_transform 算 R_view, T_view 使得
        v_view = v_world @ R_view + T_view (row-vec).

    返回 dict: R_view (3,3 torch), T_view (3, torch), fx, fy, cx, cy, H, W, plane_z, cam_pos.

    用法: 传给 PoseOptimizer.optimize(world_camera_params=...). 优化器直接在 Genesis world
    里学 mesh→world 的 R, t, s. step5 不需要 compose_pose 转换.
    """
    import json as _json
    import math as _math
    import torch as _t
    from pytorch3d.renderer import look_at_view_transform

    try:
        cfg = _json.loads(Path(template_path).read_text())
    except Exception as e:
        print(f"[pt3d_world_camera] 读 {template_path} 失败: {e}")
        return None

    cam = cfg["env"]["camera"]
    eye = cam["single_pos"]
    at  = cam["single_lookat"]
    up  = cam["single_up"]
    fov_v = float(cam["single_fov"])
    plane_z = float(cfg["env"].get("scene", {}).get("plane_z", 0.0))

    R_view, T_view = look_at_view_transform(eye=[eye], at=[at], up=[up])
    R_view = R_view.squeeze(0).to(device).float()
    T_view = T_view.squeeze(0).to(device).float()

    # 内参: fy = (H/2) / tan(fov_v/2), 跟 GSRL gs_camera.py:187 同公式
    fy = (H / 2.0) / _math.tan(_math.radians(fov_v) / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0

    return {
        "R_view":  R_view,
        "T_view":  T_view,
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "H": int(H), "W": int(W),
        "plane_z": plane_z,
        "cam_pos": _t.tensor(eye, dtype=_t.float32, device=device),
    }


def solve_init_pose_from_mask_pointmap(
    mask,
    pointmap,
    world_camera_params: dict,
    plane_z: float,
    mesh_z_extent: float,
    device,
):
    """用 MoGe metric pointmap 反推 (tx, ty, scale).

    跟 solve_init_pose_from_mask 同样的输出语义, 但不靠像素↔plane_z 几何相交.
    pointmap (H, W, 3): PT3D camera frame, metric meters.
    流程:
      1. mask 像素的 pointmap 3D (camera frame metric).
      2. R_view.T + cam_pos → world frame 3D.
      3. 找 world-z 最小的一批像素 (~5%, 至少 5 个), 这些就是物体贴地的部分.
      4. 这批像素 world (X, Y) 中位数 = base 中心.
    scale 用跟 solve_init_pose_from_mask 同一公式: mask_h_px / proj_h_unit (1m canonical).

    返回 None / (tx, ty, scale).
    """
    import torch as _t

    wc = world_camera_params
    cam_pos = wc["cam_pos"].to(device).float()
    R_view = wc["R_view"].to(device).float()
    T_view = wc["T_view"].to(device).float()
    fx = float(wc["fx"]); fy = float(wc["fy"])
    cx = float(wc["cx"]); cy = float(wc["cy"])

    mask_dev = mask.to(device)
    pm = pointmap.to(device).float()                  # (H, W, 3)
    ys, xs = _t.where(mask_dev > 0.5)
    if len(ys) == 0:
        return None
    mask_h_px = float(ys.max().item() - ys.min().item() + 1)

    # mask 像素的 3D (camera frame, metric)
    P_cam = pm[ys, xs]                                # (N, 3)

    # camera → world: v_world = v_cam @ R_view.T + cam_pos
    # (PT3D look_at_view_transform 给出 R_view, T_view 满足 v_view = v_world @ R_view + T_view;
    #  反过来 v_world = (v_view - T_view) @ R_view.T = v_view @ R_view.T + cam_pos.)
    P_world = P_cam @ R_view.T + cam_pos.unsqueeze(0) # (N, 3)

    # 找贴地像素: world-z 最低的 ~5% (至少 5 个), 抗噪取 median(X, Y).
    N = P_world.shape[0]
    k = max(5, int(0.05 * N))
    z_sorted, idx_sorted = P_world[:, 2].sort()
    floor_idx = idx_sorted[:k]
    floor_pts = P_world[floor_idx]                    # (k, 3)
    tx = float(floor_pts[:, 0].median().item())
    ty = float(floor_pts[:, 1].median().item())
    floor_z_med = float(floor_pts[:, 2].median().item())

    # 也算一下 mask 所有像素的 world 中位 (debug, 不用)
    all_xy_med = (float(P_world[:, 0].median().item()),
                  float(P_world[:, 1].median().item()))
    all_z_med = float(P_world[:, 2].median().item())

    print(f"  [init_xy:pointmap] floor band (k={k}/{N}) → world ({tx:+.3f},{ty:+.3f})  "
          f"floor_z_med={floor_z_med:+.3f} (plane_z={plane_z:+.3f})  |  "
          f"all-mask median xy=({all_xy_med[0]:+.3f},{all_xy_med[1]:+.3f}) z={all_z_med:+.3f}")

    # ── scale: 跟 backproject 路径同一个公式 ──
    def _project(P_world_pt):
        P_view = P_world_pt @ R_view + T_view
        zv = float(P_view[2].item())
        if abs(zv) < 1e-8:
            return None
        u = -fx * float(P_view[0].item()) / zv + cx
        v = -fy * float(P_view[1].item()) / zv + cy
        return (u, v)

    P_bot = _t.tensor([tx, ty, plane_z], device=device, dtype=R_view.dtype)
    P_top = _t.tensor([tx, ty, plane_z + mesh_z_extent],
                       device=device, dtype=R_view.dtype)
    pb = _project(P_bot); pt_ = _project(P_top)
    if pb is None or pt_ is None:
        return None
    proj_h_unit = ((pt_[0] - pb[0]) ** 2 + (pt_[1] - pb[1]) ** 2) ** 0.5
    if proj_h_unit < 1e-3:
        return None

    scale = mask_h_px / proj_h_unit
    return tx, ty, scale


def solve_init_pose_from_mask(
    mask,
    world_camera_params: dict,
    plane_z: float,
    mesh_z_extent: float,
    device,
):
    """Closed-form 反推 (tx, ty, scale) 给 yaw_only + floor_lock 模式.

    几何前提:
      - mesh +Z 是 world up (M_LOAD 之后)
      - 物体底面 (mesh canonical z_min) 在 plane_z 上 (floor_lock 保证)
      - mask 是物体在 single_pos 那只相机里的二值轮廓

    Step 1: (tx, ty) ← mask centroid 反投影到 plane_z
      历史: 试过 mask 底部 backproject (本来想让"底像素本来就在地上"), 但 (a) 厚物
      体 (disc) 的底像素是盘子近相机一侧, 给的 (tx, ty) 偏向 camera; (b) image-bottom
      附近 ray 接近水平, 单像素噪声放大成 50cm+ 偏移. 反而是 centroid 在多个测试
      场景下更稳 — mug 8cm 误差 optimizer 能救, tree 偏的也是另一个 basin (要靠
      grid search 兜底, 不是闭式本身能搞定的).
      像素 (u_c, v_c) → view-frame 方向 → world-frame 方向 → 跟 plane_z 求交.
      PT3D pixel↔view 约定 (跟 pipeline.py:251-267 一致):
        u = -fx · x_view / z_view + cx,  v = -fy · y_view / z_view + cy

    Step 2: scale ← 让 mesh canonical 高度投影到 image 的像素高度 = mask 高度
      P_bot = (tx, ty, plane_z)
      P_top = (tx, ty, plane_z + mesh_z_extent)        # canonical 单位 scale
      proj_h_unit = ||project(P_top) - project(P_bot)|| 像素
      scale = mask_h_px / proj_h_unit

    返回:
      None — mask 空 / ray 不穿过平面 / 投影退化时
      (tx, ty, scale) — float 三元组
    """
    import torch as _t

    wc = world_camera_params
    cam_pos = wc["cam_pos"].to(device).float()
    R_view = wc["R_view"].to(device).float()
    T_view = wc["T_view"].to(device).float()
    fx = float(wc["fx"]); fy = float(wc["fy"])
    cx = float(wc["cx"]); cy = float(wc["cy"])

    mask_dev = mask.to(device)
    ys, xs = _t.where(mask_dev > 0.5)
    if len(ys) == 0:
        return None
    u_c = float(xs.float().mean().item())
    v_c = float(ys.float().mean().item())
    mask_h_px = float(ys.max().item() - ys.min().item() + 1)

    # ── Step 1: backproject (u_c, v_c) to world ray, intersect plane_z ─────
    d_view = _t.tensor(
        [-(u_c - cx) / fx, -(v_c - cy) / fy, 1.0],
        device=device, dtype=R_view.dtype,
    )
    d_world = d_view @ R_view.T                       # (3,)
    if abs(float(d_world[2].item())) < 1e-8:
        return None                                    # ray 平行 plane
    t_star = (plane_z - float(cam_pos[2].item())) / float(d_world[2].item())
    if t_star <= 0:
        return None                                    # plane 在相机后方
    hit = cam_pos + t_star * d_world
    tx = float(hit[0].item()); ty = float(hit[1].item())
    print(f"  [init_xy:centroid] mask centroid ({u_c:.0f},{v_c:.0f}) → world ({tx:+.3f},{ty:+.3f})")

    # ── Step 2: pixel height of canonical mesh @ (tx, ty, plane_z) ─────────
    def _project(P_world):
        P_view = P_world @ R_view + T_view             # (3,)
        zv = float(P_view[2].item())
        if abs(zv) < 1e-8:
            return None
        u = -fx * float(P_view[0].item()) / zv + cx
        v = -fy * float(P_view[1].item()) / zv + cy
        return (u, v)

    P_bot = _t.tensor([tx, ty, plane_z], device=device, dtype=R_view.dtype)
    P_top = _t.tensor([tx, ty, plane_z + mesh_z_extent],
                       device=device, dtype=R_view.dtype)
    pb = _project(P_bot); pt_ = _project(P_top)
    if pb is None or pt_ is None:
        return None
    proj_h_unit = ((pt_[0] - pb[0]) ** 2 + (pt_[1] - pb[1]) ** 2) ** 0.5
    if proj_h_unit < 1e-3:
        return None

    scale = mask_h_px / proj_h_unit
    return tx, ty, scale


def yaw_rotation_matrix(theta_rad: float, device):
    """绕 z-up 世界的 +Z 轴转 theta. PyTorch3D row-vector 约定.

    用于 step3c 生成 init_R candidates (8 个绕 +Z 的 yaw, 覆盖 360°).
    跟 mesh-after-M_LOAD 的 +Z 是 up 方向对齐 (.glb 是 gltf y-up, M_LOAD 转 z-up).
    """
    import math
    import torch
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return torch.tensor(
        [[c, s, 0.0],
         [-s, c, 0.0],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    )
