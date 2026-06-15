#!/usr/bin/env python
"""Render gsrl_config.json's scene via PyTorch3D for A/B comparison with Genesis.

目标: 同一个 gsrl_config (同样的 mesh / spawn_pos / spawn_quat / scale + 同样的
single_pos / lookat / up / fov / res), PT3D 渲一张, Genesis 也渲一张 (走原 step5_render),
看两张图是不是基本一致. 不一致 = 渲染管线层面的偏差; 一致 = 渲染管线没问题, bug 在
step3c 出来的 R/t 本身.

跟 Genesis 对齐的关键:
  - 把 Genesis 加载 .glb 时**自动**做的那个 M_LOAD (Y_UP_TRANSFORM.T, 也就是我们的
    M_PT3D 矩阵) 在 PT3D 这边也手动应用一次, 保证两边看到的是**同一份顶点朝向**.
  - 相机用 look_at_view_transform, eye/at/up 直接喂 single_*.
  - 内参 fx/fy 从 single_fov + 图像高度反推 (跟 GSRL gs_camera.py:187 同一公式).

跑 (sam3d-objects env, 有 pytorch3d):
    python scripts/render_pt3d_check.py --config <gsrl_config.json> --output out.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as Rscipy

from pytorch3d.renderer import (
    BlendParams, MeshRasterizer, MeshRenderer, PerspectiveCameras,
    PointLights, RasterizationSettings, SoftPhongShader, TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


# Genesis (genesis/utils/mesh.py:Y_UP_TRANSFORM) 在加载 .glb 时通过 trimesh.apply_transform
# 应用这个矩阵 (col-vec convention). 我们这里用 row-vec, 等价是 verts @ M.T.
# 注意 Genesis 用的是 Y_UP_TRANSFORM.T, 我们存的就是 .T 之后的 3x3 (跟
# load_glb_as_pytorch3d 里的 M_PT3D 一致).
M_PT3D = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)

OBJECT_COLORS = {
    "mug":  (0.30, 0.45, 0.85),   # blue
    "tree": (0.85, 0.30, 0.30),   # red
}
DEFAULT_COLOR = (0.7, 0.7, 0.7)


def load_and_transform_mesh(asset_path, scale, spawn_quat_wxyz, spawn_pos, device,
                              skip_m_load=False):
    """读 .glb, (可选) 应用 Genesis 同款 M_LOAD, 然后 scale / rotate / translate 到 world.

    步骤跟 Genesis 一致:
        1. v_raw 从 .glb
        2. v_loaded = v_raw @ M_PT3D.T   (= Y_UP_TRANSFORM.T 应用到 col-vec)
           — 若 skip_m_load=True (= Genesis file_meshes_are_zup=True 等价), 跳过这步
        3. v_scaled = v_loaded * scale
        4. v_world = R_spawn @ v_scaled + spawn_pos   (col-vec, R 来自 spawn_quat wxyz)
    """
    mesh = trimesh.load(str(asset_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    v_raw = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    if skip_m_load:
        v_loaded = v_raw     # raw .glb 顶点, 不做 M_LOAD
    else:
        v_loaded = v_raw @ M_PT3D.T.astype(np.float64)     # M_LOAD
    v_scaled = v_loaded * float(scale)

    # spawn_quat in gsrl is wxyz, scipy 要 xyzw
    qw, qx, qy, qz = spawn_quat_wxyz
    R_spawn = Rscipy.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float64)
    spawn_pos = np.asarray(spawn_pos, dtype=np.float64)
    # col-vec: v_world = R @ v_mesh + t  ⇔  row-vec: v_world_row = v_mesh_row @ R.T + t
    v_world = v_scaled @ R_spawn.T + spawn_pos

    return (
        torch.tensor(v_world, dtype=torch.float32, device=device),
        torch.tensor(f, dtype=torch.int64, device=device),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="gsrl_config.json path")
    p.add_argument("--output", default=None, help="output PNG path; default = <config_dir>/pt3d_check.png")
    p.add_argument("--bg-color", type=float, nargs=3, default=(0.05, 0.07, 0.15),
                   help="background RGB (0-1), default ≈ Genesis depth blue")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    env = cfg["env"]
    cam = env["camera"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pt3d-check] device = {device}")

    # ── 相机 (Genesis convention, z-up world) ───────────────────────────────
    eye = np.asarray(cam["single_pos"],    dtype=np.float64)
    at  = np.asarray(cam["single_lookat"], dtype=np.float64)
    up  = np.asarray(cam["single_up"],     dtype=np.float64)
    fov_v_deg = float(cam["single_fov"])
    W, H = int(cam["res"][0]), int(cam["res"][1])

    # PT3D 用 look_at_view_transform 自动算 R, T (世界系 → PT3D cam, row-vec)
    R_view, T_view = look_at_view_transform(
        eye=[eye.tolist()], at=[at.tolist()], up=[up.tolist()],
    )
    R_view, T_view = R_view.to(device), T_view.to(device)

    # 内参: GSRL gs_camera.py:187 同一公式 — fy = (H/2) / tan(fov_v/2)
    fy = (H / 2.0) / np.tan(np.radians(fov_v_deg) / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0

    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        R=R_view, T=T_view,
        device=device, in_ndc=False,
    )
    print(f"[pt3d-check] cam pos    = {eye.tolist()}")
    print(f"[pt3d-check] cam lookat = {at.tolist()}")
    print(f"[pt3d-check] cam up     = {up.tolist()}")
    print(f"[pt3d-check] fov_v={fov_v_deg:.2f}°  res={W}x{H}  fx=fy={fx:.1f}")

    # ── Meshes ─────────────────────────────────────────────────────────────
    all_verts, all_faces, all_colors = [], [], []
    face_offset = 0
    for key in ["mug", "tree"]:
        if key not in env or env[key].get("asset_path") is None:
            continue
        obj = env[key]
        morph = obj["entity_kwargs"]["morph_kwargs"]
        scale = morph.get("scale", 1.0)
        # file_meshes_are_zup=True ↔ Genesis 跳 M_LOAD ↔ 我们这边也跳
        skip = bool(morph.get("file_meshes_are_zup", False))
        v, f = load_and_transform_mesh(
            obj["asset_path"], scale,
            obj["spawn_quat"], obj["spawn_pos"], device,
            skip_m_load=skip,
        )
        if skip:
            print(f"[pt3d-check] {key}: file_meshes_are_zup=True → 跳 M_LOAD (跟 Genesis 一致)")
        all_verts.append(v)
        all_faces.append(f + face_offset)
        face_offset += int(v.shape[0])

        color = torch.tensor(OBJECT_COLORS.get(key, DEFAULT_COLOR),
                              device=device, dtype=torch.float32)
        all_colors.append(color[None].expand(v.shape[0], 3))
        print(f"[pt3d-check] {key}: {obj['asset_path']}  "
              f"scale={scale:.4f}  spawn_pos={obj['spawn_pos']}  verts={v.shape[0]}")

    if not all_verts:
        print("[pt3d-check] [fatal] mug + tree 都没 asset_path, 无法渲染")
        sys.exit(1)

    big_verts = torch.cat(all_verts, dim=0)
    big_faces = torch.cat(all_faces, dim=0)
    big_colors = torch.cat(all_colors, dim=0)
    meshes = Meshes(
        verts=[big_verts], faces=[big_faces],
        textures=TexturesVertex(verts_features=big_colors[None]),
    )

    # ── 渲染 ───────────────────────────────────────────────────────────────
    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        bin_size=0, max_faces_per_bin=200_000,
    )
    blend = BlendParams(sigma=1e-4, gamma=1e-4, background_color=tuple(args.bg_color))
    lights = PointLights(
        device=device,
        location=[[float(eye[0]), float(eye[1]), float(eye[2]) + 1.0]],
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend),
    )
    rendered = renderer(meshes)
    img = rendered[0, ..., :3].clamp(0, 1).cpu().numpy()
    arr = (img * 255).astype(np.uint8)

    out = Path(args.output) if args.output else Path(args.config).parent / "pt3d_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)
    print(f"\n[pt3d-check] saved: {out}")


if __name__ == "__main__":
    main()
