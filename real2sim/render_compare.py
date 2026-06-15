"""Step 4: PyTorch3D 把 step3c 优化出来的 (R, t, s) 渲成 comparison.png.

提取自 full_workflow.py (2026-06), 保持原行为. 不进 Genesis, 纯 PT3D phong shading 渲.

World-frame mode (2026-06+): step3c 输出 R/t 是 Genesis world 系的 (frame="world").
此时相机摆位必须用 Genesis 模板 (single_pos/lookat/up/fov), 而不是 PT3D-cam-frame
时代的 R=I T=0. 否则 comparison/render.png 跟 genesis_preview/sim_c0.png 对不上.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .mesh_io import load_glb_as_pytorch3d


def step4_render(
    scene_image: str,
    results: Dict,
    mesh_paths: Dict[str, Path],
    output_dir: Path,
    template_path: Optional[Path] = None,
):
    import json
    import numpy as np
    import torch
    from PIL import Image
    from pytorch3d.renderer import (
        BlendParams, MeshRasterizer, MeshRenderer, PerspectiveCameras,
        PointLights, RasterizationSettings, SoftPhongShader, TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig = Image.open(scene_image).convert("RGB")
    W, H = orig.size
    orig_tensor = torch.from_numpy(np.array(orig)).float().to(device) / 255.0

    # ── World-frame mode: 相机摆位从 Genesis 模板读 (跟 render_pt3d_check.py 完全一致) ──
    # results["frame"] == "world" 时 R/t 是世界系下的 mesh→world. 相机不在 (0,0,0),
    # 必须用 single_pos/lookat/up + look_at_view_transform 算 R_view/T_view.
    # 没传 template 或 frame != "world" → 回退 R=I T=0 (老 PT3D-cam-frame 路径).
    frame = results.get("frame", "pt3d_cam")
    use_world = (frame == "world") and (template_path is not None)
    if use_world:
        tpl = json.loads(Path(template_path).read_text())
        cam_cfg = tpl["env"]["camera"]
        eye = np.asarray(cam_cfg["single_pos"],    dtype=np.float64)
        at  = np.asarray(cam_cfg["single_lookat"], dtype=np.float64)
        up  = np.asarray(cam_cfg["single_up"],     dtype=np.float64)
        fov_v_deg = float(cam_cfg["single_fov"])
        R_view, T_view = look_at_view_transform(
            eye=[eye.tolist()], at=[at.tolist()], up=[up.tolist()],
        )
        R_view, T_view = R_view.to(device), T_view.to(device)
        # 内参: GSRL gs_camera.py:187 同公式. 用 scene image 的 H, 不用 template 的 res
        # (template 是 64x64 之类的低分辨率, 我们要按 scene image 的实际分辨率渲).
        fy = (H / 2.0) / float(np.tan(np.radians(fov_v_deg) / 2.0))
        fx = fy
        cx, cy = W / 2.0, H / 2.0
        print(f"[step4] world-frame camera: eye={eye.tolist()}, at={at.tolist()}, "
              f"up={up.tolist()}, fov_v={fov_v_deg}°, fx=fy={fx:.1f}")
    else:
        cam = results.get("camera", {})
        fx = cam.get("fx", W)
        fy = cam.get("fy", H)
        cx = cam.get("cx", W / 2)
        cy = cam.get("cy", H / 2)
        R_view = torch.eye(3, device=device)[None]
        T_view = torch.zeros(1, 3, device=device)
        print(f"[step4] pt3d-cam-frame camera (legacy): R=I, T=0, fx={fx:.1f}")

    all_verts, all_faces, all_colors = [], [], []
    face_offset = 0
    for obj in results.get("objects", []):
        name = obj["name"]
        R = obj["R"]
        t = obj["t"]
        s = obj["scale"]
        if isinstance(R, list):
            R = torch.tensor(R)
        if isinstance(t, list):
            t = torch.tensor(t)
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()

        glb_path = mesh_paths.get(name)
        if glb_path is None:
            print(f"  [WARN] No mesh for {name}, skipping")
            continue
        verts, faces, vc = load_glb_as_pytorch3d(glb_path, device)

        # PT3D row-vec, 跟 step3c 优化器一致: cameras.transform_points(X_world) = X_world @ R + T.
        # camera R = I, T = 0 → mesh 摆到 world 等价于 X_world = X_canonical @ R_opt + t_opt.
        verts = verts * s
        verts = verts @ R.to(device) + t.to(device)

        all_verts.append(verts)
        all_faces.append(faces + face_offset)
        face_offset += len(verts)

        if vc is not None:
            all_colors.append(vc)
        else:
            color = torch.rand(3, device=device)[None].expand(len(verts), 3)
            all_colors.append(color)

    if not all_verts:
        print("[step4] no meshes to render")
        return

    big_verts = torch.cat(all_verts, dim=0)
    big_faces = torch.cat(all_faces, dim=0)
    big_colors = torch.cat(all_colors, dim=0)
    mesh = Meshes(
        verts=[big_verts],
        faces=[big_faces],
        textures=TexturesVertex(verts_features=big_colors[None]),
    )

    blend = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        bin_size=0, max_faces_per_bin=200_000,
    )

    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        R=R_view, T=T_view,
        device=device, in_ndc=False,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, blend_params=blend, lights=lights),
    )

    rendered = renderer(mesh, cameras=cameras)
    render_img = rendered[0, ..., :3].clamp(0, 1)
    alpha = (rendered[0, ..., 3] > 0.01).float().unsqueeze(-1)
    overlay = orig_tensor * (1 - alpha * 0.5) + render_img * (alpha * 0.5)
    sil = alpha.expand(-1, -1, 3)
    side = torch.cat([orig_tensor, sil, overlay], dim=1)

    def save_tensor(t, name):
        arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(output_dir / name)

    save_tensor(overlay, "overlay.png")
    save_tensor(side, "comparison.png")
    save_tensor(render_img, "render.png")
    print(f"[step4] saved to {output_dir}/")
    print(f"  comparison.png  — 原图 | 轮廓 | 叠加")
    print(f"  overlay.png     — 原图+mesh 半透明叠加")
    print(f"  render.png      — 纯渲染")
