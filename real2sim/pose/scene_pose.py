#!/usr/bin/env python3
"""step3d_scene_pose.py — 把 SAM3D-objects 跑在 scene.jpg 上, 拿 scene 相机系下的 pose+mesh.

跟 step2 (per-object multi-view) 的区别:
    step2: 输入 5 张 mug_tree 多视角图 → mesh in mug_tree-scan-camera 系
    step3d: 输入 1 张 scene.jpg + 该物体的 scene mask → mesh in scene-camera 系
            **位姿直接就是我们要的 init**, 不再需要 R_scene gravity 补偿那套 hack.

依赖 (sam3d-objects env). 路径由 pipeline_paths 解析, 见模块 docstring.
    paths.scene_masks_path       — step3a per-object mask
    paths.scene_pointmap_path    — step3b metric pointmap (PT3D 系, (H,W,3))
    paths.scene_intrinsics_path  — step3b K (用户提供 / MoGe 估)
    paths.scene_image            — 场景图

输出:
    paths.scene_sam3d_dir/<obj_name>/result.glb   — SAM3D 在 scene 下重建的 mesh
    paths.scene_sam3d_dir/<obj_name>/params.npz   — pose (R, t, s) in scene-camera 系

跑法:
    conda activate sam3d-objects
    python -m real2sim.pose.scene_pose
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # real2sim/pose/.. .. == R2SConfig/
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.io.scenes import OBJECTS
from real2sim.io.paths import resolve as resolve_paths, sanitize_filename, should_rebuild
from full_workflow import MV_SAM3D_DIR


def main():
    paths = resolve_paths()
    force = should_rebuild()

    # ── 1. 读 scene 数据 ───────────────────────────────────────────
    from PIL import Image as _Image
    scene_img = _Image.open(paths.scene_image).convert("RGB")
    scene_np = np.array(scene_img)  # (H, W, 3) uint8
    H, W = scene_np.shape[:2]
    print(f"[step3d] scene.jpg {W}x{H}  (scene_id={paths.scene_id}, prompt_id={paths.prompt_id})")

    mask_file = paths.scene_masks_path
    pm_file   = paths.scene_pointmap_path
    K_file    = paths.scene_intrinsics_path
    for f in [mask_file, pm_file, K_file]:
        if not f.exists():
            print(f"[fatal] 缺 {f}, 先跑 step3a/3b"); sys.exit(1)

    bundle = torch.load(mask_file, map_location="cpu", weights_only=False)
    labels = bundle["labels"]
    masks = bundle["masks"]
    print(f"[step3d] {len(labels)} masks: {labels}")

    # scene_pointmap.npy 在 PyTorch3D 相机系 (X 左, Y 上, Z 前), step3b 翻了 X, Y.
    # SAM3D 的 compute_pointmap 期望"标准相机系" (X 右, Y 下, Z 前), 内部再做翻转.
    # 我们这里翻回去.
    pm_p3d = np.load(pm_file).astype(np.float32)  # (H, W, 3)
    pm_std = pm_p3d.copy()
    pm_std[..., 0] = -pm_std[..., 0]   # undo X flip
    pm_std[..., 1] = -pm_std[..., 1]   # undo Y flip
    # (H, W, 3) → (3, H, W) channel-first, torch tensor
    pointmap_chw = torch.from_numpy(pm_std).permute(2, 0, 1).contiguous()
    print(f"[step3d] scene_pointmap → standard camera space, shape (3,{H},{W})")
    print(f"           X range [{pm_std[...,0].min():.3f}, {pm_std[...,0].max():.3f}]")
    print(f"           Y range [{pm_std[...,1].min():.3f}, {pm_std[...,1].max():.3f}]")
    print(f"           Z range [{pm_std[...,2].min():.3f}, {pm_std[...,2].max():.3f}]")

    # ── 2. cd 到 MV-SAM3D, import inference (跟 run_inference_weighted.py 一致) ───────────────────────
    original_cwd = os.getcwd()
    os.chdir(MV_SAM3D_DIR)
    sys.path.insert(0, str(MV_SAM3D_DIR))
    sys.path.insert(0, str(MV_SAM3D_DIR / "notebook"))
    print(f"[step3d] cd → {MV_SAM3D_DIR}")

    from inference import Inference
    config_path = "checkpoints/hf/pipeline.yaml"
    if not Path(config_path).exists():
        print(f"[fatal] {MV_SAM3D_DIR}/{config_path} 不存在")
        sys.exit(1)
    print(f"[step3d] loading SAM3D pipeline from {config_path} ...")
    inference = Inference(config_path, compile=False)
    if hasattr(inference._pipeline, "rendering_engine"):
        inference._pipeline.rendering_engine = "pytorch3d"
    print(f"[step3d] SAM3D ready.")

    # ── 3. 每个物体跑一次 ──────────────────────────────────────────
    out_root = paths.scene_sam3d_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for label, mask in zip(labels, masks):
        slug = sanitize_filename(label)
        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # mask: (H, W) bool / float → uint8 0/255
        mask_np = mask.cpu().numpy().astype(bool)
        mask_u8 = (mask_np * 255).astype(np.uint8)
        print(f"\n[step3d] === {label} ({slug}) ===")
        print(f"  mask area = {int(mask_np.sum())} px")

        if not force and (out_dir / "result.glb").exists() and (out_dir / "params.npz").exists():
            print(f"  cache hit: result.glb + params.npz exist; REBUILD=1 to force.")
            continue

        # 跑 SAM3D, full pipeline (Stage 1 + Stage 2). pointmap 自己提供.
        # with_mesh_postprocess=True: 出 .glb mesh
        # with_layout_postprocess=True: 出 layout-optimized (R, t, s)
        with torch.no_grad():
            result = inference._pipeline.run(
                image=scene_np,            # (H, W, 3) uint8 RGB
                mask=mask_u8,              # (H, W) uint8 0/255
                seed=42,
                stage1_only=False,
                with_mesh_postprocess=True,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=True,
                stage1_inference_steps=50,
                stage2_inference_steps=25,
                pointmap=pointmap_chw,
            )

        # 拿出 mesh
        glb = result.get("glb", None)
        if glb is not None:
            mesh_path = out_dir / "result.glb"
            glb.export(str(mesh_path))
            print(f"  saved mesh → {mesh_path}")
        else:
            print(f"  [warn] no mesh produced")

        # 拿出 pose: rotation (wxyz quat), translation, scale
        def tonp(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        params = {}
        for k in ["rotation", "translation", "scale", "downsample_factor",
                  "pointmap_scale", "pointmap_shift"]:
            if k in result:
                params[k] = tonp(result[k])
        if params:
            params_path = out_dir / "params.npz"
            np.savez(params_path, **params)
            print(f"  saved params → {params_path}")
            for k, v in params.items():
                if v.size <= 8:
                    print(f"    {k}: {v.flatten().tolist()}")
        else:
            print(f"  [warn] no pose params in result")

    os.chdir(original_cwd)
    print(f"\n[step3d] done. outputs under {out_root}")


if __name__ == "__main__":
    main()
