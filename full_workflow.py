#!/usr/bin/env python3
"""
Real2Sim Full Workflow (V2, 双环境版本)
========================================

R2SConfig 顶层入口。从物体多视角照片 → SAM3 mask → SAM3D mesh
→ Real2Sim 位姿优化 → 渲染对比。

**项目无法在单个 conda env 内跑通**，按步骤切换：

    Step    内容                                              env
    ──────────────────────────────────────────────────────────────────
    1       多视角物体照片 → SAM3 mask                        sam3
    2       SAM3D 多视角重建 mesh (subprocess MV-SAM3D)       sam3d-objects
    3a      场景图 → SAM3 分割 (写盘到 workdir/scene_masks/)  sam3
    3b      场景图 → SAM3D 单图 metric pointmap (可选)        sam3d-objects
    3c      加载预存 mask + pointmap → Real2Sim 优化           sam3d-objects
    4       pytorch3d 渲染对比                                sam3d-objects

每步执行时会打印当前应使用的 env，import 失败会清晰提示。
日志写入 R2SConfig/logs/full_workflow_{timestamp}.log。

用法:
    # 在 sam3 env 下
    conda activate sam3
    python full_workflow.py --step 1
    python full_workflow.py --step 3a

    # 在 sam3d-objects env 下
    conda activate sam3d-objects
    python full_workflow.py --step 2   # 只打印 SAM3D 命令
    python full_workflow.py --step 3c
    python full_workflow.py --step 4

参数 --step 支持: 1, 2, 3a, 3b, 3c, 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────
# 路径与配置: 集中在 real2sim/io/{paths,scenes}.py.
# 这里只保留 PROJECT_ROOT 和外部仓库目录 (别人的代码, 不动).
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent          # R2SConfig/
MV_SAM3D_DIR = PROJECT_ROOT / "MV-SAM3D"                # 别人的仓库, 不动
SAM3_DIR = PROJECT_ROOT / "sam3"                        # 别人的仓库, 不动
LOG_DIR = PROJECT_ROOT / "logs"
VISUALIZATION_DIR = MV_SAM3D_DIR / "visualization"      # SAM3D 写死的输出根, 我们 symlink 进 build/

# 让 real2sim 包可被 import (顶层目录 == sys.path 第一个)
sys.path.insert(0, str(PROJECT_ROOT))

# 共享配置 / 路径解析器 (路径布局: 见 real2sim/io/paths.py 顶部 docstring)
from real2sim.io.scenes import OBJECTS, SCENE_PROMPT, MASK_CONFIDENCE  # noqa: E402
from real2sim.io.paths import (  # noqa: E402
    resolve as resolve_paths,
    obj_hash,
    sanitize_filename as _sanitize_pp,
    update_latest_symlink,
    should_rebuild,
    write_object_prompt_audit,
    write_prompt_audit,
    PipelinePaths,
)

# Re-export helpers for the moved step scripts (step3e_align_meshes,
# step3d_scene_pose still `from full_workflow import find_glb_files / MV_SAM3D_DIR`).
from real2sim.perception.mesh_io import (  # noqa: E402, F401
    find_glb_files,
    link_object_mesh,
    load_glb_as_pytorch3d,
)
from real2sim.pose.scene_geometry import (  # noqa: E402
    pt3d_world_camera_from_genesis,
    yaw_rotation_matrix,
)
from real2sim.export.render_compare import step4_render as _step4_render_impl  # noqa: E402

# SAM3D 入口 (相对 MV-SAM3D/)
SAM3D_SCRIPT = "run_inference_weighted.py"


# ─────────────────────────────────────────────────────────────────────
# 日志: 把所有 print / 错误 同时写到 logs/full_workflow_{ts}.log
# ─────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logging(step_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"full_workflow_step{step_name}_{ts}.log"
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[log] writing to {log_path}")
    print(f"[log] cwd={os.getcwd()}  python={sys.executable}")
    print(f"[log] argv={sys.argv}")
    return log_path


def banner(step: str, env: str, desc: str):
    print()
    print("=" * 70)
    print(f"[STEP {step}]  env=<{env}>  ::  {desc}")
    print("=" * 70)


def assert_env(required: str, sentinel_imports: List[str]):
    """尝试 import 标志包；失败则告诉用户切环境。"""
    missing = []
    for mod in sentinel_imports:
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"  - {mod}  ({type(e).__name__}: {e})")
    if missing:
        print(f"\n[ERROR] 当前环境似乎不是 `{required}`。以下 import 失败：")
        for line in missing:
            print(line)
        print(f"\n请运行：  conda activate {required}  然后重试。")
        sys.exit(2)
    print(f"[env check] {required} OK  ({', '.join(sentinel_imports)} all importable)")


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────

# re-exported for step3d / step3e that import sanitize_filename from here
sanitize_filename = _sanitize_pp


def _make_serializable(results: Dict) -> Dict:
    import numpy as np
    import torch

    def convert(v):
        if isinstance(v, torch.Tensor):
            return v.tolist()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: convert(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(vv) for vv in v]
        return v

    return convert(results)


# ─────────────────────────────────────────────────────────────────────
# Step 1: 多视角物体照片 → SAM3 mask           (env: sam3)
# ─────────────────────────────────────────────────────────────────────

def step1_object_masks():
    banner("1", "sam3", "Generate per-view masks for each object via SAM3")
    assert_env("sam3", ["sam3.model_builder", "sam3.model.sam3_image_processor"])

    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    paths = resolve_paths()
    force = should_rebuild()

    print("[step1] loading SAM3 model...")
    model = build_sam3_image_model()

    # SAM3 的 MLP fc1 用 fused addmm_act 强制输出 bfloat16，
    # 后面 fc2 的 nn.Linear 需要 autocast 来跟上 dtype；
    # 参考 sam3/examples/sam3_image_predictor_example.ipynb
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor(model, confidence_threshold=0.01, device=device_type)
    torch.autocast(device_type, dtype=torch.bfloat16).__enter__()
    print(f"[step1] autocast enabled (device_type={device_type}, dtype=bfloat16)")

    summary = {}
    for obj in OBJECTS:
        name = obj["name"]
        prompt = obj["prompt"]
        src_dir = paths.object_images_dir(name)
        prompt_slug = sanitize_filename(prompt)

        img_out = paths.object_images_link_dir(name, prompt)
        mask_out = paths.object_masks_dir(name, prompt)
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        write_object_prompt_audit(name, prompt, paths)

        if not src_dir.exists():
            print(f"[step1] [{name}] SKIP: {src_dir} 不存在")
            continue

        # Cache hit: skip if mask dir already has the right number of masks (or REBUILD=1).
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        existing_imgs = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
        existing_masks = list(mask_out.glob("*_mask.png"))
        if not force and existing_masks and len(existing_masks) >= len(existing_imgs):
            print(f"[step1] [{name}] cache hit: {len(existing_masks)} mask(s) in {mask_out}; "
                  f"REBUILD=1 to force.")
            summary[name] = {"masks_dir": str(mask_out), "ok": len(existing_masks), "cached": True}
            continue

        image_files = sorted(
            f for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        )
        print(f"\n[step1] [{name}] {len(image_files)} images  prompt='{prompt}'")

        per_obj_ok = 0
        for img_path in image_files:
            stem = img_path.stem
            image = Image.open(img_path).convert("RGB")

            state = processor.set_image(image)
            state_copy = state.copy()
            state_copy["backbone_out"] = state["backbone_out"].copy()

            output = processor.set_text_prompt(state=state_copy, prompt=prompt)
            scores = output["scores"]
            masks = output["masks"]

            if len(scores) == 0 or scores.max().item() < MASK_CONFIDENCE:
                top = scores.max().item() if len(scores) else 0
                print(f"  {stem}: no mask (max_score={top:.3f})  -> zeros")
                binary_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                best_idx = int(torch.argmax(scores).item())
                best_mask = masks[best_idx]
                if best_mask.ndim == 3:
                    best_mask = best_mask.squeeze(0)
                binary_mask = (best_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
                print(f"  {stem}: ok (score={scores[best_idx].item():.3f})")
                per_obj_ok += 1

            # SAM3D (MV-SAM3D notebook/load_images_and_masks.py:30-40) 要求 mask
            # 是 **RGBA** 格式, alpha 通道当 mask (RGB 通道不读)。
            # 这里跟 MV-SAM3D/run.sh 官方 mask 格式对齐: RGB = 原图但背景置零,
            # alpha = binary_mask。这样人眼打开 PNG 能直接看到抠出的物体,
            # 方便手动检查 mask 内容是否正确。
            H_m, W_m = binary_mask.shape
            img_rgb = np.array(image, dtype=np.uint8)  # (H, W, 3), image 来自 line 223
            mask_bool = binary_mask > 0
            rgba = np.zeros((H_m, W_m, 4), dtype=np.uint8)
            rgba[..., :3] = np.where(mask_bool[..., None], img_rgb, 0)
            rgba[..., 3] = binary_mask    # SAM3D 实际读这个 (alpha > 0)
            Image.fromarray(rgba, mode="RGBA").save(mask_out / f"{stem}_mask.png")

            dst = img_out / img_path.name
            if not dst.exists():
                try:
                    os.link(img_path, dst)
                except OSError:
                    from shutil import copy2
                    copy2(img_path, dst)

        summary[name] = {"masks_dir": str(mask_out), "ok": per_obj_ok}

    print(f"\n[step1] done. build/objects/ dirs:")
    for name, info in summary.items():
        print(f"  {name}: {info}")
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────
# Step 2: 打印 SAM3D 命令                       (env: sam3d-objects)
# ─────────────────────────────────────────────────────────────────────

def step2_sam3d_commands():
    banner("2", "sam3d-objects", "Print SAM3D mesh-reconstruction commands")
    paths = resolve_paths()
    print(f"[step2] 请确认当前 env 为 sam3d-objects (pytorch3d + sam3d_objects)。")
    print(f"[step2] 切到 MV-SAM3D/ 目录跑下面的命令；输出会落到")
    print(f"        MV-SAM3D/visualization/<basename(input_path)>/...\n")

    print(f"cd {MV_SAM3D_DIR}\n")
    for obj in OBJECTS:
        name = obj["name"]
        prompt = obj["prompt"]
        prompt_slug = sanitize_filename(prompt)
        input_path = paths.build_object_dir(name, prompt).resolve()
        img_dir = paths.object_images_link_dir(name, prompt)
        mask_dir = paths.object_masks_dir(name, prompt)

        if not img_dir.exists() or not mask_dir.exists():
            print(f"# [{name}] SKIP: step 1 产物缺失 ({img_dir} 或 {mask_dir})")
            continue

        img_files = sorted(
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        n = len(img_files)
        if n == 0:
            print(f"# [{name}] SKIP: 没图")
            continue
        image_names = ",".join(str(i) for i in range(n))

        print(f"# ── {name} ({n} views) ──")
        print(
            f"python {SAM3D_SCRIPT} \\\n"
            f"    --input_path {input_path} \\\n"
            f"    --mask_prompt {prompt_slug} \\\n"
            f"    --image_names {image_names}\n"
        )

    print(f"# 预期输出: MV-SAM3D/visualization/<input_basename>/<prompt_slug>/<dated>/result.glb")
    print(f"# fast_start.sh 会在跑完 SAM3D 后, 把最新 result.glb symlink 到")
    print(f"#   <build_object_dir>/mesh.glb, 后续 step3 系列从这里读 mesh.\n")


# ─────────────────────────────────────────────────────────────────────
# Step 3a: 场景图 → SAM3 分割 → 写盘            (env: sam3)
# ─────────────────────────────────────────────────────────────────────

def step3a_scene_segmentation(scene_image: str, scene_prompt: str):
    banner("3a", "sam3", "Scene segmentation (SAM3) -> build/scenes/<scene>/prompts/<prompt_id>/scene_masks.pt")
    assert_env("sam3", ["sam3.model_builder", "sam3.model.sam3_image_processor"])

    import torch
    from PIL import Image, ImageDraw
    from real2sim.perception.segmentation import GroundedSAM

    paths = resolve_paths()
    force = should_rebuild()
    paths.build_prompt_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_audit(paths)

    out_path = paths.scene_masks_path
    overlay_path = paths.scene_masks_overlay_path
    inpaint_path = paths.inpainted_image_path

    if not force and out_path.exists() and inpaint_path.exists():
        print(f"[step3a] cache hit: {out_path} + {inpaint_path} exist; REBUILD=1 to force.")
        return

    print(f"[step3a] scene image = {scene_image}")
    print(f"[step3a] prompt      = '{scene_prompt}'")
    print(f"[step3a] output dir  = {paths.build_prompt_dir}")

    image = Image.open(scene_image).convert("RGB")
    sam = GroundedSAM(device="cuda")
    seg = sam.segment(image, scene_prompt, expand_mask_iters=0, expand_kernel=15)

    masks = seg.masks.cpu()
    boxes = seg.boxes.cpu()
    scores = seg.scores.cpu()
    labels = seg.labels
    print(f"[step3a] detected {masks.shape[0]} object(s):")
    for i, lab in enumerate(labels):
        print(f"  [{i}] '{lab}'  score={scores[i].item():.3f}")

    torch.save(
        {
            "image_path": str(scene_image),
            "prompt": scene_prompt,
            "masks": masks.bool(),     # (N, H, W)
            "boxes": boxes,            # (N, 4)
            "scores": scores,
            "labels": labels,
        },
        out_path,
    )
    print(f"[step3a] saved -> {out_path}")

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, b in enumerate(boxes.tolist()):
        draw.rectangle(b, outline=(255, 0, 0), width=3)
        draw.text((b[0] + 4, b[1] + 4), labels[i], fill=(255, 0, 0))
    vis.save(overlay_path)
    print(f"[step3a] overlay -> {overlay_path}")

    # ── Inpaint background here (sam3 env can pip install LaMa).
    # step 3c (sam3d-objects env) loads the result and skips its own inpaint.
    if masks.shape[0] > 0:
        try:
            from real2sim.perception.inpainting import LaMaInpainter

            combined_mask = masks.any(dim=0)
            inp_device = "cuda" if torch.cuda.is_available() else "cpu"
            inpainter = LaMaInpainter(device=inp_device)
            if not inpainter.model_available:
                print("[step3a] LaMa not installed — skip inpaint."
                      " step 3c will fall back to original image for depth.")
            else:
                inpainted = inpainter.inpaint(image, combined_mask, refine_iters=0)
                inpainted.save(inpaint_path)
                print(f"[step3a] inpainted -> {inpaint_path}")
        except Exception as e:
            print(f"[step3a] inpaint failed ({e}); step 3c will fall back to original image.")
    else:
        print("[step3a] no masks; nothing to inpaint.")


# ─────────────────────────────────────────────────────────────────────
# Step 3b: 场景图 → SAM3D 单图 metric pointmap  (env: sam3d-objects)
# ─────────────────────────────────────────────────────────────────────

def step3b_scene_pointmap(scene_image: str, user_K: Optional[List[float]] = None):
    """MoGe single-image metric pointmap.

    user_K: 可选 [fx, fy, cx, cy]，单位像素。提供后：
        (a) 由 fx + 图像宽推 horizontal FoV，作为 MoGe.infer(fov_x=...) 的提示；
        (b) 保存到 scene_intrinsics.npy 的也是用户的 K，不是 MoGe 预测的。

    Output: paths.build_moge_dir/{scene_pointmap.npy, scene_intrinsics.npy}
    (scene-only, prompt-independent → 同一 scene 不同 prompt 复用).
    """
    banner("3b", "sam3d-objects", "Scene metric pointmap via MoGe")
    assert_env("sam3d-objects", ["torch", "moge.model.v1"])

    import math
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v1 import MoGeModel

    paths = resolve_paths()
    force = should_rebuild()
    paths.build_moge_dir.mkdir(parents=True, exist_ok=True)
    pm_path = paths.scene_pointmap_path
    K_path = paths.scene_intrinsics_path

    if not force and pm_path.exists() and K_path.exists() and user_K is None:
        print(f"[step3b] cache hit: {pm_path} + {K_path} exist; REBUILD=1 to force.")
        print(f"         (note: cache is keyed on scene_id only; pass user K with --K to force re-run.)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(scene_image).convert("RGB")
    W, H = image.size
    image_tensor = (
        torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    ).to(device)
    print(f"[step3b] image = {scene_image}  {W}x{H}")

    fov_x_deg = None
    if user_K is not None:
        fx_u, fy_u, cx_u, cy_u = user_K
        fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * fx_u)))
        print(f"[step3b] user K = fx={fx_u:.1f} fy={fy_u:.1f} cx={cx_u:.1f} cy={cy_u:.1f}")
        print(f"[step3b] → fov_x = {fov_x_deg:.2f}° (fed to MoGe as prior)")

    print("[step3b] loading MoGe (Ruicheng/moge-vitl)...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    with torch.no_grad():
        if fov_x_deg is not None:
            out = moge.infer(image_tensor, fov_x=fov_x_deg)
        else:
            out = moge.infer(image_tensor)

    # MoGe raw output: standard camera space (X right, Y down, Z fwd, metric meters).
    # pipeline.py 期望 PyTorch3D 相机系 → (x, y, z) -> (-x, -y, z)。
    # 但我们的 pipeline 当前只用 pointmap[..., 2] (Z 切片)，Z 不变。
    # 为保持与 docstring 一致，还是显式翻转 X/Y。
    points = out["points"]  # (H, W, 3)
    points_p3d = points.clone()
    points_p3d[..., 0] = -points[..., 0]
    points_p3d[..., 1] = -points[..., 1]

    if user_K is not None:
        fx_abs, fy_abs, cx_abs, cy_abs = user_K
        print(f"[step3b] using user-supplied K")
    else:
        # MoGe intrinsics 是 normalized (除以 W/H)，还原成像素单位。
        # 参考 MV-SAM3D/notebook/mesh_alignment.py: force fx = fy = fy_norm * H。
        K_norm = out["intrinsics"].cpu().numpy()
        fy_abs = float(K_norm[1, 1]) * H
        fx_abs = fy_abs
        cx_abs = float(K_norm[0, 2]) * W
        cy_abs = float(K_norm[1, 2]) * H
        print(f"[step3b] using MoGe-predicted K")
    K_abs = np.array(
        [[fx_abs, 0.0, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.save(pm_path, points_p3d.cpu().numpy().astype(np.float32))
    np.save(K_path, K_abs)

    z = points_p3d[..., 2]
    print(f"[step3b] saved {pm_path}  shape={tuple(points_p3d.shape)}")
    print(f"[step3b] saved {K_path}  fx={fx_abs:.1f} fy={fy_abs:.1f} cx={cx_abs:.1f} cy={cy_abs:.1f}")
    print(f"[step3b] Z range = [{z.min().item():.3f}, {z.max().item():.3f}] m  median={z.median().item():.3f} m")


# ─────────────────────────────────────────────────────────────────────
# Step 3c: Real2Sim 位姿优化                    (env: sam3d-objects, world-frame)
# ─────────────────────────────────────────────────────────────────────


def _apply_per_object_env_overrides(entry, matched_name, verts, device, template_path):
    """把 step3c per-object 的 env-var hook 集中处理 (world-frame mode).

    支持:
      INIT_XY_<name>=tx,ty        手动指定 init t_x, t_y (world frame, meters);
                                  顺便 skip closed-form 反推 (否则会被覆盖)
      INIT_SCALE_<name>=<float>   覆盖 init_scale
      FREEZE_SCALE_<name>=1       跳 Stage 2 + refine, scale 永远 = init_scale
      YAW_ONLY[/_<name>]=1        R 退到 1DOF (绕 world +Z yaw), R = R_yaw_z(θ)
      FLOOR_LOCK[/_<name>]=1      t 锁: t_z = plane_z - s · z_min, 只优化 (t_x, t_y)
    """
    xy_key = f"INIT_XY_{matched_name}"
    if xy_key in os.environ:
        raw = os.environ[xy_key].strip()
        try:
            parts = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
            if len(parts) != 2:
                raise ValueError(f"expected 2 floats, got {len(parts)}")
        except Exception as e:
            print(f"     [warn] {xy_key}='{raw}' 解析失败 ({e}), 忽略")
        else:
            tx, ty = parts
            import torch as _t
            old = entry["init_t"]
            entry["init_t"] = _t.tensor(
                [tx, ty, float(old[2].item())],
                device=old.device, dtype=old.dtype,
            )
            entry["skip_closed_form"] = True
            print(f"     init_xy 被 {xy_key}=({tx:+.3f},{ty:+.3f}) 覆盖, 跳 closed-form")

    init_key = f"INIT_SCALE_{matched_name}"
    if init_key in os.environ:
        v = float(os.environ[init_key])
        entry["init_scale"] = v
        print(f"     init_scale 被 {init_key}={v} 覆盖")

    freeze_key = f"FREEZE_SCALE_{matched_name}"
    if freeze_key in os.environ and os.environ[freeze_key] == "1":
        entry["freeze_scale"] = True
        print(f"     freeze_scale 启用 ({freeze_key}=1, scale 钉死在 {entry.get('init_scale', 1.0)})")

    yaw_key = f"YAW_ONLY_{matched_name}"
    yaw_on = (yaw_key in os.environ and os.environ[yaw_key] == "1") \
             or os.environ.get("YAW_ONLY", "0") == "1"
    if yaw_on:
        entry["yaw_only"] = True
        print(f"     yaw_only 启用 (R = R_yaw_z(θ), 绕 world +Z 转, 1 DOF)")

    floor_key = f"FLOOR_LOCK_{matched_name}"
    floor_on = (floor_key in os.environ and os.environ[floor_key] == "1") \
               or os.environ.get("FLOOR_LOCK", "0") == "1"
    if floor_on and yaw_on:
        # mesh-after-M_LOAD 的 up 是 +Z, mug 底是 mesh 最低 +Z 点
        z_min = float(verts[:, 2].min().item())
        entry["floor_constraint"] = {
            "y_min": z_min,    # 字段名保留历史, 实际是 mesh +Z 方向最低点
        }
        print(f"     floor_lock 启用 (t_z = plane_z - s · {z_min:.4f}, 只优化 t_x, t_y)")
    elif floor_on and not yaw_on:
        print(f"     [warn] FLOOR_LOCK 要求 YAW_ONLY=1, 跳过")


def step3c_real2sim(
    scene_image: str,
    scene_prompt: str,
    mesh_paths: Dict[str, Path],
    output_dir: Path,
):
    banner("3c", "sam3d-objects",
           "Real2Sim pose optimization (uses precomputed mask + optional pointmap)")
    assert_env("sam3d-objects", ["torch", "pytorch3d", "trimesh"])

    import numpy as np
    import torch
    from real2sim import Real2SimPipeline, PipelineConfig

    paths = resolve_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_file = paths.scene_masks_path
    if not mask_file.exists():
        print(f"[ERROR] {mask_file} 不存在。请先在 sam3 env 跑 `--step 3a`。")
        sys.exit(2)
    bundle = torch.load(mask_file, map_location="cpu", weights_only=False)
    print(f"[step3c] loaded scene_masks.pt: {bundle['masks'].shape[0]} mask(s)")
    if bundle["image_path"] != str(scene_image):
        print(f"[step3c] [warn] scene image 路径与 step3a 产物不一致：")
        print(f"   step3a: {bundle['image_path']}")
        print(f"   step3c: {scene_image}")

    pointmap_file = paths.scene_pointmap_path
    metric_pointmap = None
    if pointmap_file.exists():
        metric_pointmap = torch.from_numpy(np.load(pointmap_file)).float()
        print(f"[step3c] loaded scene_pointmap.npy {tuple(metric_pointmap.shape)}")
    else:
        print(f"[step3c] {pointmap_file} 不存在，回退 DepthAnything (Z 仍为 2.0 估计)")

    # ── World-frame setup ───────────────────────────────────────────────
    # 整个优化在 Genesis world (z-up) 里跑, 相机用 single_pos / single_lookat / single_up 摆好.
    # R, t 直接是 mesh→Genesis-world, step5 不需要 compose_pose 转换. 跟
    # scripts/render_pt3d_check.py 完全一致的几何处理. K 也从 single_fov 反推.
    template_path = PROJECT_ROOT / "example" / "1.json"
    H_scene, W_scene = bundle["masks"].shape[-2:]
    world_camera_params = pt3d_world_camera_from_genesis(template_path, H_scene, W_scene, device)
    if world_camera_params is None:
        print(f"[step3c] [fatal] 读不到 {template_path}, 退出")
        sys.exit(2)
    print(f"[step3c] world frame: cam_pos={world_camera_params['cam_pos'].tolist()}, "
          f"plane_z={world_camera_params['plane_z']}, "
          f"fx={world_camera_params['fx']:.1f}, image={W_scene}x{H_scene}")

    # K 单独构造一份 (pipeline.py 还接受 K 当参数, 但优化器 in world-frame mode 实际从
    # world_camera_params 取). 给它填一个跟 world_camera_params 同步的, 避免老 code path 报错.
    wcp = world_camera_params
    K = torch.tensor(
        [[wcp["fx"], 0, wcp["cx"]],
         [0, wcp["fy"], wcp["cy"]],
         [0, 0, 1]], dtype=torch.float32,
    )

    inpaint_file = paths.inpainted_image_path
    precomputed_inpainted = None
    if inpaint_file.exists():
        from PIL import Image as _Image
        precomputed_inpainted = _Image.open(inpaint_file).convert("RGB")
        print(f"[step3c] loaded inpainted.png {precomputed_inpainted.size}")
    else:
        print(f"[step3c] {inpaint_file} 不存在；pipeline 会用原图喂 DepthAnything")

    labels = bundle["labels"]
    masks = bundle["masks"]
    print(f"[step3c] scene labels = {labels}")

    # 允许 prompt 跟 name 不同 (例如 name='black_mug_tree' 但 prompt='black straight object'),
    # 这种情况下 scene label slug 不会匹配 name, 但应该匹配 prompt_slug。
    name_to_prompt_slug = {
        obj["name"]: sanitize_filename(obj["prompt"]).lower() for obj in OBJECTS
    }

    mesh_data_list = []
    precomputed_masks = []
    # label→mesh 匹配: 选 best-score 而不是 first-match.
    # 之前 first-match 在 prompt 嵌套时翻车: slug='black_straight_mug_tree' 撞到
    # blue_mug 的 prompt='mug' (子串成立), 直接 break, tree 的 mask 错配到 blue_mug.
    # 现在排名:  prompt_slug 完全相等 (4) > name 完全相等 (3) > name 子串 (2) > prompt 子串 (1)
    # 同级再比 match 长度 (越长越特异), 保证 tree label 只能赢 black_mug_tree.
    def _match_rank(slug, name_l, prompt_slug):
        if prompt_slug and prompt_slug == slug:
            return (4, len(prompt_slug))
        if name_l == slug:
            return (3, len(name_l))
        best = (0, 0)
        if slug in name_l:
            best = max(best, (2, len(slug)))
        if name_l in slug:
            best = max(best, (2, len(name_l)))
        if prompt_slug:
            if slug in prompt_slug:
                best = max(best, (1, len(slug)))
            if prompt_slug in slug:
                best = max(best, (1, len(prompt_slug)))
        return best

    for label, mask in zip(labels, masks):
        slug = sanitize_filename(label).lower()
        best_name = None
        best_rank = (0, 0)
        for name, _ in mesh_paths.items():
            name_l = name.lower()
            prompt_slug = name_to_prompt_slug.get(name, "")
            rank = _match_rank(slug, name_l, prompt_slug)
            if rank > best_rank:
                best_rank = rank
                best_name = name
        if best_name is None:
            print(f"  [warn] label='{label}' (slug='{slug}') 没找到匹配的 mesh，跳过")
            continue
        matched_name = best_name

        glb_path = mesh_paths[matched_name]
        print(f"  '{label}' -> mesh {matched_name} <- {glb_path}")
        verts, faces, vc = load_glb_as_pytorch3d(glb_path, device)
        entry = {"verts": verts, "faces": faces, "name": matched_name}
        if vc is not None:
            entry["vertex_colors"] = vc

        # ── Init pose (world-frame) ─────────────────────────────────────
        # init_R = I, 8 个绕 world +Z 的 yaw candidates 覆盖 360°. optimizer 选 IoU 最好的.
        # init_t = cam_lookat 的 xy + plane_z + 微调 (floor_lock 会重算 t_z).
        # 不再用 step3e 的 ICP init (那是 PT3D-cam frame 的, 跟 world frame 不对应).
        entry["world_camera_params"] = world_camera_params
        entry["init_R"] = torch.eye(3, device=device)
        candidates_list = []
        for k in range(8):
            theta = 2.0 * 3.141592653589793 * k / 8.0
            candidates_list.append(yaw_rotation_matrix(theta, device))
        entry["init_R_candidates"] = candidates_list

        import json as _json
        tpl_cfg = _json.loads(template_path.read_text())
        lookat = tpl_cfg["env"]["camera"]["single_lookat"]
        plane_z_val = world_camera_params["plane_z"]
        entry["init_t"] = torch.tensor(
            [float(lookat[0]), float(lookat[1]), float(plane_z_val) + 0.05],
            device=device, dtype=torch.float32,
        )
        # init_scale 这里不设 (走 OptimizerConfig 默认 1.0); 想 override 用 INIT_SCALE_<name> env
        print(f"     init_R = I, init_t = {entry['init_t'].tolist()}, "
              f"8 个绕 world +Z yaw candidates")

        # debug viz dir: optimizer 会在这里存 {name}_init.png (init pose render)
        entry["debug_save_dir"] = str(paths.run_dir / "per_object")

        _apply_per_object_env_overrides(entry, matched_name, verts, device, template_path)
        mesh_data_list.append(entry)
        precomputed_masks.append({"mask": mask, "label": label})

    if not mesh_data_list:
        print("[step3c] 没有任何物体可优化，退出。")
        sys.exit(1)

    config = PipelineConfig()
    config.output_dir = str(output_dir)
    # per_object/ eval visualizations go into the per-run run_dir, not build/,
    # so each run keeps a self-contained QA folder next to comparison/ and
    # gsrl_config.json. Everything else (scene.json etc.) stays in build/.
    config.viz_output_dir = str(paths.run_dir)
    config.segmentation.text_prompt = scene_prompt
    config.segmentation.expand_mask_iters = 0
    config.segmentation.expand_kernel = 15
    config.optimizer.stages = 3
    config.optimizer.iters_per_stage = 300
    config.save_intermediate = True

    # 测试钩子: FREEZE_SCALE=1 → optimizer 跳过 Stage 2 + refine, scale 永远等于 init_scale.
    # 配合 INIT_SCALE_<name>=<v> 一起用, 比如固定 mug_1_tripo @ scale=1.2 求 R/t.
    if os.environ.get("FREEZE_SCALE", "0") == "1":
        config.optimizer.freeze_scale = True
        print(f"[step3c] FREEZE_SCALE=1 → optimizer 跳过 Stage 2, scale 钉死在 init_scale")

    # STAGE1_LEARN_SCALE=1: Stage 1 把 scale 一起学 (Adam), 跳过 Stage 2 area_ratio.
    # 需要 yaw_only + floor_lock 一起开. 用 IoU loss 直接管 scale, 比 area_ratio 鲁棒.
    if os.environ.get("STAGE1_LEARN_SCALE", "0") == "1":
        config.optimizer.learn_scale = True
        print(f"[step3c] STAGE1_LEARN_SCALE=1 → Stage 1 学 s (Adam, exp 参数化), "
              f"跳过 Stage 2 area_ratio")

    # STAGE1_TXY_GRID=N (推荐 N=5): Stage 1 在 init_t.xy 周围撒 N×N grid 起点,
    # 每个起点 × 8 yaw 都跑一遍, 选 IoU 最高的当 winner. 用来逃 silhouette loss
    # 的多 basin (非凸物体易卡 wrong basin, 单 init 救不回来).
    # 时间代价 × N². N=5 → 25 grid × 8 yaw = 200 candidates, ~2 min/object.
    # STAGE1_TXY_GRID_RADIUS=<m> 控制 grid 半径, 默认 0.5m.
    grid_n = int(os.environ.get("STAGE1_TXY_GRID", "1"))
    if grid_n > 1:
        config.optimizer.stage1_txy_grid = grid_n
        config.optimizer.stage1_txy_grid_radius = float(
            os.environ.get("STAGE1_TXY_GRID_RADIUS", "0.5")
        )
        print(f"[step3c] STAGE1_TXY_GRID={grid_n} → Stage 1 撒 {grid_n}×{grid_n}={grid_n*grid_n} "
              f"(tx,ty) 起点, 半径 ±{config.optimizer.stage1_txy_grid_radius:.2f}m. "
              f"总 cand = 8 yaw × {grid_n*grid_n} = {8*grid_n*grid_n}")

    # Stage 1 iter 数放大 (做调试用). 默认 1.0 = 80 / 60 iter.
    # 跑 STAGE1_ITER_SCALE=10 bash fast_start.sh 3c → 800 / 600 iter.
    # 跑 STAGE1_ITER_SCALE=100 → 8000 / 6000 iter (1w 量级).
    iter_scale = float(os.environ.get("STAGE1_ITER_SCALE", "1.0"))
    if iter_scale != 1.0:
        config.optimizer.stage1_iters_coarse = int(config.optimizer.stage1_iters_coarse * iter_scale)
        config.optimizer.stage1_iters_fine   = int(config.optimizer.stage1_iters_fine   * iter_scale)
        print(f"[step3c] STAGE1_ITER_SCALE={iter_scale} → "
              f"coarse={config.optimizer.stage1_iters_coarse}, "
              f"fine={config.optimizer.stage1_iters_fine}")

    pipeline = Real2SimPipeline(config)
    results = pipeline.run(
        image_path=scene_image,
        text_prompt=scene_prompt,
        mesh_data=mesh_data_list,
        precomputed_masks=precomputed_masks,
        metric_pointmap=metric_pointmap,
        precomputed_inpainted=precomputed_inpainted,
        K=K,
    )

    print(f"\n[step3c] results saved to {output_dir}")
    for obj in results.get("objects", []):
        t = obj.get("t", [])
        if hasattr(t, "tolist"):
            t = t.tolist()
        print(f"  {obj.get('name', '?')}: scale={obj.get('scale', 0):.3f}  t={t}")

    serializable = _make_serializable(results)
    with open(output_dir / "pipeline_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────
# Step 4: 渲染对比                              (env: sam3d-objects)
# 实现在 real2sim/render_compare.py, 这里只是 banner + env check 的 wrapper.
# ─────────────────────────────────────────────────────────────────────

def step4_render(scene_image: str, results: Dict, mesh_paths: Dict[str, Path], output_dir: Path):
    banner("4", "sam3d-objects", "Rendering & comparison")
    assert_env("sam3d-objects", ["torch", "pytorch3d", "trimesh"])
    template_path = PROJECT_ROOT / "example" / "1.json"
    _step4_render_impl(scene_image, results, mesh_paths, output_dir,
                       template_path=template_path if template_path.exists() else None)


# ─────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────

def main():
    paths = resolve_paths()
    parser = argparse.ArgumentParser(description="Real2Sim Full Workflow (双环境版)")
    parser.add_argument("--step", required=True,
                        choices=["1", "2", "3a", "3b", "3c", "4"],
                        help="一次只跑一步 (因为不同步骤需要不同 conda env)")
    parser.add_argument("--scene", type=str, default=None,
                        help="场景图路径 (default: pipeline_paths.scene_image)")
    parser.add_argument("--scene-prompt", type=str, default=SCENE_PROMPT,
                        help="场景中物体文字 prompt")
    parser.add_argument("--meshes", type=str, nargs="*", default=None,
                        help="直接指定 mesh: name=path 形式 (step 3c/4); "
                             "不传则走 pipeline_paths.object_mesh_link.")
    parser.add_argument("--output", type=str, default=None,
                        help="step 4 的输出目录 (comparison/ 等). 默认 = pipeline_paths.run_dir. "
                             "step 3c 的输出 (scene.json / pipeline_results.json / pt3d_intermediate/) "
                             "固定走 paths.build_prompt_dir, 不受 --output 影响.")
    parser.add_argument("--K", type=float, nargs=4, default=None,
                        metavar=("FX", "FY", "CX", "CY"),
                        help="(step 3b) 已知相机内参 (像素单位)。"
                             "提供后由 fx+宽推 fov_x 喂给 MoGe 当先验，"
                             "并把这个 K 存为 scene_intrinsics.npy。"
                             "不传则让 MoGe 自己估。")
    args = parser.parse_args()

    setup_logging(args.step)

    step = args.step
    scene_image = args.scene if args.scene else str(paths.scene_image)
    scene_prompt = args.scene_prompt
    # step 3c 写 build_prompt_dir (cross-run-deterministic 数据, 不放 run_dir).
    # step 4  写 run_dir (或 --output 指定的目录), 读 step3c 产物从 build_prompt_dir.
    output_dir = Path(args.output) if args.output else paths.run_dir

    print(f"[main] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[main] step         = {step}")
    print(f"[main] scene_id     = {paths.scene_id}")
    print(f"[main] prompt_id    = {paths.prompt_id}")
    print(f"[main] scene image  = {scene_image}")
    print(f"[main] run dir      = {paths.run_dir}")
    if step == "3c":
        print(f"[main] step3c → {paths.build_prompt_dir}  (scene.json / pipeline_results.json)")
    if step == "4":
        print(f"[main] step4 reads {paths.pipeline_results_path}")
        print(f"[main] step4 writes {output_dir}/comparison/")

    if step == "1":
        step1_object_masks()
        return

    if step == "2":
        step2_sam3d_commands()
        return

    if step == "3a":
        step3a_scene_segmentation(scene_image, scene_prompt)
        return

    if step == "3b":
        step3b_scene_pointmap(scene_image, user_K=args.K)
        return

    object_names = [o["name"] for o in OBJECTS]
    glb_paths = find_glb_files(object_names, paths=paths)
    if args.meshes:
        for item in args.meshes:
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                p = Path(item)
                name = p.parent.name or p.stem
                path = item
            glb_paths[name] = Path(path)

    if not glb_paths:
        print("\n[ERROR] 没有找到任何 .glb。先跑 step 2 (会自动 symlink 到")
        print(f"        {paths.build_root}/objects/<obj>__<hash>/mesh.glb)，")
        print("        或用 --meshes name=path 直接指定。")
        sys.exit(1)
    print(f"[main] mesh paths:")
    for n, p in glb_paths.items():
        print(f"  {n}: {p}")

    if step == "3c":
        # 默认写 build_prompt_dir (跨 run cacheable); --output 给测试场景一个 escape hatch.
        step3c_out = Path(args.output) if args.output else paths.build_prompt_dir
        step3c_real2sim(scene_image, scene_prompt, glb_paths, step3c_out)
        return

    if step == "4":
        results_file = paths.pipeline_results_path
        if not results_file.exists():
            print(f"[ERROR] {results_file} 不存在，请先跑 --step 3c。")
            sys.exit(2)
        import torch
        with open(results_file) as f:
            results = json.load(f)
        for obj in results.get("objects", []):
            obj["R"] = torch.tensor(obj["R"])
            obj["t"] = torch.tensor(obj["t"])
        step4_render(scene_image, results, glb_paths, paths.comparison_dir)
        return


if __name__ == "__main__":
    main()
