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
# 路径常量 —— 全部相对于 R2SConfig/ (本脚本所在目录)
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent          # R2SConfig/
MV_SAM3D_DIR = PROJECT_ROOT / "MV-SAM3D"                # 别人的仓库，不动
SAM3_DIR = PROJECT_ROOT / "sam3"                        # 别人的仓库，不动
WORK_DIR = PROJECT_ROOT / "workdir"                     # 中间产物
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_workflow"  # 最终产物
LOG_DIR = PROJECT_ROOT / "logs"
VISUALIZATION_DIR = MV_SAM3D_DIR / "visualization"      # SAM3D 输出落点

# 让 real2sim 包可被 import (顶层目录 == sys.path 第一个)
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────
# 配置区 —— 按你的实验修改这里
# ─────────────────────────────────────────────────────────────────────

# 每个物体: 名称, 多视角图片目录 (绝对路径), SAM3 mask prompt
# 默认从 R2SConfig/assets/ 读 (我们自己的素材，不动 MV-SAM3D/assets/)。
OBJECTS: List[Dict] = [
    {"name": "red_mug",        "image_dir": PROJECT_ROOT / "assets/red_mug/images",        "prompt": "red mug"},
    {"name": "black_mug_tree", "image_dir": PROJECT_ROOT / "assets/black_mug_tree/images", "prompt": "black straight object"},
]

# 场景定义
SCENE_IMAGE = PROJECT_ROOT / "assets/scene.jpg"
# 跟 OBJECTS prompt 保持一致, 避免 step3a 出的 scene mask 跟 step1+step2 重建的 mesh 对不上
# (例如 "black mug tree" 在场景里会分割整套, 但我们 mesh 只重建了竖杆 "black straight object")。
SCENE_PROMPT = "red mug. black straight object."

# SAM3D 入口
SAM3D_SCRIPT = "run_inference_weighted.py"   # 相对 MV-SAM3D/

# Mask 置信度
MASK_CONFIDENCE = 0.1


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

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


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

    WORK_DIR.mkdir(parents=True, exist_ok=True)
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
        src_dir = Path(obj["image_dir"])
        prompt_slug = sanitize_filename(prompt)

        obj_dir = WORK_DIR / name
        img_out = obj_dir / "images"
        mask_out = obj_dir / prompt_slug
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[step1] [{name}] SKIP: {src_dir} 不存在")
            continue

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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

    print(f"\n[step1] done. workdir = {WORK_DIR}")
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────
# Step 2: 打印 SAM3D 命令                       (env: sam3d-objects)
# ─────────────────────────────────────────────────────────────────────

def step2_sam3d_commands():
    banner("2", "sam3d-objects", "Print SAM3D mesh-reconstruction commands")
    print(f"[step2] 请确认当前 env 为 sam3d-objects (pytorch3d + sam3d_objects)。")
    print(f"[step2] 切到 MV-SAM3D/ 目录跑下面的命令；输出会落到")
    print(f"        {VISUALIZATION_DIR.relative_to(PROJECT_ROOT)}/...\n")

    print(f"cd {MV_SAM3D_DIR}\n")
    for obj in OBJECTS:
        name = obj["name"]
        prompt_slug = sanitize_filename(obj["prompt"])
        input_path = (WORK_DIR / name).resolve()
        img_dir = input_path / "images"
        mask_dir = input_path / prompt_slug

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

    print(f"# 预期输出: {VISUALIZATION_DIR}/<obj_name>/<prompt_slug>/.../result.glb")
    print(f"# 和         {VISUALIZATION_DIR}/<obj_name>/<prompt_slug>/.../params.npz\n")


# ─────────────────────────────────────────────────────────────────────
# Step 3a: 场景图 → SAM3 分割 → 写盘            (env: sam3)
# ─────────────────────────────────────────────────────────────────────

def step3a_scene_segmentation(scene_image: str, scene_prompt: str):
    banner("3a", "sam3", "Scene segmentation (SAM3) -> workdir/scene_masks.pt")
    assert_env("sam3", ["sam3.model_builder", "sam3.model.sam3_image_processor"])

    import torch
    from PIL import Image, ImageDraw
    from real2sim.segmentation import GroundedSAM

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[step3a] scene image = {scene_image}")
    print(f"[step3a] prompt      = '{scene_prompt}'")

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

    out_path = WORK_DIR / "scene_masks.pt"
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
    vis_path = WORK_DIR / "scene_masks_overlay.png"
    vis.save(vis_path)
    print(f"[step3a] overlay -> {vis_path}")

    # ── Inpaint background here (sam3 env can pip install LaMa).
    # step 3c (sam3d-objects env) loads the result and skips its own inpaint.
    if masks.shape[0] > 0:
        try:
            from real2sim.inpainting import LaMaInpainter

            combined_mask = masks.any(dim=0)
            inp_device = "cuda" if torch.cuda.is_available() else "cpu"
            inpainter = LaMaInpainter(device=inp_device)
            if not inpainter.model_available:
                print("[step3a] LaMa not installed — skip inpaint."
                      " step 3c will fall back to original image for depth.")
            else:
                inpainted = inpainter.inpaint(image, combined_mask, refine_iters=0)
                inpaint_path = WORK_DIR / "inpainted.png"
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
    """
    banner("3b", "sam3d-objects", "Scene metric pointmap via MoGe")
    assert_env("sam3d-objects", ["torch", "moge.model.v1"])

    import math
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v1 import MoGeModel

    WORK_DIR.mkdir(parents=True, exist_ok=True)
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

    pm_path = WORK_DIR / "scene_pointmap.npy"
    K_path = WORK_DIR / "scene_intrinsics.npy"
    np.save(pm_path, points_p3d.cpu().numpy().astype(np.float32))
    np.save(K_path, K_abs)

    z = points_p3d[..., 2]
    print(f"[step3b] saved {pm_path}  shape={tuple(points_p3d.shape)}")
    print(f"[step3b] saved {K_path}  fx={fx_abs:.1f} fy={fy_abs:.1f} cx={cx_abs:.1f} cy={cy_abs:.1f}")
    print(f"[step3b] Z range = [{z.min().item():.3f}, {z.max().item():.3f}] m  median={z.median().item():.3f} m")


# ─────────────────────────────────────────────────────────────────────
# Step 3c: Real2Sim 位姿优化                    (env: sam3d-objects)
# ─────────────────────────────────────────────────────────────────────

def find_glb_files(mesh_dir: Path, object_names: List[str]) -> Dict[str, Path]:
    found = {}
    for name in object_names:
        direct = mesh_dir / f"{name}.glb"
        if direct.exists():
            found[name] = direct
            continue
        sub = mesh_dir / name / "result.glb"
        if sub.exists():
            found[name] = sub
            continue
        candidates = list(mesh_dir.glob(f"**/{name}*.glb"))
        if candidates:
            # 取**最新**的 mesh (按 mtime), 不然 SAM3D 重跑后老 mesh 还会被用
            found[name] = max(candidates, key=lambda p: p.stat().st_mtime)
            continue
        candidates = list(VISUALIZATION_DIR.glob(f"{name}*/**/result.glb"))
        if candidates:
            found[name] = max(candidates, key=lambda p: p.stat().st_mtime)
            continue
        print(f"  [WARN] No .glb found for '{name}' in {mesh_dir} (or visualization/)")
    return found


def load_glb_as_pytorch3d(path: Path, device, target_triangles: int = 5000):
    """加载 MV-SAM3D 的 result.glb 并转换到 PyTorch3D 世界系。

    trimesh 默认 y-up (X right, Z out)；PyTorch3D 是 y-up (X **left**, Z **inwards**)。
    跟 MV-SAM3D 自己 layout_post_optimization_utils.py:113-119 的 get_mesh 一致:
        verts @ [[1,0,0],[0,0,-1],[0,1,0]].T

    target_triangles: mesh 简化目标三角形数 (silhouette pose opt 不需要 1M faces)。
        MV-SAM3D 自己也是简化到 5000 (load_and_simplify_mesh)。 None = 不简化。
    """
    import torch
    import numpy as np
    import trimesh

    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes in {path}")
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene

    raw_faces = len(combined.faces)
    # mesh 简化 (open3d quadric decimation，跟 MV-SAM3D 一样)
    if target_triangles is not None and raw_faces > target_triangles:
        try:
            import open3d as o3d
            m_o3d = o3d.geometry.TriangleMesh()
            m_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(combined.vertices))
            m_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(combined.faces))
            m_o3d.remove_duplicated_vertices()
            m_o3d.remove_degenerate_triangles()
            m_o3d.remove_duplicated_triangles()
            m_o3d.remove_non_manifold_edges()
            if len(m_o3d.triangles) > target_triangles:
                m_o3d = m_o3d.simplify_quadric_decimation(target_triangles)
            verts_np = np.asarray(m_o3d.vertices)
            faces_np = np.asarray(m_o3d.triangles)
            print(f"     mesh simplify: {raw_faces} → {len(faces_np)} faces")
        except Exception as e:
            print(f"     [warn] simplify 失败 ({e})，用原 mesh {raw_faces} faces")
            verts_np = np.asarray(combined.vertices)
            faces_np = np.asarray(combined.faces)
    else:
        verts_np = np.asarray(combined.vertices)
        faces_np = np.asarray(combined.faces)

    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int64, device=device)

    # z-up → y-up. 跟 MV-SAM3D layout_post_optimization_utils.get_mesh:113-119 完全一致:
    #   mesh @ [[1,0,0],[0,0,-1],[0,1,0]].T   (注释: "from z-up to y-up")
    # SAM3D canonical mesh 是 z-up, PyTorch3D world 是 y-up (x-left, z-inwards),
    # 不做这一步等价于把 mesh 的 "up" (+Z_mesh) 当成 PT3D 的 "into screen" (+Z_world),
    # 物体就会"躺着"。params.npz["rotation"] 是在 y-up 系下定义的, 所以
    # init_R = R_sam 直接用即可, 不需要 .T 或相似变换。
    M_PT3D = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=device,
    )
    verts = verts @ M_PT3D.T
    vc = None
    return verts, faces, vc


def fit_dominant_plane_ransac(
    pointmap_np,
    n_iters: int = 400,
    dist_thresh: float = 0.02,
    rng_seed: int = 0,
):
    """RANSAC 拟合 scene_pointmap 中的最大平面 (一般就是桌面).

    pointmap_np: (H, W, 3) numpy, PyTorch3D 系下的 (X_left, Y_up, Z_forward).
    返回 (normal[3], inlier_ratio) 或 None.
    """
    import numpy as np
    H, W, _ = pointmap_np.shape
    pts = pointmap_np.reshape(-1, 3)
    valid = np.abs(pts[:, 2]) > 1e-3
    pts = pts[valid]
    if len(pts) < 500:
        return None

    rng = np.random.default_rng(rng_seed)
    best_inliers = 0
    best_normal = None
    for _ in range(n_iters):
        idx = rng.integers(0, len(pts), size=3)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -float(n @ p1)
        dist = np.abs(pts @ n + d)
        n_inl = int((dist < dist_thresh).sum())
        if n_inl > best_inliers:
            best_inliers = n_inl
            best_normal = n
            best_d = d

    if best_normal is None:
        return None
    return best_normal, best_inliers / len(pts)


def build_scene_R_from_gravity(gravity_dir, device):
    """从 scene-frame 下的重力方向构造 R_scene (gravity-aligned world → scene-camera frame).

    PyTorch3D row-vector 约定: X_scene = X_world @ R.
    R 的行 = world 基向量在 scene 系下的表达:
        R[0,:] = world +X 在 scene 系
        R[1,:] = world +Y 在 scene 系 = anti-gravity
        R[2,:] = world +Z 在 scene 系
    """
    import torch
    g = gravity_dir / (gravity_dir.norm() + 1e-8)
    anti_g = -g  # world +Y in scene frame

    # 拿 scene +Z 作为 forward 参考, 投到水平面上得到 world +Z 在 scene 系的表达
    z_ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    world_z = z_ref - (z_ref @ anti_g) * anti_g
    nz = world_z.norm()
    if nz < 0.1:
        # +Z 几乎和重力同向, 改用 +X 作 fallback
        x_ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        world_z = x_ref - (x_ref @ anti_g) * anti_g
        nz = world_z.norm()
    world_z = world_z / (nz + 1e-8)

    # world +X = anti_g × world +Z (右手系, y-up)
    world_x = torch.linalg.cross(anti_g, world_z)
    world_x = world_x / (world_x.norm() + 1e-8)

    R = torch.stack([world_x, anti_g, world_z], dim=0)  # (3, 3) rows
    return R


def yaw_rotation_matrix(theta_rad: float, device):
    """绕 y-up 世界的 +Y 轴转 theta. PyTorch3D row-vector 约定."""
    import math
    import torch
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return torch.tensor(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        device=device, dtype=torch.float32,
    )


def compute_align_long_axis_rotation(verts, device):
    """PCA 找 mesh 最长主轴, 构造 R 把这个轴打到 world +Y.

    PyTorch3D row-vector 约定: long_axis @ R = (0, 1, 0).
    返回 R (3x3) 使得 mesh canonical 顶点 v 经过 v @ R 之后, 长轴方向沿 +Y.
    """
    import torch
    v = verts - verts.mean(dim=0, keepdim=True)
    cov = (v.T @ v) / max(len(v) - 1, 1)
    # eigh 升序返回 → 最后一个特征向量是最大方差方向
    eigvals, eigvecs = torch.linalg.eigh(cov)
    long_axis = eigvecs[:, -1]  # (3,) unit vector

    # PCA 不给方向, 用 skewness 决定正负: 一般物体 base 重 (mug_tree 圆盘底,
    # red_mug 杯底) → "重的那一端朝下", 即 long_axis 该指向轻的那端 (skew > 0).
    proj = v @ long_axis  # (N,)
    skew = float((proj ** 3).mean())
    if skew < 0:
        long_axis = -long_axis

    target = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=long_axis.dtype)
    d = float((long_axis * target).sum())
    # 已经 ≈ 平行 → 返回 I
    if d > 0.9999:
        return torch.eye(3, device=device, dtype=long_axis.dtype)
    # ≈ 反平行 → 绕任意垂直轴翻 180° (用 +X 轴)
    if d < -0.9999:
        return torch.tensor(
            [[1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, -1.0]],
            device=device, dtype=long_axis.dtype,
        )

    # 一般情况: 用 Rodrigues 构造 column-vector 旋转, 然后取 .T 给 row-vector 用.
    # column-vec: R_col @ long_axis = target  (axis = long × target, angle = acos(d))
    axis = torch.linalg.cross(long_axis, target)
    axis = axis / (axis.norm() + 1e-8)
    import math
    angle = math.acos(max(-1.0, min(1.0, d)))
    K = torch.tensor(
        [[0.0, -axis[2].item(), axis[1].item()],
         [axis[2].item(), 0.0, -axis[0].item()],
         [-axis[1].item(), axis[0].item(), 0.0]],
        device=device, dtype=long_axis.dtype,
    )
    R_col = (torch.eye(3, device=device, dtype=long_axis.dtype)
             + math.sin(angle) * K
             + (1.0 - math.cos(angle)) * (K @ K))
    R_row = R_col.T.contiguous()  # row-vector 约定
    return R_row


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_file = WORK_DIR / "scene_masks.pt"
    if not mask_file.exists():
        print(f"[ERROR] {mask_file} 不存在。请先在 sam3 env 跑 `--step 3a`。")
        sys.exit(2)
    bundle = torch.load(mask_file, map_location="cpu", weights_only=False)
    print(f"[step3c] loaded scene_masks.pt: {bundle['masks'].shape[0]} mask(s)")
    if bundle["image_path"] != str(scene_image):
        print(f"[step3c] [warn] scene image 路径与 step3a 产物不一致：")
        print(f"   step3a: {bundle['image_path']}")
        print(f"   step3c: {scene_image}")

    pointmap_file = WORK_DIR / "scene_pointmap.npy"
    metric_pointmap = None
    if pointmap_file.exists():
        metric_pointmap = torch.from_numpy(np.load(pointmap_file)).float()
        print(f"[step3c] loaded scene_pointmap.npy {tuple(metric_pointmap.shape)}")
    else:
        print(f"[step3c] {pointmap_file} 不存在，回退 DepthAnything (Z 仍为 2.0 估计)")

    # ── 从 pointmap 估场景重力, 构造 R_scene (gravity-world → scene-camera) ──
    # 假设场景里有面积最大的水平面 (桌面 / 棋盘格), 法向 = anti-gravity.
    R_scene = None
    if metric_pointmap is not None:
        pm_np = metric_pointmap.cpu().numpy()
        plane = fit_dominant_plane_ransac(pm_np, n_iters=400, dist_thresh=0.02)
        if plane is not None:
            normal, ratio = plane
            # 选取与 +Y 正向夹角更小的那一侧 (PyTorch3D world +Y 是 up = anti-gravity)
            if normal[1] < 0:
                normal = -normal
            gravity_dir = torch.tensor(-normal, dtype=torch.float32, device=device)
            R_scene = build_scene_R_from_gravity(gravity_dir, device)
            print(f"[step3c] gravity-fit OK: plane normal={normal.round(3).tolist()}  "
                  f"inlier_ratio={ratio:.2f}")
        else:
            print("[step3c] gravity-fit 失败 (点太少), R_scene = I")
    if R_scene is None:
        R_scene = torch.eye(3, device=device)

    # MoGe 估出来的内参（step3b 保存）。没有就走默认 60° FoV。
    K_file = WORK_DIR / "scene_intrinsics.npy"
    K = None
    if K_file.exists():
        K = torch.from_numpy(np.load(K_file)).float()
        print(f"[step3c] loaded scene_intrinsics.npy fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    else:
        print(f"[step3c] {K_file} 不存在；pipeline 会用默认 60° FoV")

    inpaint_file = WORK_DIR / "inpainted.png"
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
    for label, mask in zip(labels, masks):
        slug = sanitize_filename(label).lower()
        matched_name = None
        for name, _ in mesh_paths.items():
            name_l = name.lower()
            prompt_slug = name_to_prompt_slug.get(name, "")
            if (
                slug in name_l or name_l in slug
                or (prompt_slug and (slug in prompt_slug or prompt_slug in slug))
            ):
                matched_name = name
                break
        if matched_name is None:
            print(f"  [warn] label='{label}' (slug='{slug}') 没找到匹配的 mesh，跳过")
            continue

        glb_path = mesh_paths[matched_name]
        print(f"  '{label}' -> mesh {matched_name} <- {glb_path}")
        verts, faces, vc = load_glb_as_pytorch3d(glb_path, device)
        entry = {"verts": verts, "faces": faces, "name": matched_name}
        if vc is not None:
            entry["vertex_colors"] = vc

        # 从 params.npz 取 rotation 当 init_R prior（candidate 0）。
        # 经 debug_sam3d_prior.py 验证 (2026-05-17):
        #   - rotation:    ✅ 可用 → 给 init_R, red mug 不优化就 IoU 0.63
        #   - translation: ❌ 不可用（SAM3D 跑的是 crop 不是 scene，坐标不对得上）
        #   - scale:       ❌ 不可用（SAM3D 是 SSI-normalized scale，
        #                     乘 pointmap_scale 后也只能到 ~1.5m，实际 mug 0.17m，差 ~9×）
        # → 让 mask + depth 自动估 init_t 和 init_scale。
        #
        # convention: 直接用 R_sam (不转置).
        # MV-SAM3D 自己 (layout_post_optimization_utils.py:113-119) 的流程是
        #   y-up_verts = z-up_verts @ M_PT3D.T   (mesh 转 y-up)
        #   world_verts = tfm_ori.transform_points(y-up_verts)   (其中 rotation=R_sam)
        # 也就是说 R_sam 是在 y-up 系下定义的。我们 load_glb_as_pytorch3d 已经把 mesh
        # 做了同样的 z-up→y-up, 所以这里 R_sam 直接当 PyTorch3D camera R 即可,
        # 不再做 .T 也不做相似变换。
        # (历史: 2026-05-17 误判 "mesh 已经是 y-up", 把 M_PT3D 撤掉同时 R 不转置,
        # 结果两个错误抵消一半 → 红 mug 蒙到 IoU=0.69, mug_tree 0.08 暴露问题。)
        params_npz = glb_path.parent / "params.npz"
        R_sam = None
        if params_npz.exists():
            try:
                from pytorch3d.transforms import quaternion_to_matrix
                p = np.load(params_npz)
                quat = np.asarray(p["rotation"]).flatten()  # (4,) wxyz
                R_sam = quaternion_to_matrix(
                    torch.from_numpy(quat).float().to(device)
                ).contiguous()  # (3,3)
                entry["init_R"] = R_sam
                s_legacy = float(np.asarray(p["scale"]).mean())
                print(f"     init_R from params.npz quat=[{quat[0]:.3f}, {quat[1]:.3f}, "
                      f"{quat[2]:.3f}, {quat[3]:.3f}]  (sam_scale={s_legacy:.3f} 仅打印不使用)")
            except Exception as e:
                print(f"     [warn] params.npz 读不出来: {e}")

        # ── 构造 init pose ──────────────────────────────────────────────
        # 策略: ICP 给的 (t, s) 比 mask centroid + extent ratio 更准 (3D 校准),
        # 优先用. R 那边 ICP 不一定全局最优 (scene mesh 单视角质量限制),
        # 所以 R candidate 集合 = ICP-yaw + gravity-yaw, 让 Stage 1 自己选 IoU 最高的.
        init_pose_file = WORK_DIR / f"init_pose_{matched_name}.npz"
        candidates_list = []
        if init_pose_file.exists():
            ip = np.load(init_pose_file)
            R_icp = torch.from_numpy(ip["R"]).float().to(device)
            t_icp = torch.from_numpy(ip["t"]).float().to(device)
            s_icp = float(ip["scale"])
            # 用 ICP 的 t, s 当 init (它们来自 3D pointmap, 比 2D mask 估算可靠)
            entry["init_R"] = R_icp
            entry["init_t"] = t_icp
            entry["init_scale"] = s_icp
            # ICP-based yaw refinement: 8 个 yaw (45° 步长) + ICP 原 R, 共 9 个
            for d_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                theta = 2.0 * 3.141592653589793 * d_deg / 360.0
                R_yaw = yaw_rotation_matrix(theta, device)
                candidates_list.append(R_icp @ R_yaw)
            icp_info = (f"ICP-yaw (9, init from {init_pose_file.name}: "
                        f"s={s_icp:.4f}, fit={float(ip['icp_fitness']):.3f}, "
                        f"rmse={float(ip['icp_rmse']):.4f}m)")
        else:
            icp_info = "no ICP file"

        # Gravity-yaw candidates: R_sam @ Ry(k·45°) @ R_scene (旧路径)
        if R_sam is not None:
            for k in range(8):
                theta = 2.0 * 3.141592653589793 * k / 8.0
                R_yaw = yaw_rotation_matrix(theta, device)
                candidates_list.append(R_sam @ R_yaw @ R_scene)
            grav_info = "+ R_sam@Ry@R_scene (8)"
        else:
            for k in range(8):
                theta = 2.0 * 3.141592653589793 * k / 8.0
                R_yaw = yaw_rotation_matrix(theta, device)
                candidates_list.append(R_yaw @ R_scene)
            grav_info = "+ Ry@R_scene (8)"

        entry["init_R_candidates"] = candidates_list
        print(f"     {len(candidates_list)} init_R candidates: {icp_info} {grav_info}")

        mesh_data_list.append(entry)
        precomputed_masks.append({"mask": mask, "label": label})

    if not mesh_data_list:
        print("[step3c] 没有任何物体可优化，退出。")
        sys.exit(1)

    config = PipelineConfig()
    config.output_dir = str(output_dir)
    config.segmentation.text_prompt = scene_prompt
    config.segmentation.expand_mask_iters = 0
    config.segmentation.expand_kernel = 15
    config.optimizer.stages = 3
    config.optimizer.iters_per_stage = 300
    config.save_intermediate = True

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
# ─────────────────────────────────────────────────────────────────────

def step4_render(
    scene_image: str,
    results: Dict,
    mesh_paths: Dict[str, Path],
    output_dir: Path,
):
    banner("4", "sam3d-objects", "Rendering & comparison")
    assert_env("sam3d-objects", ["torch", "pytorch3d", "trimesh"])

    import numpy as np
    import torch
    from PIL import Image
    from pytorch3d.renderer import (
        BlendParams, MeshRasterizer, MeshRenderer, PerspectiveCameras,
        PointLights, RasterizationSettings, SoftPhongShader, TexturesVertex,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig = Image.open(scene_image).convert("RGB")
    W, H = orig.size
    orig_tensor = torch.from_numpy(np.array(orig)).float().to(device) / 255.0

    cam = results.get("camera", {})
    fx = cam.get("fx", W)
    fy = cam.get("fy", H)
    cx = cam.get("cx", W / 2)
    cy = cam.get("cy", H / 2)

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

        # PyTorch3D row-vector 约定 (跟 step3c 优化器一致):
        # cameras.transform_points(X_world) = X_world @ R + T
        # 这里 camera R = I, T = 0, 所以我们直接把 mesh 摆到 world 等价于:
        # X_world = X_canonical @ R_opt + t_opt
        # 之前写的是 R.T (= (R @ V.T).T = V @ R.T), 跟 step3c 不一致, 渲出来位姿差一个 transpose,
        # 导致 step4 comparison.png 跟 per_object eval 不一样。
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

    R_cam = torch.eye(3, device=device)
    T_cam = torch.zeros(3, device=device)
    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        R=R_cam[None], T=T_cam[None],
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


# ─────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real2Sim Full Workflow (双环境版)")
    parser.add_argument("--step", required=True,
                        choices=["1", "2", "3a", "3b", "3c", "4"],
                        help="一次只跑一步 (因为不同步骤需要不同 conda env)")
    parser.add_argument("--scene", type=str, default=str(SCENE_IMAGE),
                        help="场景图路径")
    parser.add_argument("--scene-prompt", type=str, default=SCENE_PROMPT,
                        help="场景中物体文字 prompt")
    parser.add_argument("--mesh-dir", type=str, default=str(VISUALIZATION_DIR),
                        help="搜索 .glb 的根目录 (step 3c/4)")
    parser.add_argument("--meshes", type=str, nargs="*", default=None,
                        help="直接指定 mesh: name=path 形式 (step 3c/4)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    parser.add_argument("--K", type=float, nargs=4, default=None,
                        metavar=("FX", "FY", "CX", "CY"),
                        help="(step 3b) 已知相机内参 (像素单位)。"
                             "提供后由 fx+宽推 fov_x 喂给 MoGe 当先验，"
                             "并把这个 K 存为 scene_intrinsics.npy。"
                             "不传则让 MoGe 自己估。")
    args = parser.parse_args()

    setup_logging(args.step)

    step = args.step
    scene_image = args.scene
    scene_prompt = args.scene_prompt
    output_dir = Path(args.output)
    mesh_dir = Path(args.mesh_dir)

    print(f"[main] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[main] step         = {step}")
    print(f"[main] scene image  = {scene_image}")
    print(f"[main] output dir   = {output_dir}")

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

    if args.meshes:
        glb_paths = {}
        for item in args.meshes:
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                p = Path(item)
                name = p.parent.name or p.stem
                path = item
            glb_paths[name] = Path(path)
    else:
        object_names = [o["name"] for o in OBJECTS]
        glb_paths = find_glb_files(mesh_dir, object_names)

    if not glb_paths:
        print("\n[ERROR] 没有找到任何 .glb。请用 --meshes name=path 指定，")
        print("        或确认 step 2 已跑完，--mesh-dir 指向 visualization/。")
        sys.exit(1)
    print(f"[main] mesh paths:")
    for n, p in glb_paths.items():
        print(f"  {n}: {p}")

    if step == "3c":
        step3c_real2sim(scene_image, scene_prompt, glb_paths, output_dir)
        return

    if step == "4":
        results_file = output_dir / "pipeline_results.json"
        if not results_file.exists():
            print(f"[ERROR] {results_file} 不存在，请先跑 --step 3c。")
            sys.exit(2)
        import torch
        with open(results_file) as f:
            results = json.load(f)
        for obj in results.get("objects", []):
            obj["R"] = torch.tensor(obj["R"])
            obj["t"] = torch.tensor(obj["t"])
        step4_render(scene_image, results, glb_paths, output_dir / "comparison")
        return


if __name__ == "__main__":
    main()
