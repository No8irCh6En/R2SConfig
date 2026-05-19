# R2SConfig — Real2Sim Pose Alignment Pipeline

把场景照片 + 物体多视角扫描照片 → 输出每个物体在场景相机系下的 `(R, t, s)`,
以及对齐到场景的可视化 mesh。用于后续 Genesis 仿真 / GS 渲染的初始化配置。

---

## Pipeline 概览

```
       ┌──────────────────────── sam3 env ─────────────────────────┐ ┌────────── sam3d-objects env ──────────┐
                                                                    
 物体多视角图  ─Step 1→  per-view mask                                                                       
                              │                                                                              
                              └─Step 2─→ SAM3D multi-view mesh (canonical)                                  
                                                                                                              
 scene.jpg ──Step 3a→ scene masks (per-object)                                                                 
                              │                                                                              
                              └─Step 3b─→ scene metric pointmap (MoGe)                                       
                                            │                                                                
                                            ├─Step 3d→ SAM3D single-view mesh + pose in scene frame         
                                            │              │                                                
                                            │              └─Step 3e→ ICP (multi-view mesh → scene mesh)    
                                            │                          → init_pose_<obj>.npz               
                                            │                                                                
                                            └─Step 3c→ Real2Sim pose optimization                            
                                                       (Stage 1: depth+IoU+chamfer, Stage 2: scale)         
                                                              │                                              
                                                              └─Step 4→ render comparison
```

每步只做一件事, 失败可单独重跑。

| Step | env             | 输入                              | 输出                                              |
|------|-----------------|-----------------------------------|---------------------------------------------------|
| 1    | sam3            | 物体多视角图                       | `MV-SAM3D/visualization/<obj>/masks/*.png`        |
| 2    | sam3d-objects   | 多视角图 + masks                  | `MV-SAM3D/visualization/<obj>/*.glb`              |
| 3a   | sam3            | scene.jpg                         | `workdir/scene_masks.pt`                          |
| 3b   | sam3d-objects   | scene.jpg                         | `workdir/scene_pointmap.npy`, `scene_intrinsics.npy` |
| 3d   | sam3d-objects   | scene.jpg + scene masks + pointmap| `workdir/scene_sam3d/<obj>/{result.glb, params.npz}` |
| 3e   | sam3d-objects   | multi-view mesh + 3d 出的 scene mesh | `workdir/init_pose_<obj>.npz` (ICP-aligned R/t/s) |
| 3c   | sam3d-objects   | mesh + 所有 init + pointmap        | `outputs/full_workflow/scene.json`                |
| 4    | sam3d-objects   | scene.json + meshes               | `outputs/full_workflow/comparison/*.png`          |

---

## 环境

需要两个 conda env (都已经预装好):

- **`sam3`** — Grounding-DINO + SAM2 (做分割)
- **`sam3d-objects`** — PyTorch3D + MV-SAM3D + MoGe (做重建、深度、优化)

GPU 必需。系统驱动只要支持 CUDA ≥ 11.8, env 里 torch 自带的 CUDA runtime 会工作。

```bash
conda env list | grep -E "sam3|sam3d-objects"
```

---

## 素材准备

```
R2SConfig/assets/
├── red_mug/images/                # 物体 A 多视角照片 (3-5 张)
│   ├── 0.jpg
│   └── ...
├── black_mug_tree/images/         # 物体 B 多视角照片 (3-5 张)
│   └── ...
└── scene.jpg                      # 场景图 (同时包含所有物体)
```

要换物体: 改 `full_workflow.py` 顶部的 `OBJECTS` 列表 (name / image_dir / prompt) 和 `SCENE_PROMPT`。

---

## Quick start

```bash
# 一次跑完整条 pipeline
bash fast_start.sh

# 一切顺利的话, 最后看
ls outputs/full_workflow/scene.json outputs/full_workflow/comparison/
```

跑完一次后, 后续只想调 step3c 优化 (最常见的迭代点):

```bash
bash fast_start.sh 3c
```

---

## 单步运行

```bash
bash fast_start.sh 1     # 多视角 mask
bash fast_start.sh 2     # 多视角 SAM3D 重建
bash fast_start.sh 3a    # 场景分割
bash fast_start.sh 3b    # 场景 metric pointmap (MoGe)
bash fast_start.sh 3d    # SAM3D 跑在 scene.jpg 上
bash fast_start.sh 3e    # ICP 把 multi-view mesh 对齐到场景
bash fast_start.sh 3c    # 实际 R/t/s 优化
bash fast_start.sh 4     # 渲染对比图
```

**指定相机 vFoV** (Genesis 用 vertical FoV 度数, 默认会用 MoGe 估的, 估错可手动覆盖):

```bash
# 假设 vfov=64.81°, 在 full_workflow.py 里 step3b 调用时传 --K
python full_workflow.py --step 3b --K 64.81
```

**Stage 1 加大迭代次数** (调试时排查梯度下降是否健康):

```bash
STAGE1_ITER_SCALE=10 bash fast_start.sh 3c    # coarse/fine iter ×10
```

---

## 产物

### `outputs/full_workflow/`

| 文件                       | 说明                                                      |
|----------------------------|-----------------------------------------------------------|
| `scene.json`               | **主输出**: 每个物体的 `rotation` (3×3), `translation` (3), `scale` (float) + 相机 K |
| `<obj>_transform.pt`       | 4×4 transform 矩阵 (`T[:3,:3] = R*s, T[:3,3] = t`)        |
| `comparison/*.png`         | mesh 叠到原图的对比图                                       |
| `per_object/<obj>_eval.png`| 单物体 4-panel 评估图 (mask vs render, IoU, depth consistency) |
| `pipeline_results.json`    | 完整 dump (metrics + history + candidate logs)            |

### `workdir/` (中间产物)

| 文件                                    | 说明                                       |
|----------------------------------------|--------------------------------------------|
| `scene_masks.pt`                       | 场景 per-object mask (3a)                  |
| `scene_pointmap.npy` (H, W, 3)         | metric 3D pointmap, PT3D 系                |
| `scene_intrinsics.npy` (3, 3)          | 相机内参 K                                  |
| `scene_sam3d/<obj>/result.glb`         | SAM3D 单帧重建 mesh                         |
| `scene_sam3d/<obj>/params.npz`         | SAM3D 给的 scene-frame pose                |
| `init_pose_<obj>.npz`                  | 3e 出的 ICP-aligned init (R, t, scale, icp_fitness, icp_rmse) |
| `yaw_sweep_<obj>.png`                  | yaw_sweep.py 出的轴向旋转扫描图              |

---

## 坐标 / 矩阵约定

**所有 R/t 都是 PyTorch3D row-vector 约定**:

```
verts_view = (verts_world * s) @ R + t
```

- 喂给 column-vector 框架 (open3d / OpenCV / 大部分文献): 用 `R.T`
- scale 是各向同性, 直接乘 verts

**相机系**: PyTorch3D camera frame — X 左, Y 上, Z 前 (右手系)。

**单位**: 米 (metric)。MoGe 出的 pointmap 已经是 metric, 不需要再 scale。

**canonical mesh**:
- SAM3D 出的是 z-up; 我们 load 后乘 `M_PT3D.T = [[1,0,0],[0,0,1],[0,-1,0]]` 转 y-up
- mug_tree 转完后 pole 沿 +Z (canonical 长轴, extent 约 (0.49, 0.53, 1.00))
- red_mug 转完后开口沿 +Y (extent 约 (0.63, 1.00, 0.70))

---

## 项目结构

```
R2SConfig/
├── README.md                  ← 你正在看的这个
├── fast_start.sh              ← pipeline 入口
├── full_workflow.py           ← 编排所有 step
├── scripts/                   ← 用于 debug / 评估的辅助脚本
├── real2sim/                  ← Real2Sim 核心模块 (step3c 用)
│   ├── pipeline.py            ← Real2SimPipeline 主类
│   ├── optimizer.py           ← PoseOptimizer (Stage 1: depth+IoU+chamfer, Stage 2: scale)
│   ├── camera.py              ← rotation_6d ↔ matrix, intrinsics utils
│   ├── config.py              ← OptimizerConfig 等 dataclass
│   └── utils.py
├── assets/                    ← 用户素材 (不进 git)
├── workdir/                   ← 中间产物 (不进 git)
├── outputs/                   ← 最终产物 (不进 git)
├── logs/                      ← 每步日志 (不进 git)
├── sam3/                      ← Grounding-DINO + SAM2 仓库
└── MV-SAM3D/                  ← SAM3D-objects 仓库
```

---

## License / Credits

- SAM3D-objects: see `MV-SAM3D/`
- Grounding-DINO + SAM2: see `sam3/`
- MoGe (monocular geometry): bundled in sam3d-objects env
- PyTorch3D: differentiable rendering backend
