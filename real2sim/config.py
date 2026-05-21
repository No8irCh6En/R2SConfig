from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CameraConfig:
    """Camera configuration. If intrinsics are unknown, fov_deg or focal_from_hw is used."""

    width: int = 1920
    height: int = 1080
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    fov_deg: float = 60.0  # used only when fx/fy not provided


@dataclass
class SegmentationConfig:
    """Grounding-DINO + SAM segmentation settings."""

    text_prompt: str = "object"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    expand_mask_iters: int = 3
    expand_kernel: int = 15


@dataclass
class InpaintingConfig:
    """LaMa inpainting settings."""

    model_path: str = "advimman/lama"
    device: str = "cuda"
    pad_to: int = 8
    refine_mask_iters: int = 0


@dataclass
class DepthConfig:
    """DepthAnything settings."""

    model_size: str = "large"  # small | base | large
    device: str = "cuda"


@dataclass
class OptimizerConfig:
    """Pose/scale/camera optimization settings."""

    stages: int = 3
    iters_per_stage: int = 200
    lr_pose: float = 0.01
    lr_scale: float = 0.005
    lr_camera: float = 0.001

    # Loss weights per stage: [mask, depth, rgb]
    loss_weights: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0],  # stage 1: silhouette only
        [0.5, 0.5, 0.0],  # stage 2: silhouette + depth
        [0.3, 0.3, 0.4],  # stage 3: silhouette + depth + rgb
    ])
    ssim_weight: float = 0.2
    scale_regularizer: float = 0.01
    depth_scale_regularizer: float = 0.1

    # Stage 1 (depth-anchored R/t) 实际控制参数 ─ PoseOptimizer.optimize 真正用的
    stage1_iters_coarse: int = 80     # 128x128 那级跑几 iter
    stage1_iters_fine:   int = 60     # 384x384 那级跑几 iter
    stage1_lr_r6d:       float = 5e-3 # 6D 旋转的 Adam lr
    stage1_lr_t:         float = 1e-2 # translation 的 Adam lr
    stage1_depth_beta:   float = 0.05 # Huber depth loss 的过渡阈值 (meters)
    stage1_iou_weight:   float = 1.0  # soft-IoU loss 的权重 (depth_loss 权重 = 1.0)
    # 历史: 旧版本用 L1 mask + 0.3 weight, 但 L1 在 red/green 区域信号太弱,
    # loss 低 IoU 低同时出现 (2026-05-18 实测). 改成 soft-IoU 让 red/green 直接进 loss.

    # SoftSilhouette sigma (silhouette 边界平滑度). 大 → 远距吸引力强但边界糊,
    # 小 → 边界锐利但远处无梯度. coarse-to-fine: 先大后小.
    stage1_sigma_coarse: float = 1e-2  # L0 用 (128x128, 粗搜)
    stage1_sigma_fine:   float = 1e-4  # L1 用 (384x384, refine)
    stage1_faces_per_px: int   = 50    # 必须 >> 1, 不然 SoftSilhouette 退化成 hard

    # 3D Chamfer loss (P_tgt → V_pred 单向). pointmap (mask 内) 给出物体可见表面的
    # 真实 3D 位置, mesh-after-pose 跟它对齐. 解决 silhouette 对绕长轴 yaw 不敏感
    # 的问题 (例如 mug_tree 的挂钩在 3D 凸出方向上很有信息, 2D 投影分不清).
    stage1_chamfer_weight:    float = 0.5    # chamfer loss 权重 (相对 IoU=1.0)
    stage1_chamfer_max_pts:   int   = 1500   # 从 mask 内 pointmap 采几个点
    stage1_chamfer_max_verts: int   = 3000   # 从 mesh verts 采几个 (对距离矩阵的 M)
    stage1_chamfer_in_coarse: bool  = False  # L0 (128x128) 也加 chamfer? 默认只 L1
    # candidate 综合排序: score = IoU - w · chamfer_norm
    # chamfer_norm = chamfer_dist / (mesh_extent * s). 单位无量纲.
    selection_chamfer_weight: float = 0.5    # 排序时 chamfer 相对 IoU 的权重


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output_dir: str = "outputs/real2sim"
    # Optional: where to drop per-run QA visualizations (per_object/<name>_eval.png).
    # If None, falls back to output_dir. In the real2sim driver this is wired to
    # paths.run_dir so per_object/ lives next to comparison/ and gsrl_config.json.
    viz_output_dir: Optional[str] = None
    save_intermediate: bool = True
