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
    # 若 True, optimize() 跳过 Stage 2 (analytical scale) + Stage 1 refine,
    # s_final 永远等于 init_scale. 用来 ablation / 已知真实 scale 的固定测试.
    freeze_scale: bool = False

    # 若 True, R 退化为 1DOF (绕 R_scene 给定的 gravity 轴 yaw), roll/pitch 锁死 0.
    # 适合"杯子平放在桌面"这种 upright 场景: 缩小搜索空间, 320x240 低分辨率 silhouette
    # 容易让 R 收敛到错误的倾斜局部极值. 需要 optimizer.optimize 时同时传 R_scene
    # (gravity-world → scene-camera rotation, step3c 在 pipeline.py 算的);
    # 不传则退回相机 +Y 当 up (相机倾斜时不准确).
    yaw_only: bool = False

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

    # Asymmetric silhouette loss (Tversky generalization of IoU).
    #   IoU      = TP / (TP + FP + FN)            (= Tversky with α=β=1)
    #   Tversky  = TP / (TP + α·FP + β·FN)
    # where, with rendered alpha (green) vs target mask (red):
    #   TP = green∩red, FP = green∖red (render spills OUTSIDE mask),
    #   FN = red∖green (mask NOT covered by render).
    # α>β ⇒ "rendered silhouette must stay INSIDE the mask, but it's OK if the
    # mask isn't fully covered" — pins the handle DIRECTION (a wrong-way handle
    # pokes green outside red → heavily penalized) while tolerating small under-
    # coverage. The TP numerator means shrinking kills the reward (loss→1), so
    # this does NOT collapse scale (unlike a pure precision/coverage penalty).
    # Defaults 1.0/1.0 reproduce plain IoU exactly (opt-in, backward-compatible).
    stage1_iou_fp_weight: float = 1.0   # α: penalty on green-outside-red (precision)
    stage1_iou_fn_weight: float = 1.0   # β: penalty on red-not-covered  (recall)

    # SoftSilhouette sigma (silhouette 边界平滑度). 大 → 远距吸引力强但边界糊,
    # 小 → 边界锐利但远处无梯度. coarse-to-fine: 先大后小.
    stage1_sigma_coarse: float = 1e-2  # L0 用 (128x128, 粗搜)
    stage1_sigma_fine:   float = 1e-4  # L1 用 (384x384, refine)
    stage1_faces_per_px: int   = 50    # 必须 >> 1, 不然 SoftSilhouette 退化成 hard

    # 若 True (+ yaw_only + floor_lock), Stage 1 里 s 也是可学参数, 用 IoU loss 直接优化.
    # tz 仍由 floor_lock 自动算 (= plane_z - s·z_min). 开了就跳过 Stage 2 area_ratio.
    # 比 area_ratio 鲁棒: IoU 形状敏感, mesh 撑出 mask 边界会被惩罚.
    learn_scale:        bool   = False
    stage1_lr_scale:    float  = 5e-3  # s_log 的 Adam lr (exp 参数化, 量纲 ≈ 0.1-0.5)

    # Level 3: Distance-Transform silhouette loss.
    # IoU loss has a degenerate min for asymmetric objects (mug handle, kettle
    # spout): when mesh feature shape ≠ real feature shape, optimizer can pick a
    # yaw that HIDES the mesh feature behind the body (mesh handle occluded →
    # smaller union → higher IoU), even though the visual pose is wrong.
    # DT loss = Σ_p (α·DT_out + (1-α)·DT_in) doesn't have this min: target's
    # feature pixels still demand a rendered match (high DT_in penalty if α=0
    # there), so "hidden" loses. Acts as soft Chamfer between rendered/target
    # contours without the cost of explicit contour extraction.
    use_dt_loss:        bool   = False
    stage1_dt_weight:   float  = 2.0   # DT loss is normalized by max(H,W); 2.0 ≈ IoU scale
    # Asymmetry within the DT loss (same idea as Tversky α/β, but distance-weighted):
    #   dt = w_out·loss_out + w_in·loss_in
    #   loss_out = green-outside-red, distance-from-mask weighted (precision)
    #   loss_in  = red-not-covered,  distance-from-edge weighted  (recall)
    # w_out>w_in ⇒ handle stays inside mask. NOTE: DT has no TP reward, so a high
    # w_out with learn_scale CAN shrink scale — keep scale on area-ratio
    # (--no-learn-scale) when using DT asymmetry. Defaults 1.0/1.0 = symmetric.
    stage1_dt_w_out:    float  = 1.0   # precision (green outside mask)
    stage1_dt_w_in:     float  = 1.0   # recall (mask not covered)
    # When use_dt_loss=True, also add this weight × (1 - soft_iou) on top of DT.
    # 0 (default) = pure DT (DT alone preserves scale ≈ contour match).
    # >0 blends in IoU for higher coverage at the cost of scale being pulled
    # back toward the IoU-preferred (often-smaller) value. Try 0.3-0.5 if you
    # want a moderate coverage boost without losing the DT scale.
    stage1_dt_iou_blend: float  = 0.0

    # Stage 1 (tx, ty) 多起点 grid search. 1 = 关 (老行为, 8 yaw × 1 init_t).
    # >1 = 在 init_t 周围 NxN grid 撒点, 每个点都跑一遍 (yaw × N×N candidates).
    # 用来逃出 silhouette loss 的局部极值 — 非凸物体 (tree 这种) 容易卡 wrong basin.
    # 总 candidates = 8 yaw × N² grid, 时间 × N². 推荐 N=5 (25 grid × 8 yaw = 200 cand).
    stage1_txy_grid:        int   = 1
    # grid 半径 (米). 在 init_t.xy ± stage1_txy_grid_radius 之间均匀撒 N 个点.
    stage1_txy_grid_radius: float = 0.5


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
