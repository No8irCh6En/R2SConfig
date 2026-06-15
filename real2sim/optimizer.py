"""
Differentiable pose/scale/camera refinement via PyTorch3D.

Uses multi-start rotation sampling + resolution pyramid for robust
scale, translation, and camera refinement.

## Capabilities & Limitations

**Works well:**
- Scale refinement: ~0.2% error with good initialization
- Translation refinement: converges from mask-based heuristics
- Multi-view consistency: can use depth/mask from multiple views

**Known limitation — rotation recovery:**
Silhouette-only rotation optimization is fundamentally ill-posed for
most common object shapes (symmetry, self-occlusion, depth ambiguity).
This optimizer uses multi-start sampling (every 45° around Y by default)
to find a reasonable rotation, but for precise rotation you should:
  - Provide a good initial rotation (e.g. known object orientation)
  - Use texture/RGB loss if vertex colors are available
  - Use multi-view constraints if multiple images are available
  - Consider keypoint-based methods for rotation initialization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Dict, List
from tqdm import tqdm
import math

from .camera import rotation_6d_to_matrix, rotation_matrix_to_6d, intrinsics_from_hfov, build_camera_matrix
from .config import OptimizerConfig


def _yaw_matrix_world(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable yaw rotation 3x3 绕 +Z (PT3D row-vec convention).

    M_LOAD 之后 mesh canonical 的 up 是 +Z (.glb 是 gltf y-up, M_LOAD 把 y-up 转 z-up).
    所以 yaw 应该绕 +Z 而不是 +Y. 这是 SAM3D / Tripo .glb mesh 的实际 up 方向.

    Row-vec yaw_z(θ):  +X → (cos, sin, 0), +Y → (-sin, cos, 0), +Z → (0, 0, 1) preserved.
    """
    c = torch.cos(theta).reshape(())
    s = torch.sin(theta).reshape(())
    zero = torch.zeros((), device=theta.device, dtype=theta.dtype)
    one = torch.ones((), device=theta.device, dtype=theta.dtype)
    row0 = torch.stack([c, s, zero])
    row1 = torch.stack([-s, c, zero])
    row2 = torch.stack([zero, zero, one])
    return torch.stack([row0, row1, row2])




class ScaleInvariantDepthLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)
        mask = mask & (pred > self.eps) & (target > self.eps)
        if mask.sum() < 10:
            return torch.tensor(0.0, device=pred.device)
        p = torch.log(pred[mask] + self.eps)
        t = torch.log(target[mask] + self.eps)
        diff = p - t
        return (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)


def _make_renderers(size: tuple[int, int], sigma: float, device: torch.device,
                    faces_per_pixel: int = 50):
    """Build silhouette/phong/raster renderers with **differentiable** silhouette.

    关键: 必须 blur_radius > 0 且 faces_per_pixel > 1 才能让 SoftSilhouetteShader
    真正给远离 mesh 的像素非零 alpha → 才能反传梯度。
        blur_radius = log(1/threshold - 1) * sigma,  threshold = 1e-4
    这是 SoftRas / PyTorch3D 推荐公式 (见 BlendParams docstring).

    sigma 越大 → halo 越宽 → 长距吸引力越强 (但边界模糊). 多 candidate 粗搜阶段用大,
    refine 用小.
    """
    from pytorch3d.renderer import (
        MeshRasterizer, RasterizationSettings,
        SoftSilhouetteShader, SoftPhongShader, MeshRenderer, BlendParams,
    )
    import numpy as _np
    H, W = size
    blend = BlendParams(sigma=sigma, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    # 经典 SoftRas 公式: blur_radius 应该跟 sigma 同量级
    blur_radius = float(_np.log(1.0 / 1e-4 - 1.0) * sigma)
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=0,
        max_faces_per_bin=200_000,
    )
    sil = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend),
    )
    phong = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, blend_params=blend),
    )
    # rasterizer 用单独 settings: depth zbuf 用 faces_per_pixel=1 (最近面)
    rast_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        bin_size=0, max_faces_per_bin=200_000,
    )
    rast = MeshRasterizer(raster_settings=rast_settings)
    return sil, phong, rast

class PoseOptimizer:
    """两阶段 pose 估计 (GS-Playground 论文 III.D 风格)，**不做** 12-candidate 多起点.

    Stage 1 — depth-anchored (R, t) registration
        固定 scale, 用 Huber loss 对齐 rendered_depth 跟 metric pointmap[..., 2].
        理由: 深度信号一阶不耦合 scale, 把 (R, t) 唯一地 anchor 到真实 3D 位置。
        没 overlap 时 fallback 到 mask L1 把 mesh 拽进 mask, 一旦 overlap 就切到 depth.

    Stage 2 — analytical scale matching
        固定 (R, t), 用面积比 `s_new = s * sqrt(area_target / area_pred)` 求 scale。
        透视下不是严格 closed-form, 但 2-3 次迭代就收敛.

    跟旧多 candidate + mask L1 方案的区别:
      旧: 12 candidate × 50 iter + 3-level × 100 iter = 900 iter, IoU 局部极值满天飞
      新: 2-level × ~80 iter + 5 iter scale = ~165 iter, depth 锚定一意收敛
    """

    def __init__(self, config: OptimizerConfig = OptimizerConfig(), device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)

    def optimize(
        self,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        target_mask: torch.Tensor,
        target_depth: Optional[torch.Tensor] = None,
        target_pointmap: Optional[torch.Tensor] = None,  # (H, W, 3) PT3D 相机系 metric
        target_rgb: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        init_R: Optional[torch.Tensor] = None,
        init_t: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        optimize_camera: bool = False,
        vertex_colors: Optional[torch.Tensor] = None,
        rotation_samples: int = 8,
        init_R_candidates: Optional[List[torch.Tensor]] = None,
        verbose: bool = True,
        freeze_scale: Optional[bool] = None,
        yaw_only: Optional[bool] = None,
        floor_constraint: Optional[Dict] = None,
        world_camera_params: Optional[Dict] = None,
        debug_save_dir=None,
        debug_name: str = "obj",
    ) -> Dict:
        # Prepare inputs
        target_mask = target_mask.to(self.device).float()
        while target_mask.ndim > 2:
            target_mask = target_mask.squeeze(0)
        H_full, W_full = target_mask.shape

        mesh_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # ── Mesh extent: 仅打印, 不做 +Z dominant 判定 ──
        # 之前会在 "+Y dominant" 时报 WARN, 但**带手柄的杯子**手柄朝外伸的方向 extent 可能
        # 比"高度"还大, 这是正常的, WARN 是误报. SAM3D 实际输出 y-up, M_LOAD 一致应用.
        if verbose:
            _ext = (mesh_verts.max(0).values - mesh_verts.min(0).values).cpu().numpy()
            print(f"  [mesh] extent (x,y,z) = "
                  f"{[round(float(v), 3) for v in _ext.tolist()]} "
                  f"(z 是 world up; 有手柄/凸出时 y 可能比 z 大, 这是正常的)")
        if target_depth is not None:
            target_depth = target_depth.to(self.device)
        if target_pointmap is not None:
            target_pointmap = target_pointmap.to(self.device).float()
        if target_rgb is not None:
            target_rgb = target_rgb.to(self.device)
        if K is not None:
            K = K.to(self.device)

        # Camera intrinsics
        if K is None:
            fx = fy = (W_full / 2.0) / math.tan(math.pi / 6.0)
            cx, cy = W_full / 2.0, H_full / 2.0
        else:
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()

        # Initial translation from mask if not provided
        if init_t is None:
            init_t = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        else:
            init_t = init_t.to(self.device)
        if init_R is None:
            init_R = torch.eye(3, device=self.device)
        else:
            init_R = init_R.to(self.device)

        # ── world-frame setup (强制开, 不再 backward-compat 老 PT3D-cam frame 路径) ──
        # world_camera_params 必给, 表示 Genesis 相机外参 + 内参 + plane_z. 优化器直接在
        # Genesis world (z-up) 里学 mesh→world 的 R, t, s, 完全跟 render_pt3d_check.py 对齐.
        if world_camera_params is None:
            raise ValueError("world_camera_params is required (world-frame mode is the only path)")
        wc = world_camera_params
        wc_R_view = wc["R_view"].to(self.device).float()
        wc_T_view = wc["T_view"].to(self.device).float()
        wc_fx_full, wc_fy_full = wc["fx"], wc["fy"]
        wc_cx_full, wc_cy_full = wc["cx"], wc["cy"]
        wc_H_full, wc_W_full = wc["H"], wc["W"]
        wc_plane_z = wc["plane_z"]
        if verbose:
            print(f"  [world_frame] cam fixed at single_pos, plane_z={wc_plane_z:.3f}")

        # floor_constraint: 仅含 z_min (mesh +Z 方向最低点). t_z 由 plane_z - s*z_min 确定,
        # (t_x, t_y) 是 2 DOF 自由变量. 不需要 up_axis / b1 / b2 / cam_height 这些老变量了.
        _floor = floor_constraint
        if _floor is not None and verbose:
            print(f"  [floor_lock] t_z = plane_z - s * z_min, "
                  f"z_min(mesh +Z)={float(_floor['y_min']):.4f}")

        # yaw_only: R = R_yaw_z(θ), 直接绕 world +Z 转. R_scene / R_align 在 world-frame
        # 模式下都不需要 (gravity 就是 world -Z, 不用从相机系反推).
        _yaw_only = yaw_only if yaw_only is not None else self.config.yaw_only
        if _yaw_only and verbose:
            print(f"  [yaw_only] R = R_yaw_z(θ) (绕 world +Z 转), 1 DOF")

        from pytorch3d.structures import Meshes
        history: Dict[str, List[float]] = {"depth": [], "mask": [], "total": []}

        # ── Stage 1: depth-anchored (R, t) registration. scale 冻结. ────
        s_frozen = float(init_scale)

        # 两级分辨率金字塔. 高分一点信息更多, 低分先 anchor 大致位置.
        pyr_targets = [(128, 128), (min(384, H_full), min(384, W_full))]
        pyr_targets = [(h, w) for h, w in pyr_targets if h <= H_full and w <= W_full]
        if not pyr_targets:
            pyr_targets = [(H_full, W_full)]

        # 各级别预先准备好 target / renderer, 跨 candidate 复用.
        # coarse-to-fine sigma: L0 用大 sigma 长距吸引, L1 用小 sigma 精对齐.
        level_sigmas = [self.config.stage1_sigma_coarse, self.config.stage1_sigma_fine]
        level_data = []
        for li, (h, w) in enumerate(pyr_targets):
            sh, sw = h / H_full, w / W_full
            tgt_mask_l = F.interpolate(target_mask[None, None].float(), size=(h, w),
                                        mode="bilinear", align_corners=False)[0, 0]
            tgt_mask_lb = (tgt_mask_l > 0.5).float()
            if target_depth is not None:
                tgt_z_l = F.interpolate(target_depth[None, None].float(), size=(h, w),
                                         mode="bilinear", align_corners=False)[0, 0]
            else:
                tgt_z_l = None
            sigma_l = level_sigmas[li] if li < len(level_sigmas) else level_sigmas[-1]
            sil_r, _, rast_r = _make_renderers(
                (h, w), sigma_l, self.device,
                faces_per_pixel=self.config.stage1_faces_per_px,
            )
            # World-frame: 相机外参固定 (R_view, T_view), 内参随分辨率缩放
            from pytorch3d.renderer import PerspectiveCameras as _PerspCam
            world_cams = _PerspCam(
                focal_length=((wc_fx_full * sw, wc_fy_full * sh),),
                principal_point=((wc_cx_full * sw, wc_cy_full * sh),),
                image_size=((h, w),),
                R=wc_R_view[None], T=wc_T_view[None],
                device=self.device, in_ndc=False,
            )
            level_data.append({
                "h": h, "w": w,
                "tgt_mask": tgt_mask_lb, "tgt_z": tgt_z_l,
                "sil_r": sil_r, "rast_r": rast_r, "sigma": sigma_l,
                "world_cameras": world_cams,
            })

        # 构造候选 init_R 列表. 单 init_R 退化为 1 个候选.
        if init_R_candidates is None:
            candidates = [init_R]
        else:
            candidates = [c.to(self.device) for c in init_R_candidates]

        # ── Init pose render (sanity check): 用 init_R + init_t + init_scale 渲一张 PT3D 图 ──
        # 用来视觉确认 mesh 在 (R=I, t=init_t, s=init_scale) 下是直立的、位置大致对.
        # 如果这张图就歪了, 优化器永远拉不正 → 立刻知道 bug 在 init 而不是 stage1.
        if debug_save_dir is not None:
            try:
                from pathlib import Path as _P
                import numpy as _np
                _save_dir = _P(debug_save_dir)
                _save_dir.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    # 用最高级 (fine) 那只相机渲, 全分辨率
                    ld_last = level_data[-1]
                    cams_init = ld_last["world_cameras"]
                    sil_init = ld_last["sil_r"]
                    # init_R 当 R, init_t 当 t, init_scale 当 s
                    mesh_world = mesh_verts * float(init_scale) @ init_R + init_t
                    meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                    alpha = sil_init(meshes, cameras=cams_init)[..., 3].squeeze().cpu().numpy()
                    # 把 alpha 跟 target_mask 叠在一起存盘 (alpha=红, target=绿, 重合=黄)
                    tgt = ld_last["tgt_mask"].cpu().numpy()
                    H_l, W_l = alpha.shape
                    rgb = _np.zeros((H_l, W_l, 3), dtype=_np.uint8)
                    rgb[..., 0] = (alpha * 255).clip(0, 255).astype(_np.uint8)  # rendered → R
                    rgb[..., 1] = (tgt   * 255).clip(0, 255).astype(_np.uint8)  # target   → G
                    try:
                        from PIL import Image
                        Image.fromarray(rgb).save(_save_dir / f"{debug_name}_init.png")
                        if verbose:
                            print(f"  [debug] init render saved: "
                                  f"{_save_dir / (debug_name + '_init.png')} "
                                  f"(R=红 = init_R+init_t+init_scale 渲出来, G=绿 = target mask)")
                    except Exception as _e:
                        print(f"  [debug] init render save failed: {_e}")
            except Exception as _e:
                print(f"  [debug] init render block failed: {_e}")

        # learn_scale: 把 s 也当可学参数学进 Stage 1 (替代 Stage 2 area_ratio).
        # 仅 yaw_only + floor_lock 才支持 (其他模式参数化跟它不兼容).
        _learn_scale = (
            bool(getattr(self.config, "learn_scale", False))
            and _yaw_only and (_floor is not None)
        )
        if _learn_scale and verbose:
            print(f"  [learn_scale] s 进 Adam, init s = {s_frozen:.4f}, "
                  f"lr={self.config.stage1_lr_scale} (s = exp(s_log), 永远 > 0)")

        def _run_stage1_single(init_R_c: torch.Tensor, init_t_override: Optional[torch.Tensor] = None):
            """跑一个候选 init_R 的完整 Stage 1 (两级金字塔), 返回最终 R/t/s/IoU.

            World-frame mode 参数化:
              - 默认: 6D rotation (3DOF) + t (3DOF), s 冻结
              - yaw_only: θ (1DOF) — R = R_yaw_z(θ)
              - yaw_only + floor_constraint: θ (1DOF) + (t_x, t_y) (2DOF), t_z = plane_z - s*z_min
              - + learn_scale: s 也是可学参数 (用 exp 参数化), t_z 跟随 s 变

            init_t_override: 如果给了, 用它覆盖 closure 里的 init_t (用于 (tx, ty) grid search).
            """
            init_t_use = init_t_override if init_t_override is not None else init_t
            if _yaw_only:
                # World-frame: R 直接是 R_yaw_z(θ), θ 从 init_R_c 第一行 atan2 算
                theta_init = float(math.atan2(init_R_c[0, 1].item(), init_R_c[0, 0].item()))
                theta_local = torch.tensor(theta_init, device=self.device,
                                            dtype=init_R_c.dtype).clone().requires_grad_(True)
                r6d_local = None
            else:
                r6d_local = rotation_matrix_to_6d(init_R_c).detach().clone().requires_grad_(True)
                theta_local = None

            use_floor = _floor is not None and _yaw_only
            if use_floor:
                # t = (t_x, t_y, plane_z - s * z_min), 优化 (t_x, t_y) 即可.
                tx_local = torch.tensor(float(init_t_use[0]), device=self.device,
                                        dtype=init_t_use.dtype).clone().requires_grad_(True)
                ty_local = torch.tensor(float(init_t_use[1]), device=self.device,
                                        dtype=init_t_use.dtype).clone().requires_grad_(True)
                t_local = None
            else:
                t_local = init_t_use.detach().clone().requires_grad_(True)
                tx_local = None
                ty_local = None

            # learn_scale: s = exp(s_log), s_log 是 Adam 学的真正参数. exp 保证 s > 0
            # 且对 s_log 的 step 是相对 s 的 multiplicative 更新, 数值稳定.
            if _learn_scale:
                s_log = torch.tensor(math.log(max(float(s_frozen), 1e-6)),
                                      device=self.device,
                                      dtype=init_R_c.dtype).clone().requires_grad_(True)
            else:
                s_log = None

            def _get_s():
                """当前 scale (tensor if learnable, float otherwise)."""
                if s_log is not None:
                    return torch.exp(s_log)
                return s_frozen

            local_history = {"depth": [], "mask": [], "total": []}

            def _build_R():
                if _yaw_only:
                    return _yaw_matrix_world(theta_local)    # 绕 world +Z
                return rotation_6d_to_matrix(r6d_local)

            def _build_t(s_for_offset):
                """floor mode: t_z = plane_z - s * z_min, (t_x, t_y) free. 非 floor: t_local 直接.

                s_for_offset 可以是 float (s 冻结时) 或 tensor (learn_scale 时, 这样 t_z 有 grad 流回 s_log).
                """
                if use_floor:
                    z_min = float(_floor["y_min"])
                    if isinstance(s_for_offset, torch.Tensor):
                        tz = wc_plane_z - s_for_offset * z_min   # 保 grad
                    else:
                        tz = torch.tensor(wc_plane_z - s_for_offset * z_min,
                                          device=self.device, dtype=tx_local.dtype)
                    return torch.stack([tx_local, ty_local, tz])
                return t_local

            for level, ld in enumerate(level_data):
                if use_floor:
                    opt_params = [
                        {"params": [tx_local, ty_local], "lr": self.config.stage1_lr_t},
                    ]
                else:
                    opt_params = [{"params": [t_local], "lr": self.config.stage1_lr_t}]
                if _yaw_only:
                    opt_params.insert(0, {"params": [theta_local], "lr": self.config.stage1_lr_r6d})
                else:
                    opt_params.insert(0, {"params": [r6d_local], "lr": self.config.stage1_lr_r6d})
                if s_log is not None:
                    opt_params.append({"params": [s_log], "lr": self.config.stage1_lr_scale})
                opt = Adam(opt_params)
                iters = self.config.stage1_iters_coarse if level == 0 else self.config.stage1_iters_fine
                pbar = tqdm(range(iters), desc=f"S1 L{level} {ld['h']}x{ld['w']}",
                            disable=not verbose, leave=False)
                for _ in pbar:
                    opt.zero_grad()
                    R_c = _build_R()
                    s_c = _get_s()                        # tensor if learnable, float otherwise
                    t_c = _build_t(s_c)                   # t_z 由 s_c 算, 学 scale 时 grad 流回 s_log
                    # World-frame: mesh 摆 world, 相机外参固定 (R_view, T_view)
                    mesh_world = mesh_verts * s_c @ R_c + t_c
                    meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                    cameras = ld["world_cameras"]
                    rendered = ld["sil_r"](meshes, cameras=cameras)
                    alpha = rendered[..., 3].squeeze()
                    frag = ld["rast_r"](meshes, cameras=cameras)
                    rend_z = frag.zbuf[0, ..., 0]
                    overlap = (alpha > 0.5) & (ld["tgt_mask"] > 0.5) & (rend_z > 0)
                    n_over = int(overlap.sum())

                    # Soft-IoU loss: 直接对应我们关心的 metric, red/green 区域都有梯度.
                    inter_soft = (alpha * ld["tgt_mask"]).sum()
                    union_soft = alpha.sum() + ld["tgt_mask"].sum() - inter_soft
                    soft_iou = inter_soft / (union_soft + 1e-6)
                    iou_loss = (1.0 - soft_iou) * self.config.stage1_iou_weight

                    if ld["tgt_z"] is not None and n_over > 20:
                        depth_loss = F.smooth_l1_loss(
                            rend_z[overlap], ld["tgt_z"][overlap],
                            beta=self.config.stage1_depth_beta,
                        )
                        loss = depth_loss + iou_loss
                        loss_kind = f"d+iou  over={n_over}  iou={soft_iou.item():.3f}"
                    else:
                        # 没 overlap → 单 IoU loss 把 mesh 吸过来 (soft-IoU 永远有梯度)
                        depth_loss = torch.tensor(0.0, device=self.device)
                        loss = iou_loss
                        loss_kind = f"iou-only  iou={soft_iou.item():.3f}"

                    mask_loss = iou_loss
                    loss.backward()
                    opt.step()
                    local_history["depth"].append(float(depth_loss))
                    local_history["mask"].append(float(mask_loss))
                    local_history["total"].append(float(loss))
                    if verbose:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}", "type": loss_kind})

            with torch.no_grad():
                R_local = _build_R().detach()
                s_local_final = float(_get_s().item()) if s_log is not None else float(s_frozen)
                t_local_final = _build_t(s_local_final).detach()

            # 用最末级别评估 IoU (s_local_final, R_local, t_local_final)
            with torch.no_grad():
                ld_last = level_data[-1]
                mesh_world = mesh_verts * s_local_final @ R_local + t_local_final
                meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                cameras = ld_last["world_cameras"]
                alpha = ld_last["sil_r"](meshes, cameras=cameras)[..., 3].squeeze()
                pred_b = alpha > 0.5
                tgt_b = ld_last["tgt_mask"] > 0.5
                inter = (pred_b & tgt_b).sum().float()
                union = (pred_b | tgt_b).sum().float().clamp(min=1.0)
                iou_l = (inter / union).item()
            return R_local, t_local_final, iou_l, local_history, s_local_final

        if verbose:
            yaw_tag = ", yaw_only=True (R=R_yaw_z, roll/pitch=0)" if _yaw_only else ""
            scale_tag = ", learn_scale=True (s ∈ Adam)" if _learn_scale else f", scale={s_frozen:.4f} (frozen)"
            print(f"  Stage 1: depth-anchored (R, t{', s' if _learn_scale else ''}) registration"
                  f"{scale_tag}, {len(candidates)} candidate(s){yaw_tag}")

        # (tx, ty) grid: N×N points in init_t.xy ± radius. N=1 退化成 [(init_t.xy)].
        _grid_n = int(getattr(self.config, "stage1_txy_grid", 1))
        _grid_r = float(getattr(self.config, "stage1_txy_grid_radius", 0.5))
        if _grid_n <= 1:
            txy_offsets = [(0.0, 0.0)]
        else:
            import numpy as _np
            _g = _np.linspace(-_grid_r, _grid_r, _grid_n)
            txy_offsets = [(float(dx), float(dy)) for dx in _g for dy in _g]
        if verbose and len(txy_offsets) > 1:
            print(f"  Stage 1 (tx,ty) grid: {_grid_n}×{_grid_n}={len(txy_offsets)} 起点, "
                  f"±{_grid_r:.2f}m. 总 candidates = {len(candidates)} yaw × {len(txy_offsets)} txy = "
                  f"{len(candidates)*len(txy_offsets)}.")

        best_R, best_t, best_iou, best_hist, best_s = None, None, -1.0, None, float(s_frozen)
        cand_log = []
        for ci, R_init_c in enumerate(candidates):
            for gi, (dx, dy) in enumerate(txy_offsets):
                if len(txy_offsets) > 1:
                    init_t_g = torch.tensor(
                        [float(init_t[0]) + dx, float(init_t[1]) + dy, float(init_t[2])],
                        device=self.device, dtype=init_t.dtype,
                    )
                else:
                    init_t_g = None
                R_c, t_c, iou_c, hist_c, s_c = _run_stage1_single(R_init_c, init_t_override=init_t_g)
                cand_log.append({"ci": ci, "gi": gi, "iou": iou_c, "scale": s_c})
                if verbose:
                    tag = f"yaw{ci}" if len(txy_offsets) == 1 else f"yaw{ci}/g{gi}(dx={dx:+.2f},dy={dy:+.2f})"
                    if _learn_scale:
                        print(f"    candidate {tag}: stage1 IoU = {iou_c:.4f}  s = {s_c:.4f}  t=[{float(t_c[0]):+.3f},{float(t_c[1]):+.3f}]")
                    else:
                        print(f"    candidate {tag}: stage1 IoU = {iou_c:.4f}")
                if iou_c > best_iou:
                    best_R, best_t, best_iou, best_hist, best_s = R_c, t_c, iou_c, hist_c, s_c

        R_final = best_R
        t_final = best_t
        history = best_hist if best_hist is not None else history
        if verbose and len(candidates) > 1:
            extra = f"  s={best_s:.4f}" if _learn_scale else ""
            print(f"  Stage 1 winner: IoU={best_iou:.4f}{extra}")
        # learn_scale: 把 s_frozen 更新成 Stage 1 学到的 s, 后续 Stage 2 + 最终评估都用它
        if _learn_scale:
            s_frozen = float(best_s)

        sil_full, _, _ = _make_renderers((H_full, W_full), 1e-4, self.device)

        # 全分辨率固定相机, Stage 2 + refine + final eval 都用它
        from pytorch3d.renderer import PerspectiveCameras as _PerspCam
        world_cameras_full = _PerspCam(
            focal_length=((wc_fx_full, wc_fy_full),),
            principal_point=((wc_cx_full, wc_cy_full),),
            image_size=((wc_H_full, wc_W_full),),
            R=wc_R_view[None], T=wc_T_view[None],
            device=self.device, in_ndc=False,
        )

        def _adjust_t_for_floor(t, s_new):
            """floor mode: 重算 t_z 让 mug 底维持贴 plane_z. 非 floor 模式直返."""
            if _floor is None:
                return t
            z_min = float(_floor["y_min"])   # mesh +Z 方向最低点
            new_tz = float(wc_plane_z - s_new * z_min)
            return torch.stack([t[0], t[1],
                                 torch.tensor(new_tz, device=t.device, dtype=t.dtype)])

        # ── freeze_scale 短路: scale 不动, 跳过 Stage 2 + Stage 1 refine ──
        _freeze_scale = freeze_scale if freeze_scale is not None else self.config.freeze_scale
        if _learn_scale:
            # Stage 1 已经把 s 学进来了 (s_frozen 已经更新成 best_s)
            if verbose:
                print(f"\n  Stage 2 + refine SKIPPED (learn_scale=True, Stage 1 已学 s)."
                      f"\n  s_final = {s_frozen:.4f} (from Stage 1 winner)")
            s_final = float(s_frozen)
        elif _freeze_scale:
            if verbose:
                print(f"\n  Stage 2 + refine SKIPPED (freeze_scale=True). "
                      f"s_final = init_scale = {float(init_scale):.4f}")
            s_final = float(init_scale)
        else:
            # ── Stage 2: 解析求 scale.  s_new = s * sqrt(area_target / area_pred)  ──
            if verbose:
                print(f"\n  Stage 2: analytical scale (area ratio)")
            target_area = float(target_mask.bool().sum())
            s_curr = float(init_scale)
            with torch.no_grad():
                for it in range(6):
                    t_for_iter = _adjust_t_for_floor(t_final, s_curr)
                    mesh_world = mesh_verts * s_curr @ R_final + t_for_iter
                    meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                    cameras = world_cameras_full
                    alpha = sil_full(meshes, cameras=cameras)[..., 3].squeeze()
                    pred_area = float((alpha > 0.5).float().sum())
                    if pred_area < 1.0:
                        if verbose:
                            print(f"    iter {it}: pred area 0, can't fit. break.")
                        break
                    ratio = (target_area / pred_area) ** 0.5
                    s_new = s_curr * ratio
                    rel = abs(s_new - s_curr) / max(s_curr, 1e-8)
                    if verbose:
                        print(f"    iter {it}: s {s_curr:.4f} → {s_new:.4f}  "
                              f"(area_pred={int(pred_area)}, target={int(target_area)}, ratio={ratio:.3f})")
                    s_curr = s_new
                    if rel < 5e-3:
                        break
            s_final = s_curr

            # ── Stage 1 refine: 用新 scale 再 refine R, t (winner-only) ────
            # Stage 1 一开始 freeze 在 init_scale, 找 (R, t) 是为那个 scale 优化的.
            # Stage 2 改了 scale, 但 (R, t) 还是旧的, 对新 scale 不再最优.
            # 这里用新 s_final 再跑一遍 Stage 1, 起点是 winner (R, t), 让 (R, t) 重新对齐新 scale.
            # 关键: 同时给 init_t 覆盖成 winner_t, _run_stage1_single 里读的是 enclosing scope 的 init_t.
            # 总是接受 refine (它就是用更准 scale 重新优化, 起点是当前 R/t, Adam 不会让它变更差).
            if abs(s_final - init_scale) / max(init_scale, 1e-8) > 0.02:
                if verbose:
                    print(f"\n  Stage 1 refine (scale 变了 "
                          f"{abs(s_final-init_scale)/init_scale*100:.1f}%, 重新 fit R, t at s={s_final:.4f}):")
                s_frozen = float(s_final)             # 覆盖外层 closure 用的 s_frozen
                init_t = t_final.detach().clone()     # 覆盖外层 closure 用的 init_t
                R_refined, t_refined, iou_refined, _, _ = _run_stage1_single(R_final.clone())
                R_final, t_final = R_refined, t_refined
                best_iou = iou_refined
                if verbose:
                    print(f"    refined: stage1 IoU at new scale = {iou_refined:.4f}")

                # 再来一轮快 Stage 2: R, t 变了, scale 可能也要微调 (一般只 1-2%)
                if verbose:
                    print(f"\n  Stage 2 refine (after Stage 1 refine):")
                with torch.no_grad():
                    for it in range(3):
                        # floor mode 下 t_z 要跟随 s_curr 重算 (跟 Stage 2 一致), 否则 mesh
                        # 在 area-ratio 评估时位置不对 → pred_area 偏 → scale 收敛偏.
                        t_for_iter = _adjust_t_for_floor(t_final, s_curr)
                        mesh_world = mesh_verts * s_curr @ R_final + t_for_iter
                        meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                        cameras = world_cameras_full
                        alpha = sil_full(meshes, cameras=cameras)[..., 3].squeeze()
                        pred_area = float((alpha > 0.5).float().sum())
                        if pred_area < 1.0:
                            break
                        ratio = (target_area / pred_area) ** 0.5
                        s_new = s_curr * ratio
                        rel = abs(s_new - s_curr) / max(s_curr, 1e-8)
                        if verbose:
                            print(f"    iter {it}: s {s_curr:.4f} → {s_new:.4f}  ratio={ratio:.3f}")
                        s_curr = s_new
                        if rel < 5e-3:
                            break
                s_final = s_curr

        # floor mode: 在 final eval / save 前再 sync 一次 t_final 沿 up 的分量,
        # 防止小 scale 偏移让 mug 浮起 / 沉底.
        t_final = _adjust_t_for_floor(t_final, s_final)

        # ── final IoU at full res, binary ────────────────────────────────
        with torch.no_grad():
            mesh_world = mesh_verts * s_final @ R_final + t_final
            meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
            cameras = world_cameras_full
            rendered = sil_full(meshes, cameras=cameras)
            pred_b = rendered[..., 3].squeeze() > 0.5
            tgt_b = target_mask.bool()
            inter = (pred_b & tgt_b).sum().float()
            union = (pred_b | tgt_b).sum().float().clamp(min=1.0)
            final_iou = (inter / union).item()

        if verbose:
            # ── 终局 debug: scale / t / world mesh bbox / floor check ──────────
            with torch.no_grad():
                mesh_world_final = mesh_verts * s_final @ R_final + t_final
                z_min_world = float(mesh_world_final[:, 2].min().item())
                z_max_world = float(mesh_world_final[:, 2].max().item())
                world_height = z_max_world - z_min_world
                # canonical mesh +Z extent (M_LOAD 之后), 用来反查 scale 含义
                z_min_can = float((mesh_verts[:, 2].min()).item())
                z_max_can = float((mesh_verts[:, 2].max()).item())
                can_extent = z_max_can - z_min_can
            print(f"  Final IoU={final_iou:.4f}  scale={s_final:.4f}  "
                  f"t=[{t_final[0].item():.3f}, {t_final[1].item():.3f}, {t_final[2].item():.3f}]")
            print(f"    [debug] mesh canonical +Z extent = {can_extent:.4f} "
                  f"(z_min={z_min_can:.4f}, z_max={z_max_can:.4f})")
            print(f"    [debug] world mesh height = scale · extent = "
                  f"{s_final:.4f} · {can_extent:.4f} = {s_final * can_extent:.4f} m")
            print(f"    [debug] world mesh z range = [{z_min_world:.4f}, {z_max_world:.4f}]")
            if _floor is not None:
                float_above_floor = z_min_world - wc_plane_z
                tag = "贴地 ✓" if abs(float_above_floor) < 1e-3 else f"偏离 plane_z {float_above_floor*1000:+.1f}mm"
                print(f"    [debug] plane_z={wc_plane_z:.4f}, mesh 底 - plane_z = {float_above_floor:+.4f}m  ({tag})")
                # floor_lock 数学检查: t_z 应该 = plane_z - s·z_min_canonical
                expected_tz = wc_plane_z - s_final * z_min_can
                actual_tz = float(t_final[2].item())
                print(f"    [debug] floor_lock check: expected t_z = plane_z - s·z_min_can "
                      f"= {wc_plane_z:.3f} - {s_final:.4f}·({z_min_can:.4f}) = {expected_tz:.4f}; "
                      f"actual t_z = {actual_tz:.4f}  "
                      f"(diff = {(actual_tz - expected_tz)*1000:+.2f}mm)")

        # ── 出口 assert_yaw: 验 optimizer 真的输出纯 yaw R (yaw_only=True 时必须 True) ──
        from .utils import assert_yaw_pure
        if _yaw_only:
            assert_yaw_pure(R_final, f"optimizer-out ({debug_name})")

        return {
            "R": R_final.detach().cpu(),
            "t": t_final.detach().cpu(),
            "scale": s_final,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "history": history,
            "candidate_iou": final_iou,   # 不再有 candidate, 复用 final
            "final_iou": final_iou,
            "candidate_log": cand_log,
        }


def optimize_multiple_objects(
    optimizer: PoseOptimizer,
    mesh_list: List[Dict],
    target_mask: torch.Tensor,
    target_depth: Optional[torch.Tensor] = None,
    target_pointmap: Optional[torch.Tensor] = None,
    target_rgb: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> List[Dict]:
    results = []
    order = sorted(range(len(mesh_list)), key=lambda i: mesh_list[i]["mask"].sum().item(), reverse=True)
    for idx in order:
        obj = mesh_list[idx]
        if verbose:
            print(f"\nOptimizing: {obj.get('name', f'object_{idx}')}")
        _name = obj.get("name", f"object_{idx}")
        result = optimizer.optimize(
            mesh_verts=obj["verts"], mesh_faces=obj["faces"],
            target_mask=obj["mask"], target_depth=target_depth,
            target_pointmap=target_pointmap,
            target_rgb=target_rgb, K=K,
            init_R=obj.get("init_R"), init_t=obj.get("init_t"),
            init_scale=obj.get("init_scale", 1.0),
            init_R_candidates=obj.get("init_R_candidates"),
            vertex_colors=obj.get("vertex_colors"),
            verbose=verbose,
            freeze_scale=obj.get("freeze_scale"),
            yaw_only=obj.get("yaw_only"),
            floor_constraint=obj.get("floor_constraint"),
            world_camera_params=obj.get("world_camera_params"),
            debug_save_dir=obj.get("debug_save_dir"),
            debug_name=_name,
        )
        result["name"] = obj.get("name", f"object_{idx}")
        results.append(result)
    return results
