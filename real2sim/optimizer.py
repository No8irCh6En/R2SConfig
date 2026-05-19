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


def _make_cameras(fx: float, fy: float, cx: float, cy: float, H: int, W: int,
                  R: torch.Tensor, t: torch.Tensor, device: torch.device):
    from pytorch3d.renderer import PerspectiveCameras
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        image_size=((H, W),), R=R[None], T=t[None],
        device=device, in_ndc=False,
    )


def sample_initial_rotations(n: int = 8, device: torch.device = torch.device("cpu")) -> List[torch.Tensor]:
    """Generate candidate rotations.

    前 n 个: identity + (n-1) 个绕 Y 轴 (假设物体已经基本正立)。
    最后 4 个: 绕 X/Z 轴 90° 的 "把横躺物体立起来" rescue rotation。
    这处理 canonical mesh 长轴**不在** Y 上的情况 (例如 mug tree 长轴在 X 上)。
    """
    rotations = [torch.eye(3, device=device)]
    for i in range(1, n):
        angle = 2 * math.pi * i / n
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], device=device)
        rotations.append(R)

    # 绕 Z 轴 ±90° (把 X 长轴 swap 到 Y)
    R_z90 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                         device=device, dtype=torch.float32)
    R_zN90 = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                          device=device, dtype=torch.float32)
    # 绕 X 轴 ±90° (把 Z 长轴 swap 到 Y)
    R_x90 = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                         device=device, dtype=torch.float32)
    R_xN90 = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                          device=device, dtype=torch.float32)
    rotations.extend([R_z90, R_zN90, R_x90, R_xN90])
    return rotations


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
    ) -> Dict:
        # Prepare inputs
        target_mask = target_mask.to(self.device).float()
        while target_mask.ndim > 2:
            target_mask = target_mask.squeeze(0)
        H_full, W_full = target_mask.shape

        mesh_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)
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
            level_data.append({
                "h": h, "w": w, "fx": fx * sw, "fy": fy * sh,
                "cx": cx * sw, "cy": cy * sh,
                "tgt_mask": tgt_mask_lb, "tgt_z": tgt_z_l,
                "sil_r": sil_r, "rast_r": rast_r, "sigma": sigma_l,
            })

        # ── Chamfer (3D) 预计算: 从 pointmap 在 mask 内采 N_sub 个点作为目标 ──
        # mesh 一侧也做固定子采样, 跨 candidate / iter 复用同一索引 (rng 稳定).
        target_chamfer_pts = None   # (N_sub, 3) 在 PT3D 相机系
        mesh_extent_norm = 1.0
        mesh_sub_idx = None
        if target_pointmap is not None:
            mask_bool = target_mask.bool()
            if mask_bool.any():
                pts_all = target_pointmap[mask_bool]            # (N, 3)
                valid = torch.isfinite(pts_all).all(dim=-1) & (pts_all[:, 2].abs() > 1e-6)
                pts_valid = pts_all[valid]
                N = int(pts_valid.shape[0])
                n_sub = min(N, self.config.stage1_chamfer_max_pts)
                if n_sub > 0:
                    gen = torch.Generator(device=pts_valid.device).manual_seed(0)
                    sel = torch.randperm(N, generator=gen, device=pts_valid.device)[:n_sub]
                    target_chamfer_pts = pts_valid[sel].contiguous()
                    extent_canonical = (mesh_verts.max(0).values - mesh_verts.min(0).values).max().item()
                    mesh_extent_norm = max(extent_canonical * float(init_scale), 1e-3)
                    M = int(mesh_verts.shape[0])
                    m_sub = min(M, self.config.stage1_chamfer_max_verts)
                    if m_sub < M:
                        gen2 = torch.Generator(device=mesh_verts.device).manual_seed(0)
                        mesh_sub_idx = torch.randperm(M, generator=gen2, device=mesh_verts.device)[:m_sub]
                    if verbose:
                        print(f"  Chamfer: {n_sub}/{N} pointmap pts, "
                              f"{m_sub if mesh_sub_idx is not None else M}/{M} mesh verts, "
                              f"extent_norm={mesh_extent_norm:.3f}m")

        # 构造候选 init_R 列表. 单 init_R 退化为 1 个候选.
        if init_R_candidates is None:
            candidates = [init_R]
        else:
            candidates = [c.to(self.device) for c in init_R_candidates]

        def _chamfer_p2v(R_world, t_world, s_world):
            """单向 chamfer: P_tgt → V_pred. 返回 (chamfer_dist_meters, chamfer_norm).
            如果 target_chamfer_pts 为 None 直接返回 (0, 0)."""
            if target_chamfer_pts is None:
                return (torch.tensor(0.0, device=self.device),
                        torch.tensor(0.0, device=self.device))
            V_pred = mesh_verts * s_world
            V_pred = V_pred @ R_world + t_world         # PT3D row-vec, 同坐标系
            if mesh_sub_idx is not None:
                V_pred = V_pred[mesh_sub_idx]
            d = torch.cdist(target_chamfer_pts, V_pred)  # (N_sub, M_sub)
            nn_dist, _ = d.min(dim=1)
            chamfer = nn_dist.mean()
            return chamfer, chamfer / mesh_extent_norm

        def _run_stage1_single(init_R_c: torch.Tensor):
            """跑一个候选 init_R 的完整 Stage 1 (两级金字塔), 返回最终 R/t / IoU / chamfer."""
            r6d_local = rotation_matrix_to_6d(init_R_c).detach().clone().requires_grad_(True)
            t_local = init_t.detach().clone().requires_grad_(True)
            local_history = {"depth": [], "mask": [], "total": [], "chamfer": []}

            for level, ld in enumerate(level_data):
                opt = Adam([
                    {"params": [r6d_local], "lr": self.config.stage1_lr_r6d},
                    {"params": [t_local],   "lr": self.config.stage1_lr_t},
                ])
                iters = self.config.stage1_iters_coarse if level == 0 else self.config.stage1_iters_fine
                use_chamfer_this_level = (
                    target_chamfer_pts is not None
                    and (level == 1 or self.config.stage1_chamfer_in_coarse)
                )
                pbar = tqdm(range(iters), desc=f"S1 L{level} {ld['h']}x{ld['w']}",
                            disable=not verbose, leave=False)
                for _ in pbar:
                    opt.zero_grad()
                    R_c = rotation_6d_to_matrix(r6d_local)
                    cameras = _make_cameras(ld["fx"], ld["fy"], ld["cx"], ld["cy"],
                                            ld["h"], ld["w"], R_c, t_local, self.device)
                    meshes = Meshes(verts=[mesh_verts * s_frozen], faces=[mesh_faces])
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

                    # 3D Chamfer (只在 fine level, 由 config 控制). 对绕长轴 yaw 敏感.
                    if use_chamfer_this_level:
                        cham_m, cham_norm = _chamfer_p2v(R_c, t_local, s_frozen)
                        chamfer_loss = cham_norm * self.config.stage1_chamfer_weight
                        loss = loss + chamfer_loss
                        loss_kind += f"  cham={cham_m.item()*100:.1f}cm"
                    else:
                        chamfer_loss = torch.tensor(0.0, device=self.device)

                    mask_loss = iou_loss
                    loss.backward()
                    opt.step()
                    local_history["depth"].append(float(depth_loss))
                    local_history["mask"].append(float(mask_loss))
                    local_history["chamfer"].append(float(chamfer_loss))
                    local_history["total"].append(float(loss))
                    if verbose:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}", "type": loss_kind})

            R_local = rotation_6d_to_matrix(r6d_local).detach()
            t_local_final = t_local.detach()

            # 用最末级别评估 IoU + chamfer (s_frozen, R_local, t_local_final)
            with torch.no_grad():
                ld_last = level_data[-1]
                cameras = _make_cameras(ld_last["fx"], ld_last["fy"], ld_last["cx"], ld_last["cy"],
                                        ld_last["h"], ld_last["w"], R_local, t_local_final, self.device)
                meshes = Meshes(verts=[mesh_verts * s_frozen], faces=[mesh_faces])
                alpha = ld_last["sil_r"](meshes, cameras=cameras)[..., 3].squeeze()
                pred_b = alpha > 0.5
                tgt_b = ld_last["tgt_mask"] > 0.5
                inter = (pred_b & tgt_b).sum().float()
                union = (pred_b | tgt_b).sum().float().clamp(min=1.0)
                iou_l = (inter / union).item()
                cham_m_l, cham_norm_l = _chamfer_p2v(R_local, t_local_final, s_frozen)
                cham_m_l = float(cham_m_l)
                cham_norm_l = float(cham_norm_l)
            return R_local, t_local_final, iou_l, cham_m_l, cham_norm_l, local_history

        if verbose:
            print(f"  Stage 1: depth-anchored (R, t) registration, "
                  f"scale={s_frozen:.4f} (frozen), {len(candidates)} candidate(s)")
            if target_chamfer_pts is not None:
                print(f"    selection score = IoU - {self.config.selection_chamfer_weight:.2f} · chamfer_norm")

        best_R, best_t, best_iou, best_hist = None, None, -1.0, None
        best_cham_m, best_cham_norm, best_score = float("inf"), float("inf"), -1e9
        cand_log = []
        for ci, R_init_c in enumerate(candidates):
            R_c, t_c, iou_c, cham_m_c, cham_norm_c, hist_c = _run_stage1_single(R_init_c)
            score_c = iou_c - self.config.selection_chamfer_weight * cham_norm_c
            cand_log.append({"ci": ci, "iou": iou_c, "cham_m": cham_m_c,
                             "cham_norm": cham_norm_c, "score": score_c})
            if verbose:
                if target_chamfer_pts is not None:
                    print(f"    candidate {ci}: IoU={iou_c:.4f}  "
                          f"chamfer={cham_m_c*100:.2f}cm (norm={cham_norm_c:.3f})  "
                          f"score={score_c:.4f}")
                else:
                    print(f"    candidate {ci}: stage1 IoU = {iou_c:.4f}")
            if score_c > best_score:
                best_R, best_t, best_iou, best_hist = R_c, t_c, iou_c, hist_c
                best_cham_m, best_cham_norm, best_score = cham_m_c, cham_norm_c, score_c

        R_final = best_R
        t_final = best_t
        history = best_hist if best_hist is not None else history
        if verbose and len(candidates) > 1:
            if target_chamfer_pts is not None:
                print(f"  Stage 1 winner: IoU={best_iou:.4f}  "
                      f"chamfer={best_cham_m*100:.2f}cm  score={best_score:.4f}")
            else:
                print(f"  Stage 1 winner: IoU={best_iou:.4f}")

        # ── Stage 2: 解析求 scale.  s_new = s * sqrt(area_target / area_pred)  ──
        if verbose:
            print(f"\n  Stage 2: analytical scale (area ratio)")
        sil_full, _, _ = _make_renderers((H_full, W_full), 1e-4, self.device)
        target_area = float(target_mask.bool().sum())
        s_curr = float(init_scale)
        with torch.no_grad():
            for it in range(6):
                cameras = _make_cameras(fx, fy, cx, cy, H_full, W_full, R_final, t_final, self.device)
                meshes = Meshes(verts=[mesh_verts * s_curr], faces=[mesh_faces])
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
            R_refined, t_refined, iou_refined, cham_m_ref, cham_norm_ref, _ = \
                _run_stage1_single(R_final.clone())
            R_final, t_final = R_refined, t_refined
            best_iou = iou_refined
            best_cham_m = cham_m_ref
            best_cham_norm = cham_norm_ref
            if verbose:
                if target_chamfer_pts is not None:
                    print(f"    refined: IoU={iou_refined:.4f}  "
                          f"chamfer={cham_m_ref*100:.2f}cm")
                else:
                    print(f"    refined: stage1 IoU at new scale = {iou_refined:.4f}")

            # 再来一轮快 Stage 2: R, t 变了, scale 可能也要微调 (一般只 1-2%)
            if verbose:
                print(f"\n  Stage 2 refine (after Stage 1 refine):")
            with torch.no_grad():
                for it in range(3):
                    cameras = _make_cameras(fx, fy, cx, cy, H_full, W_full, R_final, t_final, self.device)
                    meshes = Meshes(verts=[mesh_verts * s_curr], faces=[mesh_faces])
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

        # ── final IoU at full res, binary ────────────────────────────────
        with torch.no_grad():
            cameras = _make_cameras(fx, fy, cx, cy, H_full, W_full, R_final, t_final, self.device)
            meshes = Meshes(verts=[mesh_verts * s_final], faces=[mesh_faces])
            rendered = sil_full(meshes, cameras=cameras)
            pred_b = rendered[..., 3].squeeze() > 0.5
            tgt_b = target_mask.bool()
            inter = (pred_b & tgt_b).sum().float()
            union = (pred_b | tgt_b).sum().float().clamp(min=1.0)
            final_iou = (inter / union).item()

        # 最终 chamfer (用 s_final, R_final, t_final 在 PT3D 系下重算)
        if target_chamfer_pts is not None:
            with torch.no_grad():
                cham_m_final, cham_norm_final = _chamfer_p2v(R_final, t_final, s_final)
                cham_m_final = float(cham_m_final)
                cham_norm_final = float(cham_norm_final)
            final_score = final_iou - self.config.selection_chamfer_weight * cham_norm_final
        else:
            cham_m_final = float("nan")
            cham_norm_final = float("nan")
            final_score = final_iou
        if verbose:
            extra = ""
            if target_chamfer_pts is not None:
                extra = (f"  chamfer={cham_m_final*100:.2f}cm "
                         f"(norm={cham_norm_final:.3f})  score={final_score:.4f}")
            print(f"  Final IoU={final_iou:.4f}  scale={s_final:.3f}  "
                  f"t={[round(x, 3) for x in t_final.tolist()]}{extra}")

        return {
            "R": R_final.detach().cpu(),
            "t": t_final.detach().cpu(),
            "scale": s_final,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "history": history,
            "candidate_iou": final_iou,   # 不再有 candidate, 复用 final
            "final_iou": final_iou,
            "chamfer_m": cham_m_final,
            "chamfer_norm": cham_norm_final,
            "final_score": final_score,
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
        )
        result["name"] = obj.get("name", f"object_{idx}")
        results.append(result)
    return results
