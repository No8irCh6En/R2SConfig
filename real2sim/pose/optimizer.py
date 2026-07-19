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

from ..perception.camera import rotation_6d_to_matrix, rotation_matrix_to_6d, intrinsics_from_hfov, build_camera_matrix
from ..config import OptimizerConfig


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


def _yaw_matrix_world_batch(theta: torch.Tensor) -> torch.Tensor:
    """Batched version of `_yaw_matrix_world`: theta (B,) -> R (B, 3, 3).

    Same +Z row-vec yaw convention, one 3x3 per candidate. Used by the batched
    multi-start optimizer so B yaw candidates render in ONE rasterization call.
    Identical to stacking `_yaw_matrix_world(theta[i])` over i (verified in tests).
    """
    theta = theta.reshape(-1)
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    row0 = torch.stack([c, s, z], dim=-1)      # (B, 3): +X → (cos, sin, 0)
    row1 = torch.stack([-s, c, z], dim=-1)     # (B, 3): +Y → (-sin, cos, 0)
    row2 = torch.stack([z, z, o], dim=-1)      # (B, 3): +Z preserved
    return torch.stack([row0, row1, row2], dim=1)   # (B, 3, 3)


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


# ─────────────────────────────────────────────────────────────────────
# Shared multi-view helpers (used by BOTH optimize_multi_view and the
# batched optimize_multi_view_multistart). Extracted verbatim so the two
# entry points can't drift apart.
# ─────────────────────────────────────────────────────────────────────

def _normalize_views(views: List[Dict], device: torch.device) -> List[Dict]:
    """Move each view's tensors to `device`, squeeze masks/depth to (H, W)."""
    norm_views = []
    for vi, v in enumerate(views):
        mask = v["mask"].to(device).float()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        R_view = v["R_view"].to(device).float()
        T_view = v["T_view"].to(device).float()
        depth = v.get("depth")
        if depth is not None:
            depth = depth.to(device).float()
            while depth.ndim > 2:
                depth = depth.squeeze(0)
        _nm = v.get("name", f"view{vi}")
        norm_views.append({
            "name": _nm,
            "cam": _nm.split("/")[0],
            "mask": mask,
            "depth": depth,
            "R_view": R_view, "T_view": T_view,
            "fx": float(v["fx"]), "fy": float(v["fy"]),
            "cx": float(v["cx"]), "cy": float(v["cy"]),
            "H": int(v["H"]), "W": int(v["W"]),
            # Per-view loss weight (default 1.0). Lets the caller down-weight
            # views with bad/occluded masks. NOT auto-set by frame index —
            # "later wrist frame = better" is unreliable (gripper occludes the
            # mug exactly when it's closest), so weighting is opt-in & explicit.
            "weight": float(v.get("weight", 1.0)),
        })
    return norm_views


def _build_view_levels(config, norm_views: List[Dict], device: torch.device,
                       use_dt: bool, crop_pad: float = 0.0) -> List[List[Dict]]:
    """Per view, build a 2-level (coarse 128², then full-res) render pyramid.

    Returns view_levels[v_idx][l_idx] = {h, w, tgt_mask, tgt_z, sil_r, rast_r,
    world_cameras, (dt_out, dt_in if use_dt)}.

    `crop_pad` > 0 renders ONLY the object's mask bounding box, padded by
    `crop_pad`×bbox-size (min 8 px), at NATIVE resolution instead of the full
    frame. The object is often <2% of the frame, so this cuts render pixels ~10×
    with an IDENTICAL IoU/pose (the dropped pixels are background in neither the
    mask nor the render, so they don't change intersection or union). Only the
    principal point (cx→cx-x0, cy→cy-y0) and image_size shift; the world→camera
    projection (R, T, focal) is untouched. The object must stay INSIDE the window
    during optimization — a generous pad + a good init_t keep it there; if it can
    drift far, raise crop_pad or leave it 0.
    """
    from pytorch3d.renderer import PerspectiveCameras as _PerspCam
    coarse_target = (128, 128)
    level_sigmas = [config.stage1_sigma_coarse, config.stage1_sigma_fine]
    view_levels: List[List[Dict]] = []
    for v in norm_views:
        H0, W0 = int(v["H"]), int(v["W"])
        # crop window [y0:y1, x0:x1] in full-frame px (whole frame if crop off).
        y0, x0, y1, x1 = 0, 0, H0, W0
        if crop_pad and crop_pad > 0:
            ys, xs = torch.where(v["mask"] > 0.5)
            if len(ys) > 0:
                by0, by1 = int(ys.min()), int(ys.max()) + 1
                bx0, bx1 = int(xs.min()), int(xs.max()) + 1
                ph = max(int((by1 - by0) * crop_pad), 8)
                pw = max(int((bx1 - bx0) * crop_pad), 8)
                y0, y1 = max(0, by0 - ph), min(H0, by1 + ph)
                x0, x1 = max(0, bx0 - pw), min(W0, bx1 + pw)
        mask_c = v["mask"][y0:y1, x0:x1]
        depth_c = v["depth"][y0:y1, x0:x1] if v["depth"] is not None else None
        H_full, W_full = int(y1 - y0), int(x1 - x0)     # crop dims = effective frame
        cx_c, cy_c = v["cx"] - x0, v["cy"] - y0         # principal point in crop frame
        # coarse level: full-frame caps at 128²; a crop (~100px) would equal the fine
        # level and collapse the pyramid, so use ~half the crop to keep coarse→fine.
        if crop_pad and crop_pad > 0:
            c_h, c_w = max(48, H_full // 2), max(48, W_full // 2)
        else:
            c_h, c_w = coarse_target
        sizes = [(min(c_h, H_full), min(c_w, W_full)),
                 (H_full, W_full)]
        sizes = [(h, w) for (h, w) in sizes if h > 0 and w > 0]
        # dedupe consecutive same sizes
        if len(sizes) > 1 and sizes[0] == sizes[1]:
            sizes = sizes[:1]
        per_lvl = []
        for li, (h, w) in enumerate(sizes):
            sh, sw = h / H_full, w / W_full
            tgt_mask_l = F.interpolate(mask_c[None, None], size=(h, w),
                                       mode="bilinear", align_corners=False)[0, 0]
            tgt_mask_lb = (tgt_mask_l > 0.5).float()
            tgt_z_l = None
            if depth_c is not None:
                tgt_z_l = F.interpolate(depth_c[None, None], size=(h, w),
                                        mode="bilinear", align_corners=False)[0, 0]
            sigma_l = level_sigmas[li] if li < len(level_sigmas) else level_sigmas[-1]
            sil_r, _, rast_r = _make_renderers(
                (h, w), sigma_l, device,
                faces_per_pixel=config.stage1_faces_per_px,
            )
            cams = _PerspCam(
                focal_length=((v["fx"] * sw, v["fy"] * sh),),
                principal_point=((cx_c * sw, cy_c * sh),),
                image_size=((h, w),),
                R=v["R_view"][None], T=v["T_view"][None],
                device=device, in_ndc=False,
            )
            ld = {
                "h": h, "w": w,
                "tgt_mask": tgt_mask_lb, "tgt_z": tgt_z_l,
                "sil_r": sil_r, "rast_r": rast_r, "world_cameras": cams,
                # raw (scaled, crop-shifted) intrinsics + extrinsics so the batched
                # multistart path can rebuild a batch-B camera (PerspectiveCameras
                # has no .extend() in this pytorch3d build).
                "cam_args": {
                    "fx": v["fx"] * sw, "fy": v["fy"] * sh,
                    "cx": cx_c * sw, "cy": cy_c * sh,
                    "h": h, "w": w, "R": v["R_view"], "T": v["T_view"],
                },
            }
            if use_dt:
                import scipy.ndimage as _ndi
                import numpy as _np
                tgt_np = (tgt_mask_lb > 0.5).cpu().numpy()
                if tgt_np.sum() == 0 or tgt_np.sum() == tgt_np.size:
                    dt_out_np = _np.zeros_like(tgt_np, dtype=_np.float32)
                    dt_in_np  = _np.zeros_like(tgt_np, dtype=_np.float32)
                else:
                    dt_out_np = _ndi.distance_transform_edt(~tgt_np).astype(_np.float32)
                    dt_in_np  = _ndi.distance_transform_edt( tgt_np).astype(_np.float32)
                norm = float(max(h, w))
                ld["dt_out"] = torch.from_numpy(dt_out_np / norm).to(device)
                ld["dt_in"]  = torch.from_numpy(dt_in_np  / norm).to(device)
            per_lvl.append(ld)
        view_levels.append(per_lvl)
    return view_levels


def _finalize_multiview(*, mesh_verts, mesh_faces, view_levels, norm_views,
                        R_final, t_final, s_frozen, plane_z, floor, use_floor,
                        freeze, learn_scale, init_scale, yaw, config, device,
                        verbose, debug_name):
    """Stage 2 (analytical area-ratio scale) + final full-res per-view IoU eval.

    Takes a converged (R_final, t_final, s_frozen) and returns the result dict
    exactly as optimize_multi_view used to. Shared by the single-init and the
    batched multi-start paths so scale/IoU accounting can't diverge.
    """
    from pytorch3d.structures import Meshes

    # ── Stage 2: area-ratio across ALL views (sum target / sum pred) ───
    if freeze:
        if verbose:
            print(f"\n  Stage 2 skipped (freeze_scale=True); s_final = {init_scale:.4f}")
        s_final = float(init_scale)
    elif learn_scale:
        if verbose:
            print(f"\n  Stage 2 skipped (learn_scale=True; Stage 1 jointly learned s); "
                  f"s_final = {s_frozen:.4f}")
        s_final = float(s_frozen)
    else:
        target_area_total = sum(float(v["mask"].bool().sum()) for v in norm_views)
        full_levels = [pl[-1] for pl in view_levels]
        if verbose:
            print(f"\n  Stage 2: area-ratio across {len(full_levels)} views "
                  f"(sum_target={int(target_area_total)} px)")
        s_curr = float(init_scale)
        with torch.no_grad():
            for it in range(8):
                t_for_iter = (torch.stack([t_final[0], t_final[1],
                              torch.tensor(plane_z - s_curr * float(floor["y_min"]),
                                           device=device, dtype=t_final.dtype)])
                              if use_floor else t_final)
                mesh_world = mesh_verts * s_curr @ R_final + t_for_iter
                meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                pred_area_total = 0.0
                for ld in full_levels:
                    alpha = ld["sil_r"](meshes, cameras=ld["world_cameras"])[..., 3].squeeze()
                    pred_area_total += float((alpha > 0.5).float().sum())
                if pred_area_total < 1.0:
                    if verbose:
                        print(f"    iter {it}: all views render 0 px, stopping.")
                    break
                ratio = (target_area_total / pred_area_total) ** 0.5
                s_new = s_curr * ratio
                if verbose:
                    print(f"    iter {it}: s {s_curr:.4f} → {s_new:.4f} "
                          f"(sum_pred={int(pred_area_total)}, "
                          f"sum_target={int(target_area_total)})")
                if abs(s_new - s_curr) / max(s_curr, 1e-8) < 5e-3:
                    s_curr = s_new
                    break
                s_curr = s_new
        s_final = s_curr
        # sync t_z if floor-locked
        if use_floor:
            t_final = torch.stack([t_final[0], t_final[1],
                      torch.tensor(plane_z - s_final * float(floor["y_min"]),
                                   device=device, dtype=t_final.dtype)]).detach()

    # ── Final per-view IoU at full res, binary ──────────────────────────
    per_view_iou = []
    with torch.no_grad():
        mesh_world = mesh_verts * s_final @ R_final + t_final
        meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
        for v_idx, per_lvl in enumerate(view_levels):
            ld = per_lvl[-1]
            alpha = ld["sil_r"](meshes, cameras=ld["world_cameras"])[..., 3].squeeze()
            pred_b = alpha > 0.5
            tgt_b = ld["tgt_mask"] > 0.5
            inter = (pred_b & tgt_b).sum().float()
            union = (pred_b | tgt_b).sum().float().clamp(min=1.0)
            per_view_iou.append({
                "name": norm_views[v_idx]["name"],
                "iou":  float((inter / union).item()),
            })

    mean_iou = sum(x["iou"] for x in per_view_iou) / len(per_view_iou)
    if verbose:
        print(f"\n  Final: scale={s_final:.4f}  "
              f"t=[{float(t_final[0]):+.3f}, {float(t_final[1]):+.3f}, {float(t_final[2]):+.3f}]")
        print(f"  Per-view IoU (mean={mean_iou:.4f}):")
        for x in per_view_iou:
            print(f"    {x['name']:>20}: {x['iou']:.4f}")

    if yaw:
        from ..viz.utils import assert_yaw_pure
        assert_yaw_pure(R_final, f"multi_view optimizer-out ({debug_name})")

    return {
        "R": R_final.detach().cpu(),
        "t": t_final.detach().cpu(),
        "scale": s_final,
        "per_view_iou": per_view_iou,
        "final_iou": mean_iou,
    }


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

                    # Soft-Tversky loss (α=β=1 → IoU): red/green 区域都有梯度.
                    # α=stage1_iou_fp_weight 罚 green-outside-red, β=fn_weight 罚
                    # red-not-covered. α>β ⇒ 把手方向被钉住 (绿溢出红会重罚),
                    # TP 分子保证不会塌 scale.
                    tp = (alpha * ld["tgt_mask"]).sum()
                    fp = (alpha * (1.0 - ld["tgt_mask"])).sum()
                    fn = ((1.0 - alpha) * ld["tgt_mask"]).sum()
                    soft_iou = tp / (tp + fp + fn + 1e-6)
                    _a_fp = float(getattr(self.config, "stage1_iou_fp_weight", 1.0))
                    _b_fn = float(getattr(self.config, "stage1_iou_fn_weight", 1.0))
                    tversky = tp / (tp + _a_fp * fp + _b_fn * fn + 1e-6)
                    iou_loss = (1.0 - tversky) * self.config.stage1_iou_weight

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
        from ..viz.utils import assert_yaw_pure
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

    # ───────────────────────────────────────────────────────────────────
    # Multi-view path (multi-cam × multi-frame, shared static-object pose)
    # ───────────────────────────────────────────────────────────────────
    def optimize_multi_view(
        self,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        views: List[Dict],
        plane_z: float,
        init_R: Optional[torch.Tensor] = None,
        init_t: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        floor_constraint: Optional[Dict] = None,
        yaw_only: Optional[bool] = None,
        freeze_scale: Optional[bool] = None,
        learn_scale: Optional[bool] = None,
        freeze_R: bool = False,
        rot_cams: Optional[List[str]] = None,
        crop_pad: float = 0.0,
        verbose: bool = True,
        debug_save_dir=None,
        debug_name: str = "obj",
    ) -> Dict:
        """Same shape as `optimize()`, but loss sums over multiple views.

        Each view is one (cam, frame) pair. Shared per-object pose (R, t, s)
        is jointly optimized against all of them. Per-view inputs:

            views[i] = {
                "name":   str,                       # e.g. "ego/000000"
                "mask":   (H, W) torch (binary or soft),
                "R_view": (3, 3) torch float,        # PT3D row-vec, from
                "T_view": (3,)   torch float,        # bundle.world_camera_at()
                "fx", "fy", "cx", "cy": float,
                "H", "W": int,
                "depth":  (H, W) torch float OR None  # PT3D cam-frame z (optional)
            }

        Stage 1 (Adam, 2-level pyramid): per iter, sum over views of
        soft-IoU loss + (if depth present and enough overlap) smooth-L1
        depth loss.

        Stage 2 (analytical scale, area-ratio): runs ONLY on the view with
        the largest target mask. Skipped when `freeze_scale=True`.

        World frame is shared across views (plane_z, floor_constraint).
        """
        if not views:
            raise ValueError("optimize_multi_view: views list is empty")

        mesh_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # Normalize views: move tensors to device, ensure shapes/types.
        norm_views = _normalize_views(views, self.device)

        _yaw = yaw_only if yaw_only is not None else self.config.yaw_only
        _freeze = freeze_scale if freeze_scale is not None else self.config.freeze_scale
        _use_dt = bool(getattr(self.config, "use_dt_loss", False))
        # learn_scale: jointly optimize R, t, s in Stage 1's Adam. s = exp(s_log)
        # for positivity. Removes the Stage 1 (frozen s) ↔ Stage 2 (analytical s)
        # decoupling that under-scales meshes when near and far cams disagree.
        # Stage 2 becomes a no-op polish (it'd just confirm the converged s).
        _learn_scale = (learn_scale if learn_scale is not None
                        else bool(getattr(self.config, "learn_scale", False)))
        if _learn_scale and _freeze:
            raise ValueError("learn_scale and freeze_scale are mutually exclusive")

        # rot_cams: R (rotation) gradient comes ONLY from views whose camera is in
        # this set; all OTHER views render with R.detach(), so they still constrain
        # t/scale but don't pull R. Use e.g. {"wrist"} when the close wrist resolves
        # yaw better than the ego silhouette but you still want ego+wrist on depth.
        _rot_cams = set(rot_cams) if rot_cams else None
        if _rot_cams is not None and verbose:
            print(f"  [rot_cams] R gradient ONLY from cams {sorted(_rot_cams)}; "
                  f"other views render with R.detach() (still drive t/scale).")

        if init_R is None:
            init_R = torch.eye(3, device=self.device)
        else:
            init_R = init_R.to(self.device)
        if init_t is None:
            init_t = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        else:
            init_t = init_t.to(self.device)

        if verbose:
            R_status = "freeze_R=True (only t/s optimized)" if freeze_R else f"yaw_only={_yaw}"
            loss_kind = "DT" if _use_dt else "soft-IoU"
            print(f"  [multi_view] {len(norm_views)} views, {R_status}, "
                  f"freeze_scale={_freeze}, init_scale={float(init_scale):.4f}, "
                  f"plane_z={plane_z:.3f}, loss={loss_kind}")
            for v in norm_views:
                d = "depth" if v["depth"] is not None else "mask-only"
                print(f"    - {v['name']:>16}  {v['W']}x{v['H']}  {d}  "
                      f"mask_px={int(v['mask'].bool().sum())}")

        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import PerspectiveCameras as _PerspCam

        # 2-level pyramid per view (coarse 128² → full-res), built by the shared
        # helper so the batched multistart path uses identical targets/cameras.
        view_levels = _build_view_levels(self.config, norm_views, self.device, _use_dt,
                                         crop_pad=crop_pad)

        n_levels = max(len(per_lvl) for per_lvl in view_levels)

        # ── Stage 1: parameterize R/t (+ optional yaw/floor/s), sum loss across views.
        s_frozen = float(init_scale)
        _floor = floor_constraint

        # Parameters
        if _yaw:
            theta_init = float(math.atan2(init_R[0, 1].item(), init_R[0, 0].item()))
            theta_p = torch.tensor(theta_init, device=self.device,
                                   dtype=torch.float32).clone()
            if not freeze_R:
                theta_p.requires_grad_(True)
            r6d_p = None
        else:
            r6d_p = rotation_matrix_to_6d(init_R).detach().clone()
            if not freeze_R:
                r6d_p.requires_grad_(True)
            theta_p = None

        use_floor = _floor is not None and _yaw
        if use_floor:
            tx_p = torch.tensor(float(init_t[0]), device=self.device,
                                dtype=torch.float32).clone().requires_grad_(True)
            ty_p = torch.tensor(float(init_t[1]), device=self.device,
                                dtype=torch.float32).clone().requires_grad_(True)
            t_p = None
        else:
            t_p = init_t.detach().clone().requires_grad_(True)
            tx_p = ty_p = None

        # learn_scale: s = exp(s_log). exp keeps s > 0; lr 5e-3 on s_log is a
        # multiplicative step on s — numerically stable across orders of magnitude.
        if _learn_scale:
            s_log = torch.tensor(math.log(max(s_frozen, 1e-6)),
                                  device=self.device,
                                  dtype=torch.float32).clone().requires_grad_(True)
        else:
            s_log = None

        def _get_s():
            if s_log is not None:
                return torch.exp(s_log)
            return s_frozen

        def _build_R():
            return _yaw_matrix_world(theta_p) if _yaw else rotation_6d_to_matrix(r6d_p)

        def _build_t(s_val):
            """s_val: float (s frozen) OR tensor (learn_scale, grad flows back through t_z)."""
            if use_floor:
                z_min = float(_floor["y_min"])
                if isinstance(s_val, torch.Tensor):
                    tz = plane_z - s_val * z_min            # preserves grad
                else:
                    tz = torch.tensor(plane_z - s_val * z_min,
                                      device=self.device, dtype=tx_p.dtype)
                return torch.stack([tx_p, ty_p, tz])
            return t_p

        history = {"depth": [], "iou": [], "total": []}

        for li in range(n_levels):
            if use_floor:
                params = [{"params": [tx_p, ty_p], "lr": self.config.stage1_lr_t}]
            else:
                params = [{"params": [t_p], "lr": self.config.stage1_lr_t}]
            if not freeze_R:
                if _yaw:
                    params.insert(0, {"params": [theta_p], "lr": self.config.stage1_lr_r6d})
                else:
                    params.insert(0, {"params": [r6d_p], "lr": self.config.stage1_lr_r6d})
            if s_log is not None:
                params.append({"params": [s_log], "lr": self.config.stage1_lr_scale})
            opt = Adam(params)
            iters = (self.config.stage1_iters_coarse if li == 0
                     else self.config.stage1_iters_fine)
            scale_tag = " +s" if _learn_scale else ""
            R_tag = " [R frozen]" if freeze_R else ""
            pbar = tqdm(range(iters),
                        desc=f"S1 L{li} (multi-view × {len(norm_views)}{scale_tag}){R_tag}",
                        disable=not verbose, leave=False)
            for _ in pbar:
                opt.zero_grad()
                R_c = _build_R()
                s_c = _get_s()
                t_c = _build_t(s_c)
                mesh_world = mesh_verts * s_c @ R_c + t_c
                meshes = Meshes(verts=[mesh_world], faces=[mesh_faces])
                # For non-rot_cams views: render with R detached so their loss flows
                # to t/scale but NOT to R (R is then resolved by rot_cams views only).
                meshes_rd = None
                if _rot_cams is not None:
                    mesh_world_rd = mesh_verts * s_c @ R_c.detach() + t_c
                    meshes_rd = Meshes(verts=[mesh_world_rd], faces=[mesh_faces])

                total_loss = torch.zeros((), device=self.device)
                iou_sum, depth_sum, n_with_depth = 0.0, 0.0, 0
                # Tversky asymmetry: α on FP (green outside red), β on FN (red
                # uncovered). α=β=1 ⇒ plain IoU. α>β ⇒ handle stays inside mask.
                a_fp = float(getattr(self.config, "stage1_iou_fp_weight", 1.0))
                b_fn = float(getattr(self.config, "stage1_iou_fn_weight", 1.0))
                for v_idx, per_lvl in enumerate(view_levels):
                    ld = per_lvl[li] if li < len(per_lvl) else per_lvl[-1]
                    w_view = norm_views[v_idx]["weight"]
                    cams = ld["world_cameras"]
                    _m = meshes
                    if meshes_rd is not None and norm_views[v_idx]["cam"] not in _rot_cams:
                        _m = meshes_rd     # R detached for this cam (t/scale only)
                    rendered = ld["sil_r"](_m, cameras=cams)
                    alpha = rendered[..., 3].squeeze()

                    # Soft confusion-matrix pixels (alpha=green ∈ [0,1], mask=red binary):
                    #   TP = green∩red, FP = green∖red (spills outside), FN = red∖green.
                    tp = (alpha * ld["tgt_mask"]).sum()
                    fp = (alpha * (1.0 - ld["tgt_mask"])).sum()
                    fn = ((1.0 - alpha) * ld["tgt_mask"]).sum()
                    soft_iou = tp / (tp + fp + fn + 1e-6)   # α=β=1, monitoring only
                    iou_sum += float(soft_iou)

                    view_loss = torch.zeros((), device=self.device)
                    if _use_dt:
                        # DT inside/outside-mean (balanced normalization so the
                        # global mean doesn't dilute the inside term):
                        tgt_b = ld["tgt_mask"] > 0.5
                        n_in  = tgt_b.float().sum().clamp(min=1.0)
                        n_out = (~tgt_b).float().sum().clamp(min=1.0)
                        loss_in  = ((1.0 - alpha) * ld["dt_in"] *
                                    tgt_b.float()).sum() / n_in        # recall (red uncovered)
                        loss_out = (alpha * ld["dt_out"] *
                                    (~tgt_b).float()).sum() / n_out     # precision (green outside)
                        w_out = float(getattr(self.config, "stage1_dt_w_out", 1.0))
                        w_in  = float(getattr(self.config, "stage1_dt_w_in", 1.0))
                        dt_loss = w_out * loss_out + w_in * loss_in
                        view_loss = view_loss + dt_loss * self.config.stage1_dt_weight
                        # Optional asymmetric-Tversky blend on top of DT.
                        blend = float(getattr(self.config, "stage1_dt_iou_blend", 0.0))
                        if blend > 0.0:
                            tversky = tp / (tp + a_fp * fp + b_fn * fn + 1e-6)
                            view_loss = view_loss + (1.0 - tversky) * blend
                    else:
                        # Asymmetric Tversky (α=β=1 → IoU). α>β pins handle
                        # direction; TP numerator prevents scale collapse.
                        tversky = tp / (tp + a_fp * fp + b_fn * fn + 1e-6)
                        view_loss = view_loss + (1.0 - tversky) * self.config.stage1_iou_weight

                    if ld["tgt_z"] is not None:
                        frag = ld["rast_r"](_m, cameras=cams)
                        rend_z = frag.zbuf[0, ..., 0]
                        overlap = (alpha > 0.5) & (ld["tgt_mask"] > 0.5) & (rend_z > 0)
                        if int(overlap.sum()) > 20:
                            depth_loss = F.smooth_l1_loss(
                                rend_z[overlap], ld["tgt_z"][overlap],
                                beta=self.config.stage1_depth_beta,
                            )
                            view_loss = view_loss + depth_loss
                            depth_sum += float(depth_loss)
                            n_with_depth += 1

                    total_loss = total_loss + w_view * view_loss

                total_loss.backward()
                opt.step()
                history["iou"].append(iou_sum / len(view_levels))
                history["depth"].append(depth_sum / max(n_with_depth, 1))
                history["total"].append(float(total_loss))
                if verbose:
                    postfix = {
                        "loss": f"{float(total_loss):.4f}",
                        "iou_avg": f"{iou_sum/len(view_levels):.3f}",
                    }
                    if s_log is not None:
                        postfix["s"] = f"{float(_get_s()):.4f}"
                    pbar.set_postfix(postfix)

        with torch.no_grad():
            R_final = _build_R().detach()
            if s_log is not None:
                s_frozen = float(_get_s().item())   # commit learned s for Stage 2 / final eval
            t_final = _build_t(s_frozen).detach()

        # ── Stage 2 (area-ratio scale) + final per-view IoU: shared helper ──
        res = _finalize_multiview(
            mesh_verts=mesh_verts, mesh_faces=mesh_faces, view_levels=view_levels,
            norm_views=norm_views, R_final=R_final, t_final=t_final,
            s_frozen=s_frozen, plane_z=plane_z, floor=_floor, use_floor=use_floor,
            freeze=_freeze, learn_scale=_learn_scale, init_scale=init_scale,
            yaw=_yaw, config=self.config, device=self.device,
            verbose=verbose, debug_name=debug_name,
        )
        res["history"] = history
        return res

    def optimize_multi_view_multistart(
        self,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        views: List[Dict],
        plane_z: float,
        init_yaws_deg: List[float],
        init_t: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        floor_constraint: Optional[Dict] = None,
        freeze_scale: Optional[bool] = None,
        learn_scale: Optional[bool] = None,
        rot_cams: Optional[List[str]] = None,
        keep_top: int = 0,
        crop_pad: float = 0.0,
        verbose: bool = True,
        debug_save_dir=None,
        debug_name: str = "obj",
    ) -> Dict:
        """Batched yaw multi-start: fit ALL yaw candidates in ONE rasterization.

        Equivalent to looping `optimize_multi_view` over `init_yaws_deg` and taking
        the argmax-IoU winner, but the B candidates are stacked into a single
        `Meshes(B)` batch so B silhouettes render in ONE kernel instead of B
        sequential ones — fills the (otherwise idle) GPU and removes B× the Python
        / Adam / mesh-load overhead. Candidate params (yaw θ, tx, ty, +scale) are
        DISJOINT per candidate, so the summed loss gives each candidate exactly its
        own gradient — mathematically identical to B independent optimizations.

        `keep_top`: after the coarse pyramid level, keep only the top-K candidates
        (by soft-IoU) into the full-res level — a bad yaw seed reveals itself at
        128², so refining all B at full res is wasted memory/compute. Set
        `keep_top >= len(init_yaws_deg)` (or 0) to disable pruning (faithful
        full-batch, useful for verifying equivalence vs the sequential path).

        Yaw-only (that's what multi-start is for). freeze_R has no meaning here.
        """
        if not views:
            raise ValueError("optimize_multi_view_multistart: views list is empty")
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import PerspectiveCameras as _PerspCam

        mesh_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)
        norm_views = _normalize_views(views, self.device)

        _freeze = freeze_scale if freeze_scale is not None else self.config.freeze_scale
        _use_dt = bool(getattr(self.config, "use_dt_loss", False))
        _learn_scale = (learn_scale if learn_scale is not None
                        else bool(getattr(self.config, "learn_scale", False)))
        if _learn_scale and _freeze:
            raise ValueError("learn_scale and freeze_scale are mutually exclusive")
        _rot_cams = set(rot_cams) if rot_cams else None

        if init_t is None:
            init_t = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        else:
            init_t = init_t.to(self.device)
        floor = floor_constraint
        use_floor = floor is not None

        view_levels = _build_view_levels(self.config, norm_views, self.device, _use_dt,
                                         crop_pad=crop_pad)
        n_levels = max(len(pl) for pl in view_levels)
        a_fp = float(getattr(self.config, "stage1_iou_fp_weight", 1.0))
        b_fn = float(getattr(self.config, "stage1_iou_fn_weight", 1.0))

        yaws0 = [float(y) for y in init_yaws_deg]
        if verbose:
            print(f"  [multistart] {len(yaws0)} yaw candidates {yaws0} over "
                  f"{len(norm_views)} views, keep_top={keep_top}, "
                  f"freeze_scale={_freeze}, learn_scale={_learn_scale}, "
                  f"init_scale={float(init_scale):.4f}, plane_z={plane_z:.3f}")
            if _rot_cams is not None:
                print(f"  [multistart] [rot_cams] R gradient ONLY from {sorted(_rot_cams)}")

        def _batch_render_loss(li, th_p, tx_p, ty_p, tz_p, sl_p, B):
            """Render all B candidates for pyramid level `li`; return (loss, iou_sum(B,))."""
            R_c = _yaw_matrix_world_batch(th_p)                       # (B,3,3)
            if sl_p is not None:
                s_c = torch.exp(sl_p)                                 # (B,)
            else:
                s_c = torch.full((B,), float(init_scale), device=self.device)
            if use_floor:
                tz = plane_z - s_c * float(floor["y_min"])            # (B,)
            else:
                tz = tz_p                                             # (B,)
            t_c = torch.stack([tx_p, ty_p, tz], dim=-1)              # (B,3)
            mw = (mesh_verts.unsqueeze(0) * s_c.view(B, 1, 1)) @ R_c + t_c.unsqueeze(1)  # (B,V,3)
            meshes = Meshes(verts=[mw[b] for b in range(B)],
                            faces=[mesh_faces for _ in range(B)])
            meshes_rd = None
            if _rot_cams is not None:
                mw_rd = (mesh_verts.unsqueeze(0) * s_c.view(B, 1, 1)) @ R_c.detach() + t_c.unsqueeze(1)
                meshes_rd = Meshes(verts=[mw_rd[b] for b in range(B)],
                                   faces=[mesh_faces for _ in range(B)])
            total = torch.zeros((), device=self.device)
            iou_sum = torch.zeros(B, device=self.device)
            for v_idx, per_lvl in enumerate(view_levels):
                ld = per_lvl[li] if li < len(per_lvl) else per_lvl[-1]
                w_view = norm_views[v_idx]["weight"]
                ca = ld["cam_args"]
                cams = _PerspCam(
                    focal_length=[(ca["fx"], ca["fy"])] * B,
                    principal_point=[(ca["cx"], ca["cy"])] * B,
                    image_size=[(ca["h"], ca["w"])] * B,
                    R=ca["R"].to(self.device).reshape(1, 3, 3).repeat(B, 1, 1),
                    T=ca["T"].to(self.device).reshape(1, 3).repeat(B, 1),
                    device=self.device, in_ndc=False,
                )
                _m = meshes
                if meshes_rd is not None and norm_views[v_idx]["cam"] not in _rot_cams:
                    _m = meshes_rd
                rendered = ld["sil_r"](_m, cameras=cams)             # (B,H,W,4)
                alpha = rendered[..., 3]                             # (B,H,W)
                tgt = ld["tgt_mask"]                                 # (H,W)
                tp = (alpha * tgt).sum(dim=(-2, -1))                 # (B,)
                fp = (alpha * (1.0 - tgt)).sum(dim=(-2, -1))
                fn = ((1.0 - alpha) * tgt).sum(dim=(-2, -1))
                iou_sum = iou_sum + (tp / (tp + fp + fn + 1e-6)).detach()
                if _use_dt:
                    tgt_b = tgt > 0.5
                    n_in = tgt_b.float().sum().clamp(min=1.0)
                    n_out = (~tgt_b).float().sum().clamp(min=1.0)
                    loss_in = ((1.0 - alpha) * ld["dt_in"] * tgt_b.float()).sum(dim=(-2, -1)) / n_in
                    loss_out = (alpha * ld["dt_out"] * (~tgt_b).float()).sum(dim=(-2, -1)) / n_out
                    w_out = float(getattr(self.config, "stage1_dt_w_out", 1.0))
                    w_in = float(getattr(self.config, "stage1_dt_w_in", 1.0))
                    view_loss = (w_out * loss_out + w_in * loss_in) * self.config.stage1_dt_weight
                    blend = float(getattr(self.config, "stage1_dt_iou_blend", 0.0))
                    if blend > 0.0:
                        tversky = tp / (tp + a_fp * fp + b_fn * fn + 1e-6)
                        view_loss = view_loss + (1.0 - tversky) * blend
                else:
                    tversky = tp / (tp + a_fp * fp + b_fn * fn + 1e-6)   # (B,)
                    view_loss = (1.0 - tversky) * self.config.stage1_iou_weight
                if ld["tgt_z"] is not None:
                    frag = ld["rast_r"](_m, cameras=cams)
                    rend_z = frag.zbuf[..., 0]                        # (B,H,W)
                    depth_terms = []
                    for b in range(B):
                        ov = (alpha[b] > 0.5) & (tgt > 0.5) & (rend_z[b] > 0)
                        if int(ov.sum()) > 20:
                            depth_terms.append(F.smooth_l1_loss(
                                rend_z[b][ov], ld["tgt_z"][ov],
                                beta=self.config.stage1_depth_beta))
                        else:
                            depth_terms.append(torch.zeros((), device=self.device))
                    view_loss = view_loss + torch.stack(depth_terms)
                total = total + w_view * view_loss.sum()
            return total, iou_sum

        # ── candidate leaf params (all start from the same init_t, different yaw) ──
        B = len(yaws0)
        th_p = torch.tensor([math.radians(y) for y in yaws0], device=self.device,
                            dtype=torch.float32).requires_grad_(True)
        tx_p = init_t[0].detach().float().repeat(B).clone().requires_grad_(True)
        ty_p = init_t[1].detach().float().repeat(B).clone().requires_grad_(True)
        tz_p = None if use_floor else init_t[2].detach().float().repeat(B).clone().requires_grad_(True)
        sl_p = (torch.full((B,), math.log(max(float(init_scale), 1e-6)),
                           device=self.device).requires_grad_(True)
                if _learn_scale else None)
        cand_yaw = list(yaws0)

        mean_iou = None
        for li in range(n_levels):
            params = [{"params": [th_p], "lr": self.config.stage1_lr_r6d},
                      {"params": [tx_p, ty_p], "lr": self.config.stage1_lr_t}]
            if tz_p is not None:
                params.append({"params": [tz_p], "lr": self.config.stage1_lr_t})
            if sl_p is not None:
                params.append({"params": [sl_p], "lr": self.config.stage1_lr_scale})
            opt = Adam(params)
            iters = (self.config.stage1_iters_coarse if li == 0
                     else self.config.stage1_iters_fine)
            pbar = tqdm(range(iters), desc=f"S1-batch L{li} (B={B})",
                        disable=not verbose, leave=False)
            for _ in pbar:
                opt.zero_grad()
                total, iou_cand = _batch_render_loss(li, th_p, tx_p, ty_p, tz_p, sl_p, B)
                total.backward()
                opt.step()
                if verbose:
                    pbar.set_postfix({"loss": f"{float(total):.3f}",
                                      "best_iou": f"{float(iou_cand.max())/len(view_levels):.3f}"})
            with torch.no_grad():
                _, iou_cand = _batch_render_loss(li, th_p, tx_p, ty_p, tz_p, sl_p, B)
            mean_iou = iou_cand / len(view_levels)                    # (B,)
            if verbose:
                order = sorted(range(B), key=lambda i: -float(mean_iou[i]))
                print(f"  [multistart L{li}] " +
                      ", ".join(f"{cand_yaw[i]:.0f}°→{float(mean_iou[i]):.3f}" for i in order))
            # prune to top-K before the (expensive, full-res) next level
            if li < n_levels - 1 and keep_top and 0 < int(keep_top) < B:
                keep = torch.topk(mean_iou, int(keep_top)).indices.tolist()
            else:
                keep = list(range(B))
            with torch.no_grad():
                th_n = th_p.detach()[keep].clone()
                tx_n = tx_p.detach()[keep].clone()
                ty_n = ty_p.detach()[keep].clone()
                tz_n = tz_p.detach()[keep].clone() if tz_p is not None else None
                sl_n = sl_p.detach()[keep].clone() if sl_p is not None else None
            th_p = th_n.requires_grad_(True)
            tx_p = tx_n.requires_grad_(True)
            ty_p = ty_n.requires_grad_(True)
            tz_p = tz_n.requires_grad_(True) if tz_n is not None else None
            sl_p = sl_n.requires_grad_(True) if sl_n is not None else None
            cand_yaw = [cand_yaw[k] for k in keep]
            B = len(keep)

        # ── pick winner + finalize (Stage 2 scale + full-res IoU) on it ──
        with torch.no_grad():
            _, iou_cand = _batch_render_loss(n_levels - 1, th_p, tx_p, ty_p, tz_p, sl_p, B)
            mean_iou = iou_cand / len(view_levels)
            w = int(torch.argmax(mean_iou).item())
            theta_w = th_p.detach()[w]
            R_final = _yaw_matrix_world(theta_w).detach()
            s_w = (float(torch.exp(sl_p.detach()[w]).item())
                   if sl_p is not None else float(init_scale))
            if use_floor:
                t_final = torch.stack([tx_p.detach()[w], ty_p.detach()[w],
                    torch.tensor(plane_z - s_w * float(floor["y_min"]),
                                 device=self.device, dtype=tx_p.dtype)])
            else:
                t_final = torch.stack([tx_p.detach()[w], ty_p.detach()[w], tz_p.detach()[w]])

        if verbose:
            print(f"\n  [multistart] winner yaw = {math.degrees(float(theta_w)) % 360:.1f}° "
                  f"(coarse/fine IoU {float(mean_iou[w]):.4f})")

        res = _finalize_multiview(
            mesh_verts=mesh_verts, mesh_faces=mesh_faces, view_levels=view_levels,
            norm_views=norm_views, R_final=R_final, t_final=t_final, s_frozen=s_w,
            plane_z=plane_z, floor=floor, use_floor=use_floor, freeze=_freeze,
            learn_scale=_learn_scale, init_scale=init_scale, yaw=True,
            config=self.config, device=self.device, verbose=verbose, debug_name=debug_name,
        )
        res["best_yaw_deg"] = round(math.degrees(float(theta_w)) % 360.0, 1)
        res["all_starts"] = [{"yaw_deg": round(float(cand_yaw[i]), 1),
                              "iou": float(mean_iou[i])} for i in range(B)]
        res["history"] = {"depth": [], "iou": [], "total": []}
        return res


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
