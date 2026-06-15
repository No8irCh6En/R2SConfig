"""
End-to-end Real2Sim pipeline.

Image → Segmentation → Inpainting → Depth → Pose Optimization → Scene Export

Usage:
    from real2sim import Real2SimPipeline, PipelineConfig

    config = PipelineConfig()
    pipeline = Real2SimPipeline(config)
    results = pipeline.run("path/to/image.jpg", "red mug. blue cup.", mesh_paths=[...])
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict
import json

from .config import PipelineConfig
from .segmentation import GroundedSAM
from .inpainting import LaMaInpainter
from .depth import DepthEstimator
from .optimizer import PoseOptimizer
from .camera import intrinsics_from_hfov, build_camera_matrix
from .utils import ensure_output_dir, save_scene_json


class Real2SimPipeline:
    """Full Real2Sim asset pipeline.

    Given one or more images of a scene + text prompts for objects,
    produces optimized 3D scene parameters (poses, scales, camera).

    Assumes object meshes are provided externally (e.g., from SAM-3D or manual modeling).
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sam: Optional[GroundedSAM] = None
        self._inpainter: Optional[LaMaInpainter] = None
        self._depth: Optional[DepthEstimator] = None
        self._optimizer: Optional[PoseOptimizer] = None

    @property
    def sam(self) -> GroundedSAM:
        if self._sam is None:
            c = self.config.segmentation
            self._sam = GroundedSAM(
                dino_model=c.dino_model,
                sam_model=c.sam_model,
                box_threshold=c.box_threshold,
                text_threshold=c.text_threshold,
                device=c.device,
            )
        return self._sam

    @property
    def inpainter(self) -> LaMaInpainter:
        if self._inpainter is None:
            c = self.config.inpainting
            self._inpainter = LaMaInpainter(model_path=c.model_path, device=c.device, pad_to=c.pad_to)
        return self._inpainter

    @property
    def depth_estimator(self) -> DepthEstimator:
        if self._depth is None:
            c = self.config.depth
            self._depth = DepthEstimator(model_size=c.model_size, device=c.device)
        return self._depth

    @property
    def optimizer(self) -> PoseOptimizer:
        if self._optimizer is None:
            self._optimizer = PoseOptimizer(config=self.config.optimizer, device=str(self.device))
        return self._optimizer

    def run(
        self,
        image_path: str,
        text_prompt: str,
        mesh_paths: Optional[List[str]] = None,
        mesh_data: Optional[List[Dict]] = None,
        K: Optional[torch.Tensor] = None,
        output_dir: Optional[str] = None,
        precomputed_masks: Optional[List[Dict]] = None,
        metric_pointmap: Optional[torch.Tensor] = None,
        precomputed_inpainted: Optional[Image.Image] = None,
    ) -> Dict:
        """Execute the full pipeline.

        Args:
            image_path: Path to input RGB image
            text_prompt: Object descriptions, e.g. "red mug. wooden table."
            mesh_paths: List of .obj/.ply mesh file paths (one per object)
            mesh_data: List of dicts with 'verts', 'faces', 'name' keys (alternative to mesh_paths)
            K: (3, 3) camera intrinsics. Auto-derived from image size if None.
            output_dir: Override output directory
            precomputed_masks: 跳过 SAM3 分割，直接用外部 mask。每条:
                {"mask": (H,W) tensor/ndarray, "label": str,
                 optional "box": [x1,y1,x2,y2], optional "score": float}
                顺序需与 mesh_data 一致。提供此参数后 pipeline 不触发 self.sam，
                可以在没有 sam3 包的环境（如 sam3d-objects）中跑后半段。
            metric_pointmap: (H, W, 3) tensor，PyTorch3D 相机系 metric 坐标 (米)。
                提供后：(a) init_t 的 Z 从物体 mask 区域 pointmap[..., 2] 取中位数；
                (b) pointmap[..., 2] 作为 target_depth 喂给优化器（替代 DepthAnything）。

        Returns:
            Dict with camera, objects (pose/scale), masks, inpainted, depth keys
        """
        output_dir = output_dir or self.config.output_dir
        out = ensure_output_dir(output_dir)
        cfg = self.config

        # ── 1. Load image ──────────────────────────────────────────
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        print(f"[Pipeline] Loaded image: {W}x{H}")

        # ── 2. Camera intrinsics ────────────────────────────────────
        if K is None:
            fx, fy, cx, cy = intrinsics_from_hfov(W, H, cfg.camera.fov_deg)
            K = build_camera_matrix(fx, fy, cx, cy)
        else:
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()
        print(f"[Pipeline] Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        # ── 3. Segmentation ─────────────────────────────────────────
        if precomputed_masks is not None:
            from .segmentation import SegmentationResult

            print(f"[Pipeline] Using {len(precomputed_masks)} precomputed mask(s); "
                  f"skipping SAM3.")
            mask_tensors = []
            box_list = []
            score_list = []
            label_list = []
            for entry in precomputed_masks:
                m = entry["mask"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m)
                if m.dtype != torch.bool:
                    m = m > (0.5 if m.dtype.is_floating_point else 0)
                mask_tensors.append(m)
                label_list.append(entry.get("label", "object"))
                score_list.append(float(entry.get("score", 1.0)))
                if "box" in entry:
                    box_list.append(entry["box"])
                else:
                    ys, xs = torch.where(m)
                    if len(ys) > 0:
                        box_list.append([
                            xs.min().item(), ys.min().item(),
                            xs.max().item(), ys.max().item(),
                        ])
                    else:
                        box_list.append([0, 0, W, H])
            seg_result = SegmentationResult(
                masks=torch.stack(mask_tensors).to(self.device),
                boxes=torch.tensor(box_list, device=self.device, dtype=torch.float32),
                scores=torch.tensor(score_list, device=self.device, dtype=torch.float32),
                labels=label_list,
            )
        else:
            print(f"[Pipeline] Segmenting: '{text_prompt}'")
            seg_result = self.sam.segment(
                image,
                text_prompt,
                expand_mask_iters=cfg.segmentation.expand_mask_iters,
                expand_kernel=cfg.segmentation.expand_kernel,
            )
        N_obj = seg_result.masks.shape[0]
        print(f"[Pipeline] Found {N_obj} object(s)")

        if N_obj == 0:
            print("[Pipeline] No objects detected — skipping optimization.")
            return {"camera": K, "objects": [], "masks": [], "status": "no_detections"}

        # ── 4. Background inpainting ────────────────────────────────
        combined_mask = seg_result.masks.any(dim=0).to(self.device)
        if precomputed_inpainted is not None:
            print("[Pipeline] Using precomputed inpainted image; skipping LaMa.")
            inpainted = precomputed_inpainted.convert("RGB") if precomputed_inpainted.mode != "RGB" else precomputed_inpainted
        elif metric_pointmap is not None:
            # depth comes from pointmap; inpaint not actually needed
            print("[Pipeline] Skipping inpaint (metric_pointmap provided).")
            inpainted = image
        else:
            try:
                print("[Pipeline] Inpainting background...")
                inpainted = self.inpainter.inpaint(image, combined_mask, refine_iters=cfg.inpainting.refine_mask_iters)
            except RuntimeError as e:
                print(f"[Pipeline] Inpaint unavailable ({e}); using original image for depth.")
                inpainted = image

        if cfg.save_intermediate:
            inpainted.save(out / "inpainted.png")
            for i, m in enumerate(seg_result.masks):
                mask_img = Image.fromarray((m.cpu().numpy() * 255).astype(np.uint8))
                mask_img.save(out / f"mask_{i}.png")

        # ── 5. Depth estimation ─────────────────────────────────────
        if metric_pointmap is not None:
            if not isinstance(metric_pointmap, torch.Tensor):
                metric_pointmap = torch.as_tensor(metric_pointmap)
            metric_pointmap = metric_pointmap.to(self.device).float()
            assert metric_pointmap.shape[-1] == 3, \
                f"metric_pointmap must be (H,W,3), got {tuple(metric_pointmap.shape)}"
            # Use Z slice as target depth (metric meters)
            depth_map = metric_pointmap[..., 2]
            print(f"[Pipeline] Using metric pointmap (depth Z range "
                  f"[{depth_map.min().item():.3f}, {depth_map.max().item():.3f}] m)")
        else:
            print("[Pipeline] Estimating depth...")
            depth_map = self.depth_estimator.estimate(inpainted)

        if cfg.save_intermediate:
            depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            Image.fromarray((depth_vis.cpu().numpy() * 255).astype(np.uint8)).save(out / "depth.png")

        # ── 6. Prepare meshes ───────────────────────────────────────
        meshes = self._load_meshes(mesh_paths, mesh_data, seg_result, device=self.device)
        if not meshes:
            print("[Pipeline] No meshes provided — returning segmentation + depth only.")
            return {
                "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": W, "height": H},
                "objects": [],
                "masks": seg_result.masks.cpu(),
                "boxes": seg_result.boxes.cpu(),
                "inpainted": inpainted,
                "depth": depth_map.cpu(),
                "status": "no_meshes",
            }

        # ── 7. Pose optimization ────────────────────────────────────
        print(f"[Pipeline] Optimizing poses for {len(meshes)} object(s)...")
        target_rgb = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

        results = []
        for i, mesh in enumerate(meshes):
            obj_mask = seg_result.masks[i].to(self.device)

            # Estimate initial translation from mask centroid.
            # If metric_pointmap is provided, use the median Z over the object
            # region (metric, meters). Otherwise fall back to a fixed Z=2.0.
            if mesh.get("init_t") is None:
                ys, xs = torch.where(obj_mask)
                if len(ys) > 0:
                    cx_px = xs.float().mean()
                    cy_px = ys.float().mean()
                    if metric_pointmap is not None:
                        z_vals = metric_pointmap[ys, xs, 2]
                        z_est = z_vals.median().item()
                    else:
                        z_est = 2.0
                    # PyTorch3D 世界系: +X 朝屏幕左, +Y 朝上, +Z 朝深处。
                    # 像素系 +X 朝右、+Y 朝下，所以 X/Y 翻号。
                    X = -(cx_px - cx) * z_est / fx
                    Y = -(cy_px - cy) * z_est / fy
                    init_t_est = torch.tensor([X, Y, z_est], device=self.device)
                else:
                    init_t_est = None
            else:
                init_t_est = mesh["init_t"]

            # Closed-form init (世界系下从几何反推 tx, ty, scale).
            # 假设 yaw_only + floor_lock:
            #   tx, ty  ← mask centroid 反投影到 plane_z
            #   scale   ← 让 mesh canonical 高度投影后的像素高 = mask 像素高
            # 没有 world_camera_params 退回老启发式 (PT3D-cam-frame 路径).
            # mesh.skip_closed_form=True (来自 INIT_XY_<name> env hook) 会跳过闭式覆盖,
            # 保留外部传进来的 init_t. 此时 init_scale 走 mesh.init_scale (没设的话默认 1.0).
            if mesh.get("skip_closed_form", False):
                est_scale = mesh.get("init_scale", 1.0)
                _obj_name_dbg = mesh.get("name", f"object_{i}")
                print(f"  [closed-form {_obj_name_dbg}] SKIPPED (skip_closed_form=True), "
                      f"用外部 init_t={init_t_est.tolist()}, init_scale={est_scale}")
            elif mesh.get("init_scale", 1.0) == 1.0 and init_t_est is not None:
                wc = mesh.get("world_camera_params")
                _obj_name_dbg = mesh.get("name", f"object_{i}")
                if wc is not None:
                    from .scene_geometry import (
                        solve_init_pose_from_mask,
                        solve_init_pose_from_mask_pointmap,
                    )
                    verts = mesh["verts"]
                    mesh_z_extent = float((verts[:, 2].max() - verts[:, 2].min()).item())
                    plane_z = float(wc["plane_z"])
                    # 默认 mask-centroid backproject (最稳, 经多个场景验证).
                    # USE_POINTMAP_INIT=1 切到 pointmap 版 (实验/debug 用; MoGe 在
                    # 320x240 小图上 metric 不稳, 实测 mug/tree 都偏 1m+, 不推荐生产用).
                    import os as _os
                    if _os.environ.get("USE_POINTMAP_INIT", "0") == "1" and metric_pointmap is not None:
                        solved = solve_init_pose_from_mask_pointmap(
                            obj_mask, metric_pointmap, wc, plane_z, mesh_z_extent,
                            device=self.device,
                        )
                    else:
                        solved = solve_init_pose_from_mask(
                            obj_mask, wc, plane_z, mesh_z_extent, device=self.device,
                        )
                    if solved is not None:
                        tx_solved, ty_solved, est_scale_raw = solved
                        est_scale = max(0.1, min(est_scale_raw, 3.0))
                        # 用 closed-form 的 (tx, ty) 覆盖 init_t_est, tz 沿用 plane_z + 0.05
                        # (floor_lock 会自己用 plane_z - s · z_min 算出真正的 tz, init_t.z 不重要)
                        init_t_est = torch.tensor(
                            [tx_solved, ty_solved, plane_z + 0.05],
                            device=init_t_est.device, dtype=init_t_est.dtype,
                        )
                        print(f"  [closed-form {_obj_name_dbg}] "
                              f"tx={tx_solved:.3f}  ty={ty_solved:.3f}  "
                              f"mesh_z_extent={mesh_z_extent:.3f}  "
                              f"scale_raw={est_scale_raw:.3f} → {est_scale:.3f} "
                              f"(clamped [0.1, 3.0])")
                    else:
                        # solver 失败 (mask 空 / ray 不穿 plane), 退到 1.0
                        est_scale = 1.0
                        print(f"  [closed-form {_obj_name_dbg}] solver 失败, fallback init_scale=1.0")
                else:
                    # 老 PT3D-cam-frame 启发式
                    verts = mesh["verts"]
                    mesh_extent = float((verts.max(dim=0).values - verts.min(dim=0).values).max().item())
                    z_est = float(init_t_est[2].item())
                    mask_area = float(obj_mask.sum().item())
                    mask_size_px = 2.0 * (mask_area / 3.14159) ** 0.5
                    projected_size = mesh_extent * fx / max(z_est, 1e-6)
                    est_scale = max(0.1, min(mask_size_px / (projected_size + 1e-8), 3.0))
                    print(f"  [est_scale {_obj_name_dbg}] legacy heuristic, "
                          f"est_scale={est_scale:.3f}")
            else:
                est_scale = mesh.get("init_scale", 1.0)

            _obj_name = mesh.get("name", f"object_{i}")
            result = self.optimizer.optimize(
                mesh_verts=mesh["verts"],
                mesh_faces=mesh["faces"],
                target_mask=obj_mask,
                target_depth=depth_map.to(self.device),
                target_pointmap=metric_pointmap,
                target_rgb=target_rgb,
                K=K,
                init_R=mesh.get("init_R"),
                init_t=init_t_est,
                init_scale=est_scale,
                init_R_candidates=mesh.get("init_R_candidates"),
                vertex_colors=mesh.get("vertex_colors"),
                verbose=True,
                # ── World-frame mode 必备 (没传则 optimizer 内部 raise / 退化) ──
                world_camera_params=mesh.get("world_camera_params"),
                yaw_only=mesh.get("yaw_only"),
                floor_constraint=mesh.get("floor_constraint"),
                freeze_scale=mesh.get("freeze_scale"),
                debug_save_dir=mesh.get("debug_save_dir"),
                debug_name=_obj_name,
            )
            result["name"] = _obj_name
            # ── 出口 assert_yaw: 验 R 从 optimizer 出来还是纯 yaw ──
            from .utils import assert_yaw_pure
            if mesh.get("yaw_only"):
                assert_yaw_pure(result["R"], f"pipeline.py-after-optimize ({_obj_name})")

            # ── 多指标评估 + 可视化 ────────────────────────────────────
            try:
                metrics, vis_img = self._evaluate_and_visualize(
                    mesh_verts=mesh["verts"],
                    mesh_faces=mesh["faces"],
                    obj_mask=obj_mask,
                    metric_pointmap=metric_pointmap,
                    image=image,
                    K_pix=(fx, fy, cx, cy),
                    image_size=(H, W),
                    R=result["R"].to(self.device),
                    t=result["t"].to(self.device),
                    scale=result["scale"],
                    obj_name=result["name"],
                    world_camera_params=mesh.get("world_camera_params"),
                )
                result["metrics"] = metrics
                viz_base = Path(cfg.viz_output_dir) if cfg.viz_output_dir else out
                per_obj_dir = viz_base / "per_object"
                per_obj_dir.mkdir(parents=True, exist_ok=True)
                vis_img.save(per_obj_dir / f"{result['name']}_eval.png")
                print(f"  Metrics {result['name']}:")
                for k, v in metrics.items():
                    print(f"    {k:18s} = {v:.4f}" if isinstance(v, float) else f"    {k:18s} = {v}")
            except Exception as e:
                print(f"  [warn] eval/vis failed: {e}")

            results.append(result)

        # ── 8. Export ───────────────────────────────────────────────
        # 判断 R, t 在哪个 frame: 如果 mesh_data 里有 world_camera_params, 说明走的是
        # world-frame 路径, R/t 直接是 Genesis world 系下的; 否则是 PT3D-cam 系 (老路径).
        frame = "world" if any(
            obj.get("world_camera_params") is not None for obj in (mesh_data or [])
        ) else "pt3d_cam"
        export = {
            "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": W, "height": H},
            "objects": results,
            "image_path": str(image_path),
            "text_prompt": text_prompt,
            "status": "ok",
            "frame": frame,
        }
        save_scene_json(str(out / "scene.json"), export)

        # Save individual object transforms
        for obj in results:
            name = obj["name"]
            R = obj["R"]
            t = obj["t"]
            s = obj["scale"]
            # 4x4 transform matrix
            T = torch.eye(4)
            T[:3, :3] = R * s
            T[:3, 3] = t
            torch.save(T, out / f"{name}_transform.pt")

        print(f"[Pipeline] Done. Results saved to {out}/")
        return export

    def _evaluate_and_visualize(
        self,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        obj_mask: torch.Tensor,
        metric_pointmap: Optional[torch.Tensor],
        image,
        K_pix,
        image_size,
        R: torch.Tensor,
        t: torch.Tensor,
        scale: float,
        obj_name: str,
        world_camera_params: Optional[Dict] = None,
    ):
        """渲染最终 mesh, 算多指标, 输出 4-panel 可视化。

        指标 (都是 binary, threshold 0.5):
            iou             = |inter| / |union|
            recall          = |inter| / |mask|        (mesh 盖住 mask 多少)
            precision       = |inter| / |render|      (render 落在 mask 内的比例)
            bbox_iou        = bbox 级 IoU (对细长物体更友好)
            centroid_dist   = mask 中心 vs render 中心 像素距离
            depth_consist   = (rendered depth 与 pointmap z 的中位差) / median(z_mask)
                              低 = mesh 在正确 depth, 高 = mesh 太近/太远
        """
        from PIL import Image as PILImage, ImageDraw
        from pytorch3d.renderer import (
            PerspectiveCameras, MeshRasterizer, RasterizationSettings,
            SoftSilhouetteShader, MeshRenderer, BlendParams,
        )
        from pytorch3d.structures import Meshes as _Meshes

        fx, fy, cx, cy = K_pix
        H, W = image_size

        # binary silhouette + depth render
        raster = RasterizationSettings(
            image_size=(H, W), blur_radius=1e-6, faces_per_pixel=50,
            bin_size=0, max_faces_per_bin=200_000, perspective_correct=True,
        )
        blend = BlendParams(sigma=1e-4, gamma=1e-4)
        rasterizer = MeshRasterizer(raster_settings=raster)
        renderer = MeshRenderer(rasterizer=rasterizer, shader=SoftSilhouetteShader(blend_params=blend))

        # ── World-frame mode: 相机摆位用 world_camera_params (跟 optimizer / sim_c0 一致) ──
        # 没有 world_camera_params → 回退老 PT3D-cam-frame 路径 (R, t 当相机外参).
        if world_camera_params is not None:
            wc = world_camera_params
            R_view = wc["R_view"].to(self.device).float()
            T_view = wc["T_view"].to(self.device).float()
            # 内参用 world_camera_params 的 (跟 optimizer Stage 1 fine level 一样)
            wc_fx, wc_fy = wc["fx"], wc["fy"]
            wc_cx, wc_cy = wc["cx"], wc["cy"]
            cameras = PerspectiveCameras(
                focal_length=((wc_fx, wc_fy),), principal_point=((wc_cx, wc_cy),),
                image_size=((H, W),), R=R_view[None], T=T_view[None],
                device=self.device, in_ndc=False,
            )
            # mesh 在世界里摆: v_world = v_canonical * scale @ R + t
            mesh_world = mesh_verts.to(self.device) * scale @ R + t
            meshes = _Meshes(verts=[mesh_world], faces=[mesh_faces.to(self.device)])
        else:
            cameras = PerspectiveCameras(
                focal_length=((fx, fy),), principal_point=((cx, cy),),
                image_size=((H, W),), R=R[None], T=t[None],
                device=self.device, in_ndc=False,
            )
            meshes = _Meshes(verts=[mesh_verts.to(self.device) * scale], faces=[mesh_faces.to(self.device)])

        with torch.no_grad():
            sil_out = renderer(meshes, cameras=cameras)
            pred_soft = sil_out[..., 3].squeeze()
            pred = (pred_soft > 0.5).cpu().numpy()
            tgt = obj_mask.bool().cpu().numpy()

            # mesh depth (z in camera frame at每个像素 mesh 最近表面)
            # zbuf shape = (1, H, W, faces_per_pixel)，取 [..., 0] 是最近面的 z
            frag = rasterizer(meshes, cameras=cameras)
            zbuf_closest = frag.zbuf[0, ..., 0]   # (H, W)
            rendered_depth = zbuf_closest.where(zbuf_closest > 0,
                                                torch.full_like(zbuf_closest, float("nan")))

        # ── metrics ────────────────────────────────────────────────────
        inter = (pred & tgt).sum()
        union = max((pred | tgt).sum(), 1)
        mask_area = max(tgt.sum(), 1)
        pred_area = max(pred.sum(), 1)
        iou = inter / union
        recall = inter / mask_area
        precision = inter / pred_area

        # bbox IoU
        import numpy as np
        def bbox(m):
            if not m.any():
                return None
            ys, xs = np.where(m)
            return ys.min(), xs.min(), ys.max(), xs.max()
        bb_t = bbox(tgt)
        bb_p = bbox(pred)
        if bb_t is None or bb_p is None:
            bbox_iou = 0.0
        else:
            y0 = max(bb_t[0], bb_p[0]); x0 = max(bb_t[1], bb_p[1])
            y1 = min(bb_t[2], bb_p[2]); x1 = min(bb_t[3], bb_p[3])
            inter_b = max(0, y1 - y0) * max(0, x1 - x0)
            area_t = (bb_t[2] - bb_t[0]) * (bb_t[3] - bb_t[1])
            area_p = (bb_p[2] - bb_p[0]) * (bb_p[3] - bb_p[1])
            bbox_iou = inter_b / max(area_t + area_p - inter_b, 1)

        # centroid distance (像素)
        cy_t, cx_t = np.argwhere(tgt).mean(0) if tgt.any() else (0, 0)
        cy_p, cx_p = np.argwhere(pred).mean(0) if pred.any() else (0, 0)
        centroid_dist = float(np.hypot(cy_p - cy_t, cx_p - cx_t))

        # depth consistency: 在 mesh 投影区域里, rendered depth vs pointmap z 的中位偏差比例
        depth_consist = float("nan")
        if metric_pointmap is not None:
            pm_z = metric_pointmap[..., 2].cpu().numpy()
            rd = rendered_depth.cpu().numpy()
            valid = pred & ~np.isnan(rd) & (pm_z > 0)
            if valid.sum() > 10:
                diff = np.abs(rd[valid] - pm_z[valid])
                z_ref = max(np.median(pm_z[tgt]) if tgt.any() else 1.0, 1e-3)
                depth_consist = float(np.median(diff) / z_ref)

        metrics = {
            "iou": float(iou),
            "recall": float(recall),
            "precision": float(precision),
            "bbox_iou": float(bbox_iou),
            "centroid_dist_px": centroid_dist,
            "depth_consist_rel": depth_consist,
            "scale": float(scale),
            "t_x": float(t[0]), "t_y": float(t[1]), "t_z": float(t[2]),
        }

        # ── 4-panel 可视化 ─────────────────────────────────────────────
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO

        img_np = np.array(image)
        # 1) scene + mask (red)
        # 2) scene + pred render (green)
        # 3) scene + overlap (yellow = inter, red = mask only, green = render only)
        # 4) scene + mesh wireframe edges (just outlines)
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for ax in axes:
            ax.axis("off")
        axes[0].imshow(img_np)
        axes[0].imshow(tgt, cmap="Reds", alpha=0.4)
        axes[0].set_title(f"scene + target mask\n{tgt.sum()} px")

        axes[1].imshow(img_np)
        axes[1].imshow(pred, cmap="Greens", alpha=0.5)
        axes[1].set_title(f"scene + render\n{pred.sum()} px")

        # 3-color overlap
        combo = np.zeros((H, W, 3))
        only_mask = tgt & ~pred
        only_pred = pred & ~tgt
        both = tgt & pred
        combo[only_mask] = [1.0, 0.2, 0.2]
        combo[only_pred] = [0.2, 1.0, 0.2]
        combo[both]      = [1.0, 1.0, 0.0]
        axes[2].imshow(img_np)
        axes[2].imshow(combo, alpha=0.55)
        axes[2].set_title(f"overlap (yellow=hit, red=miss, green=spill)")

        # metric text panel
        axes[3].imshow(img_np, alpha=0.3)
        msg = (
            f"{obj_name}\n\n"
            f"IoU         = {metrics['iou']:.4f}\n"
            f"Recall      = {metrics['recall']:.4f}   (mesh covers mask)\n"
            f"Precision   = {metrics['precision']:.4f}   (render within mask)\n"
            f"bbox IoU    = {metrics['bbox_iou']:.4f}   (lenient for thin objs)\n"
            f"centroid    = {metrics['centroid_dist_px']:.1f} px\n"
            f"depth rel   = {metrics['depth_consist_rel']:.4f}\n\n"
            f"scale = {metrics['scale']:.4f}\n"
            f"t = ({metrics['t_x']:.3f}, {metrics['t_y']:.3f}, {metrics['t_z']:.3f})"
        )
        axes[3].text(20, 30, msg, fontsize=11, family="monospace", va="top",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        axes[3].set_title("metrics summary")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        vis_img = PILImage.open(buf).convert("RGB").copy()
        buf.close()
        return metrics, vis_img

    def _load_meshes(
        self,
        mesh_paths: Optional[List[str]],
        mesh_data: Optional[List[Dict]],
        seg_result,
        device: torch.device,
    ) -> List[Dict]:
        """Load meshes from .obj/.ply files or pre-loaded data."""
        meshes = []

        if mesh_data:
            for i, m in enumerate(mesh_data):
                entry = {
                    "verts": m["verts"].to(device),
                    "faces": m["faces"].to(device),
                    "name": m.get("name", f"object_{i}"),
                    "vertex_colors": m.get("vertex_colors"),
                    "init_R": m.get("init_R"),
                    "init_t": m.get("init_t"),
                    "init_scale": m.get("init_scale", 1.0),
                    "init_R_candidates": m.get("init_R_candidates"),
                    # World-frame mode plumbing: 这些必须保留, 否则 optimizer 走默认 6D rotation,
                    # yaw_only / floor_constraint / world_camera_params 全失效 (2026-06 修).
                    "world_camera_params": m.get("world_camera_params"),
                    "yaw_only":           m.get("yaw_only"),
                    "floor_constraint":   m.get("floor_constraint"),
                    "freeze_scale":       m.get("freeze_scale"),
                    "debug_save_dir":     m.get("debug_save_dir"),
                }
                if entry["vertex_colors"] is not None:
                    entry["vertex_colors"] = entry["vertex_colors"].to(device)
                meshes.append(entry)
            return meshes

        if mesh_paths:
            for i, path in enumerate(mesh_paths):
                verts, faces, vc = load_mesh_file(path, device)
                meshes.append({
                    "verts": verts,
                    "faces": faces,
                    "name": Path(path).stem,
                    "vertex_colors": vc,
                })
            return meshes

        return []


def load_mesh_file(
    path: str, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load a mesh from .obj or .ply, returning verts, faces, optional vertex_colors."""
    from pytorch3d.io import load_objs_as_meshes, load_ply
    from pytorch3d.structures import Meshes

    path = Path(path)
    if path.suffix.lower() == ".ply":
        verts, faces = load_ply(path)
        vc = None
    else:
        mesh = load_objs_as_meshes([str(path)], device=device)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        vc = mesh.textures.verts_features_list()[0] if mesh.textures is not None else None

    return verts.to(device), faces.to(device), vc.to(device) if vc is not None else None
