#!/usr/bin/env python3
"""step3e_align_meshes.py — ICP 把多视角 mesh (step2 出的) 对齐到 scene mesh (step3d 出的).

逻辑:
    step2 mesh_multi: SAM3D 用多视角扫描出的 mesh, canonical 朝向 A.
        我们要把它放到 scene 里, 但需要 R/t/s in scene-camera 系.
    step3d mesh_scene: SAM3D 用 scene.jpg 单视角出的 mesh, canonical 朝向 B,
        附带 pose (R_scene, t_scene, s_scene) 把它摆到 scene-camera 系.
    canonical_A 和 canonical_B 不一定一致 → 不能直接拿 (R_scene, t_scene, s_scene)
        当 mesh_multi 的 init pose.
    本脚本: 把 mesh_scene 用其 pose 摆到 scene-camera 系得到 mesh_scene_posed (3D point cloud),
        然后 ICP 把 mesh_multi 的 y-up canonical 顶点对齐到 mesh_scene_posed.
    输出: (R_final, t_final, s_final) — 直接用作 step3c 的 init.

依赖 (sam3d-objects env). 路径由 pipeline_paths 解析.
    paths.scene_sam3d_dir/<obj>/result.glb + params.npz   — step3d 出
    paths.object_mesh_link(name, prompt)                  — step2 出 (symlink)

输出:
    paths.init_pose_path(obj_name)                         — R/t/s npz
    paths.init_pose_dir/align_debug_<obj>.ply              — 对齐可视化

跑法:
    conda activate sam3d-objects
    python -m real2sim.pose.align_meshes
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # real2sim/pose/.. .. == R2SConfig/
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.io.scenes import OBJECTS
from real2sim.io.paths import resolve as resolve_paths, sanitize_filename, should_rebuild
from full_workflow import find_glb_files


# z-up → y-up rotation, same as MV-SAM3D get_mesh:113-119
M_PT3D = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]], dtype=np.float32)


def load_glb_verts_y_up(glb_path: Path) -> np.ndarray:
    """加载 GLB, 合并 sub-mesh, 返回 y-up 系下的 verts (N, 3)."""
    scene = trimesh.load(str(glb_path))
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene
    verts = np.asarray(combined.vertices, dtype=np.float32)
    # z-up → y-up: v_new = v @ M_PT3D.T (跟 load_glb_as_pytorch3d 一致)
    verts_y_up = verts @ M_PT3D.T
    return verts_y_up, combined


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3x3 matrix in PyTorch3D row-vector convention."""
    from pytorch3d.transforms import quaternion_to_matrix
    R = quaternion_to_matrix(torch.from_numpy(q.astype(np.float32))).numpy()
    return R


def downsample_points(pts: np.ndarray, max_pts: int = 20000, rng=None) -> np.ndarray:
    """随机 downsample 到 max_pts."""
    if len(pts) <= max_pts:
        return pts
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), max_pts, replace=False)
    return pts[idx]


def points_to_o3d_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def try_icp_with_init(
    src_pts: np.ndarray, tgt_pts: np.ndarray, init_T: np.ndarray,
    voxel: float = 0.01, max_iter: int = 200,
):
    """ICP point-to-point. init_T 是 4x4 transform (src → tgt).
    返回 (transform 4x4, fitness, rmse).
    """
    src_pcd = points_to_o3d_pcd(src_pts)
    tgt_pcd = points_to_o3d_pcd(tgt_pts)
    src_pcd = src_pcd.voxel_down_sample(voxel)
    tgt_pcd = tgt_pcd.voxel_down_sample(voxel)
    src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))
    tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))

    threshold = voxel * 5
    reg = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, threshold, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return np.asarray(reg.transformation), float(reg.fitness), float(reg.inlier_rmse)


def initial_pose_candidates_rigid(src_scaled_pts: np.ndarray, tgt_pts: np.ndarray):
    """生成多个 RIGID init transform (4x4) 尝试. src 已经预先 scale 过.

    对每个候选 R, 用 centroid 对齐计算 t: t = tgt_c - R @ src_c.
    """
    src_c = src_scaled_pts.mean(0)
    tgt_c = tgt_pts.mean(0)

    yaws_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    Rs = []
    for d in yaws_deg:
        a = np.deg2rad(d)
        c, s = np.cos(a), np.sin(a)
        Rs.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64))  # Ry(d)
    # 加 polarity flip: Rx(180) 让长轴翻转 (mug_tree 倒过来 case)
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    Rs_flipped = [flip @ R for R in Rs[:4]]
    Rs.extend(Rs_flipped)

    inits = []
    for R in Rs:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tgt_c - R @ src_c
        inits.append(T)
    return inits


def align_one(label: str, slug: str, multi_glb: Path, scene_dir: Path,
              out_file: Path):
    out_dir = out_file.parent
    print(f"\n[step3e] === {label} ({slug}) ===")
    print(f"  multi mesh: {multi_glb}")
    scene_glb = scene_dir / "result.glb"
    scene_params = scene_dir / "params.npz"
    if not scene_glb.exists() or not scene_params.exists():
        print(f"  [skip] missing {scene_glb} or {scene_params}, run step3d first")
        return None

    # 1. 加载 mesh_multi (y-up canonical)
    multi_verts, multi_tri = load_glb_verts_y_up(multi_glb)
    multi_verts = downsample_points(multi_verts, 30000)
    print(f"  multi verts: {len(multi_verts)}")

    # 2. 加载 mesh_scene 并 apply scene pose 到 scene-camera 系
    scene_verts, scene_tri = load_glb_verts_y_up(scene_glb)
    p = np.load(scene_params)
    R_scene = quat_wxyz_to_R(p["rotation"].flatten())  # (3,3) PT3D row-vec
    t_scene = p["translation"].flatten().astype(np.float32)
    s_scene = float(np.asarray(p["scale"]).mean())
    print(f"  scene pose: R quat={p['rotation'].flatten().tolist()}")
    print(f"              t={t_scene.tolist()}  s={s_scene:.4f}")

    # mesh_scene_posed = (scene_verts * s_scene) @ R_scene + t_scene
    scene_posed = (scene_verts * s_scene) @ R_scene + t_scene
    scene_posed = downsample_points(scene_posed, 30000)
    print(f"  scene posed verts: {len(scene_posed)}")

    # 3. 计算 scale, 预先 scale src, 然后 ICP 找 rigid (R, t)
    src_ext = (multi_verts.max(0) - multi_verts.min(0)).max()
    tgt_ext = (scene_posed.max(0) - scene_posed.min(0)).max()
    s_final = float(tgt_ext / max(src_ext, 1e-8))
    multi_scaled = multi_verts * s_final
    print(f"  pre-scale src by s={s_final:.4f}  (src_ext={src_ext:.3f} → tgt_ext={tgt_ext:.3f})")

    inits = initial_pose_candidates_rigid(multi_scaled, scene_posed)
    print(f"  trying {len(inits)} rigid init transforms for ICP")

    voxel = max(0.005, float(tgt_ext) * 0.01)
    print(f"  ICP voxel size: {voxel:.4f}m")

    best_T, best_fit, best_rmse = None, -1.0, float("inf")
    for ci, T_init in enumerate(inits):
        try:
            T, fit, rmse = try_icp_with_init(multi_scaled, scene_posed, T_init, voxel=voxel)
        except Exception as e:
            print(f"    cand {ci}: ICP failed: {e}")
            continue
        if fit > best_fit + 0.01 or (abs(fit - best_fit) < 0.01 and rmse < best_rmse):
            best_T, best_fit, best_rmse = T, fit, rmse
        print(f"    cand {ci}: fit={fit:.3f}  rmse={rmse:.4f}m")

    if best_T is None:
        print(f"  [fail] no successful ICP")
        return None

    # ICP T 是 column-vector: target_col = T @ src_col, i.e., target = R_math @ src + t.
    # 转 row-vector (PyTorch3D 用): target_row = src_row @ R_math.T + t.
    R_math = best_T[:3, :3]   # column-vector rotation
    # re-orthogonalize via SVD (ICP 已经返回 rigid, 但保险起见)
    U, _, Vt = np.linalg.svd(R_math)
    R_math = U @ Vt
    if np.linalg.det(R_math) < 0:
        R_math = -R_math
    R_final = R_math.T.astype(np.float32)            # ← 转 PyTorch3D row-vector
    t_final = best_T[:3, 3].astype(np.float32)
    print(f"  ICP best: fit={best_fit:.3f}  rmse={best_rmse:.4f}m")
    print(f"  → R (PT3D row-vec) = {R_final.tolist()}")
    print(f"  → t = {t_final.tolist()}")
    print(f"  → s = {s_final:.4f}")

    # 保存
    np.savez(out_file,
             R=R_final.astype(np.float32),
             t=t_final.astype(np.float32),
             scale=np.float32(s_final),
             icp_fitness=np.float32(best_fit),
             icp_rmse=np.float32(best_rmse))
    print(f"  saved → {out_file}")

    # 保存可视化点云: scene_posed (红) + multi_aligned (绿)
    # multi_aligned 用 PT3D row-vector 约定算: (multi_verts * s) @ R_final + t
    multi_aligned = (multi_verts * s_final) @ R_final + t_final
    out_ply = out_dir / f"align_debug_{slug}.ply"
    pcd_red = points_to_o3d_pcd(scene_posed)
    pcd_red.paint_uniform_color([1.0, 0.2, 0.2])
    pcd_grn = points_to_o3d_pcd(multi_aligned)
    pcd_grn.paint_uniform_color([0.2, 1.0, 0.2])
    combined_pcd = pcd_red + pcd_grn
    o3d.io.write_point_cloud(str(out_ply), combined_pcd)
    print(f"  saved align viz → {out_ply}")

    return out_file


def main():
    paths = resolve_paths()
    force = should_rebuild()
    out_dir = paths.init_pose_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 找每个物体的 multi-view mesh
    multi_paths = find_glb_files([obj["name"] for obj in OBJECTS], paths=paths)
    print(f"[step3e] multi-view mesh paths:")
    for n, p in multi_paths.items():
        print(f"  {n}: {p}")

    scene_root = paths.scene_sam3d_dir

    # 把 OBJECTS 跟 step3d 输出的子目录对上 (step3d 用 sanitize_filename(label))
    # step3d 的 label 来自 scene_masks.pt, 跟 scene_prompt 走的, 不一定是 OBJECTS 的 name.
    # 我们扫描 workdir/scene_sam3d/ 下所有子目录, 然后跟 OBJECTS 匹配.
    if not scene_root.exists():
        print(f"[fatal] {scene_root} 不存在, 先跑 step3d"); sys.exit(1)
    scene_sub = [d for d in scene_root.iterdir() if d.is_dir()]
    print(f"[step3e] scene_sam3d 子目录: {[d.name for d in scene_sub]}")

    name_to_prompt_slug = {obj["name"]: sanitize_filename(obj["prompt"]).lower() for obj in OBJECTS}

    # 跟 step3c 用同一个 best-score 匹配 (prompt_slug 完全相等 > name 完全相等 > 子串).
    # 之前 first-match 会让 sub='black_straight_mug_tree' 撞 blue_mug 的 prompt='mug',
    # 结果 init_pose_blue_mug.npz 实际是用 tree 的 scene-sam3d mesh 算的 ICP, 全错.
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

    matched = 0
    for sub in scene_sub:
        slug_lower = sub.name.lower()
        matched_obj = None
        best_rank = (0, 0)
        for obj in OBJECTS:
            name = obj["name"]
            name_l = name.lower()
            prompt_slug = name_to_prompt_slug.get(name, "")
            rank = _match_rank(slug_lower, name_l, prompt_slug)
            if rank > best_rank:
                best_rank = rank
                matched_obj = obj
        if matched_obj is None:
            print(f"[warn] scene sub '{sub.name}' 没匹配到 OBJECTS, skip")
            continue
        multi_glb = multi_paths.get(matched_obj["name"])
        if multi_glb is None:
            print(f"[warn] {matched_obj['name']} 没找到 multi-view .glb, skip")
            continue
        out_file = paths.init_pose_path(matched_obj["name"])
        if not force and out_file.exists():
            print(f"[step3e] {matched_obj['name']}: cache hit {out_file}; REBUILD=1 to force.")
            matched += 1
            continue
        # slug for output filename: 用 OBJECTS name 而不是 scene label slug (跟 step3c init 文件名一致)
        result = align_one(matched_obj["name"], matched_obj["name"], multi_glb, sub, out_file)
        if result:
            matched += 1

    print(f"\n[step3e] done. aligned {matched} objects. outputs in {out_dir}/init_pose_*.npz")


if __name__ == "__main__":
    main()
