"""Mesh I/O helpers shared across pipeline steps.

- find_glb_files / link_object_mesh: 根据 OBJECTS + pipeline_paths 找 / 维护 result.glb 的 symlink
- load_glb_as_pytorch3d: trimesh load + open3d 简化 + M_LOAD (z-up → y-up)

提取自 full_workflow.py (2026-06), 保持原行为. step3c/step3e/step4 都用.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


# R2SConfig/MV-SAM3D/visualization/ — SAM3D 写死的输出根, find/link 都从这里 glob
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # real2sim/perception/.. .. == R2SConfig/
VISUALIZATION_DIR = PROJECT_ROOT / "MV-SAM3D" / "visualization"


def find_glb_files(object_names: List[str], paths=None) -> Dict[str, Path]:
    """Per-object mesh paths under new build layout.

    Order of resolution per name:
      1. paths.object_mesh_link(name, prompt) — symlink we maintain (preferred)
      2. Newest result.glb under MV-SAM3D/visualization/<name>__<hash>/<slug>/ — written by step2
      3. Newest result.glb under MV-SAM3D/visualization/<name>*/.../result.glb — legacy
    """
    from ..io.scenes import OBJECTS
    from ..io.paths import resolve as resolve_paths

    if paths is None:
        paths = resolve_paths()
    name_to_prompt = {o["name"]: o["prompt"] for o in OBJECTS}
    found: Dict[str, Path] = {}
    for name in object_names:
        prompt = name_to_prompt.get(name)
        if prompt is not None:
            link = paths.object_mesh_link(name, prompt)
            if link.exists():
                found[name] = link.resolve()
                continue
        if prompt is not None:
            hashed_basename = paths.build_object_dir(name, prompt).name
            cands = list(VISUALIZATION_DIR.glob(f"{hashed_basename}/**/result.glb"))
            if cands:
                found[name] = max(cands, key=lambda p: p.stat().st_mtime)
                continue
        cands = list(VISUALIZATION_DIR.glob(f"{name}*/**/result.glb"))
        if cands:
            found[name] = max(cands, key=lambda p: p.stat().st_mtime)
            continue
        print(f"  [WARN] No .glb found for '{name}'.")
    return found


def link_object_mesh(paths) -> None:
    """After step2, populate build/objects/<obj>__<hash>/mesh.glb as a symlink to the
    latest MV-SAM3D result.glb. Idempotent. Used by fast_start.sh after step2.
    """
    from ..io.scenes import OBJECTS

    for obj in OBJECTS:
        name = obj["name"]
        prompt = obj["prompt"]
        hashed_basename = paths.build_object_dir(name, prompt).name
        cands = list(VISUALIZATION_DIR.glob(f"{hashed_basename}/**/result.glb"))
        if not cands:
            cands = list(VISUALIZATION_DIR.glob(f"{name}*/**/result.glb"))
        if not cands:
            print(f"[link_object_mesh] [{name}] no result.glb found under {VISUALIZATION_DIR}")
            continue
        latest = max(cands, key=lambda p: p.stat().st_mtime)
        link = paths.object_mesh_link(name, prompt)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(latest)
        print(f"[link_object_mesh] {name}: {link} → {latest}")


def load_glb_as_pytorch3d(path: Path, device, target_triangles: int = 5000):
    """加载 MV-SAM3D 的 result.glb 并转换到 PyTorch3D 世界系.

    trimesh 默认 y-up (X right, Z out); PyTorch3D 是 y-up (X **left**, Z **inwards**).
    跟 MV-SAM3D 自己 layout_post_optimization_utils.py:113-119 的 get_mesh 一致:
        verts @ [[1,0,0],[0,0,-1],[0,1,0]].T

    target_triangles: mesh 简化目标三角形数. MV-SAM3D 自己也简化到 5000. None = 不简化.

    环境变量 SKIP_M_LOAD=1: 完全跳过 M_LOAD (调试用, 配合 step5 的 file_meshes_are_zup=True
    让 PT3D 和 Genesis 两边都把 .glb 当原样处理).
    """
    import os
    import torch
    import numpy as np

    path = Path(path)
    # ── disk cache of the expensive open3d decimation ────────────────────
    # The source .glb can be >1M faces; decimating to `target_triangles` takes
    # ~15-40s and is deterministic, yet this ran on EVERY invocation (step4 is
    # called 3×/episode). Cache the decimated verts/faces (pre-M_LOAD) next to
    # the .glb, keyed on the source mtime, so the sweep pays it once.
    # NO_MESH_CACHE=1 to bypass.
    cache = None
    if target_triangles is not None and os.environ.get("NO_MESH_CACHE", "0") != "1":
        cache = path.with_name(path.name + f".simp{target_triangles}.npz")
    verts_np = faces_np = None
    if cache is not None and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            d = np.load(cache)
            verts_np, faces_np = d["verts"], d["faces"]
            print(f"     mesh cache hit: {len(faces_np)} faces (skipped decimation)")
        except Exception:
            verts_np = faces_np = None

    if verts_np is None:
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
                print(f"     [warn] simplify 失败 ({e}), 用原 mesh {raw_faces} faces")
                verts_np = np.asarray(combined.vertices)
                faces_np = np.asarray(combined.faces)
        else:
            verts_np = np.asarray(combined.vertices)
            faces_np = np.asarray(combined.faces)

        if cache is not None:                    # write-then-atomic-rename (race-safe)
            try:
                # NB: np.savez appends ".npz" unless the name already ends in it,
                # so keep the temp name ending in ".npz" or the rename source vanishes.
                tmp = cache.with_name(cache.name + f".tmp{os.getpid()}.npz")
                np.savez(tmp, verts=np.asarray(verts_np, dtype=np.float32),
                         faces=np.asarray(faces_np, dtype=np.int64))
                os.replace(tmp, cache)
            except Exception as e:
                print(f"     [warn] mesh cache write failed ({e})")

    verts = torch.tensor(np.asarray(verts_np), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(faces_np), dtype=torch.int64, device=device)

    if os.environ.get("SKIP_M_LOAD", "0") == "1":
        # 不做 M_LOAD, mesh 顶点保持 .glb 文件里原样. 配合 step5 的
        # file_meshes_are_zup=True 一起用, 让 PT3D 跟 Genesis 都"按 .glb 原样"处理.
        print(f"     [SKIP_M_LOAD] 不应用 M_LOAD, mesh 保持 .glb 原始朝向")
    else:
        # z-up → y-up (M_LOAD). 跟 MV-SAM3D layout_post_optimization_utils.get_mesh:113-119:
        #   mesh @ [[1,0,0],[0,0,-1],[0,1,0]].T   ("from z-up to y-up")
        M_PT3D = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32, device=device,
        )
        verts = verts @ M_PT3D.T
    vc = None
    return verts, faces, vc
