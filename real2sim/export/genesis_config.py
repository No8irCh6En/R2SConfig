#!/usr/bin/env python
"""Step 5: 把 scene.json 转成 Genesis 可吃的 config (gsrl_config.json).

scene.json 里 (R, t, s) 在 PT3D scene-camera frame; raw .glb 是 trimesh y-up gltf.
PT3D 侧 load_glb_as_pytorch3d 显式套了 M_LOAD = [[1,0,0],[0,0,-1],[0,1,0]] (col-vec)
把 raw → PT3D world (full_workflow.py:506-508). 优化 chain (col-vec):
    V_view_pt3d = s * R^T @ M_LOAD @ V_raw + t

Genesis world: z-up, +X 出屏, +Y 左 (右手). camera 走 gluLookAt:
    R_wc cols = (cam_x, cam_y, cam_z),  cam_z = pos - lookat (back direction)
    M_FLIP = diag(-1, 1, -1)   # PT3D cam (X-left, Z-fwd) ↔ Genesis cam (X-right, Z-back)

⚠️ Genesis 加载 .glb 时**自动**把 mesh 从 Y-up 转成 Z-up
   (genesis/engine/mesh.py:75 用的矩阵 = 我们 M_LOAD).
   所以 Genesis 拿到的不是 V_raw, 是 M_LOAD @ V_raw.
   spawn_quat 应作用在 "M_LOAD 过的 mesh" 上, 公式里**不能再带 M_LOAD**:

    R_genesis = R_wc @ M_FLIP @ R^T                 # col-vec, 不带 trailing M_LOAD
    spawn_pos = cam_pos + R_wc @ M_FLIP @ t         # world position
    scale     = s                                    # 直接传 morph_kwargs.scale

自动 z-offset: 计算每个物体在 world 中的 mesh bbox 最低点, 如果 min(z) < margin 就
整体 (objects + camera + robot) 抬到 z=margin 以上, 避免穿地.

输入/输出由 pipeline_paths 解析:
    scene.json   ← paths.scene_json_path  (step3c 在 build_prompt_dir 下写的, 跨 run 共享)
    template     ← example/1.json
    mesh per obj ← paths.object_mesh_link(name, prompt)  (step2 symlink)
    output       ← paths.gsrl_config_path  (run_dir/gsrl_config.json, per-run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # real2sim/export/.. .. == R2SConfig/
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.io.scenes import OBJECTS
from real2sim.io.paths import resolve as resolve_paths


def resolve_mesh(obj_name: str) -> Path:
    """Resolve the mesh path for `obj_name` using pipeline_paths."""
    paths = resolve_paths()
    prompt = next((o["prompt"] for o in OBJECTS if o["name"] == obj_name), None)
    if prompt is None:
        raise ValueError(f"unknown obj '{obj_name}' (not in OBJECTS)")
    link = paths.object_mesh_link(obj_name, prompt)
    if link.exists():
        return link.resolve()
    # fallback to MV-SAM3D visualization glob
    vis = PROJECT_ROOT / "MV-SAM3D" / "visualization"
    hashed = paths.build_object_dir(obj_name, prompt).name
    cands = list(vis.glob(f"{hashed}/**/result.glb"))
    if not cands:
        cands = list(vis.glob(f"{obj_name}*/**/result.glb"))
    if not cands:
        raise FileNotFoundError(f"no result.glb for {obj_name} under {vis}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def compose_pose(R_rowvec, t):
    """World-frame: R, t 已经是 Genesis world 系下的 mesh→world. 直接转 Genesis 期望的格式.

    Input (step3c 在 world frame 模式下输出):
        R_rowvec (3x3): mesh→world 的 PT3D row-vec convention 旋转, 即 v_world = v_mesh @ R.
        t (3,): mesh 在 world 里的位置 (Genesis z-up coords).
    Output (Genesis 期望):
        spawn_pos (3,): 就是 t.
        R_genesis (3x3): col-vec form (v_world_col = R_genesis @ v_mesh_col), 即 R^T.
    spawn_quat 从 R_genesis 算 (后面 quat_wxyz 函数做).
    """
    R = np.asarray(R_rowvec, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R_genesis = R.T.copy()
    # ── assert_yaw: 验 step5 拿到的 R (从 scene.json) 跟 compose 后的 R_genesis 是不是纯 yaw ──
    # yaw_z 矩阵的转置仍是 yaw_z (绕同轴反方向), 所以如果 R 是纯 yaw, R_genesis 也应该是.
    from real2sim.viz.utils import assert_yaw_pure
    assert_yaw_pure(R,         "step5.compose_pose IN  (R_rowvec from scene.json)")
    assert_yaw_pure(R_genesis, "step5.compose_pose OUT (R_genesis = R.T)")
    return t.copy(), R_genesis


def quat_wxyz(R: np.ndarray) -> list:
    xyzw = Rscipy.from_matrix(R).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def mesh_bbox_in_world(mesh_path: Path, R_genesis: np.ndarray, spawn_pos: np.ndarray, scale: float):
    """返回 mesh 在 world 系下经 R/spawn_pos 变换后的 (z_min, z_max).

    Genesis 默认加载 .glb 时自动应用 M_LOAD (y-up → z-up). 这里要跟它一致, 否则
    bbox 算的是 raw .glb 的 z 分量 (y-up 系下的"深度"方向), 跟 Genesis 实际渲出来
    的"垂直方向"对不上 → z_offset 抬升用错了 z_min, 整体会被错误地抬几厘米.
    SKIP_M_LOAD=1 时跟 mesh_io 一样跳过.
    """
    import os as _os
    import trimesh
    m = trimesh.load(str(mesh_path), force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)

    # 应用 M_LOAD (跟 mesh_io.load_glb_as_pytorch3d / Genesis 默认一致)
    if _os.environ.get("SKIP_M_LOAD", "0") != "1":
        # M_PT3D = [[1,0,0],[0,0,-1],[0,1,0]];  v_loaded = v_raw @ M_PT3D.T
        M_PT3D_T = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        v = v @ M_PT3D_T

    vmin, vmax = v.min(0), v.max(0)
    corners = np.array([[x, y, z] for x in (vmin[0], vmax[0])
                                    for y in (vmin[1], vmax[1])
                                    for z in (vmin[2], vmax[2])])
    corners_world = (R_genesis @ (corners * scale).T).T + spawn_pos
    return float(corners_world[:, 2].min()), float(corners_world[:, 2].max())


def patch_object(env_cfg, key, mesh_path: Path, spawn_pos, spawn_quat, scale):
    import os as _os
    obj = env_cfg.setdefault(key, {})
    obj["asset_path"] = str(mesh_path)
    obj["gs_path"] = None
    obj["spawn_pos"] = [float(x) for x in spawn_pos]
    obj["spawn_quat"] = spawn_quat
    obj.setdefault("entity_kwargs", {}).setdefault("morph_kwargs", {})
    morph = obj["entity_kwargs"]["morph_kwargs"]
    morph["file"] = str(mesh_path)
    morph["scale"] = float(scale)
    # SKIP_M_LOAD=1: 告诉 Genesis "这文件已经是 z-up, 别再 M_LOAD 一次了". 必须配合
    # PT3D 那边 load_glb_as_pytorch3d 也跳过 M_LOAD, 两边才一致.
    if _os.environ.get("SKIP_M_LOAD", "0") == "1":
        morph["file_meshes_are_zup"] = True
        print(f"     [SKIP_M_LOAD] {key}: morph.file_meshes_are_zup=True")


def main():
    paths = resolve_paths()

    p = argparse.ArgumentParser()
    p.add_argument("--scene-json", default=None,
                   help="Default: paths.scene_json_path (= build_prompt_dir/scene.json)")
    p.add_argument("--template",   default=str(PROJECT_ROOT / "example/1.json"))
    p.add_argument("--output",     default=None,
                   help="Default: paths.gsrl_config_path (= run_dir/gsrl_config.json)")
    # object → GSRL template slot mapping comes from pipeline_config.OBJECTS
    # (each entry's "slot" field). No CLI knobs to keep step5 in sync with
    # whatever scene was actually run.
    # 但 mesh 可以 override: --mesh name=path (可重复). 用于 step3c 用了 tripo mesh,
    # 但 step5 默认 resolve_mesh 拿的是 sam3d 输出, 这俩 metric 单位差很多, 必须保持一致.
    p.add_argument("--mesh", action="append", default=[], metavar="NAME=PATH",
                   help="覆盖某个 object 的 mesh: --mesh red_mug=example/mug_1_tripo.glb. 可重复.")
    p.add_argument("--camera-pos",    type=float, nargs=3, default=None)
    p.add_argument("--camera-lookat", type=float, nargs=3, default=None)
    p.add_argument("--camera-up",     type=float, nargs=3, default=None,
                   help="只是信息 — Genesis single 模式相机 up 内部写死 (0,0,1).")
    p.add_argument("--no-z-offset", action="store_true",
                   help="不自动 lift 物体到地面以上 (默认 lift).")
    p.add_argument("--z-floor-margin", type=float, default=0.005,
                   help="lift 时最低物体到 z=margin (默认 5mm).")
    args = p.parse_args()

    scene_json_path = Path(args.scene_json) if args.scene_json else paths.scene_json_path
    output_path     = Path(args.output)     if args.output     else paths.gsrl_config_path

    # 解析 --mesh name=path
    mesh_overrides: dict[str, Path] = {}
    for spec in args.mesh:
        if "=" not in spec:
            print(f"[step5] [fatal] --mesh 格式: name=path, 收到 {spec!r}"); sys.exit(2)
        nm, pth = spec.split("=", 1)
        p = Path(pth)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            print(f"[step5] [fatal] --mesh {nm} 指向的文件不存在: {p}"); sys.exit(2)
        mesh_overrides[nm] = p
        print(f"[step5] [override] mesh[{nm}] = {p}")

    if not scene_json_path.exists():
        print(f"[step5] [fatal] scene.json 不存在: {scene_json_path}")
        print(f"        先跑 step3c 生成 scene.json.")
        sys.exit(1)

    scene = json.loads(scene_json_path.read_text())
    tpl   = json.loads(Path(args.template).read_text())
    env   = tpl["env"]

    cam_cfg = env.get("camera", {})
    cam_pos    = np.array(args.camera_pos    if args.camera_pos    is not None else cam_cfg["single_pos"],    dtype=np.float64)
    cam_lookat = np.array(args.camera_lookat if args.camera_lookat is not None else cam_cfg["single_lookat"], dtype=np.float64)
    cam_up_used = np.array([0.0, 0.0, 1.0])  # Genesis hard-codes single-mode up
    if args.camera_up is not None and not np.allclose(args.camera_up, cam_up_used):
        print(f"[step5] [info] --camera-up={args.camera_up} ignored; Genesis single 模式 up 写死 (0,0,1).")

    print(f"[step5] scene.json    = {scene_json_path}")
    print(f"[step5] output config = {output_path}")
    print(f"[step5] camera world pose:")
    print(f"        pos    = {cam_pos.tolist()}")
    print(f"        lookat = {cam_lookat.tolist()}")
    print(f"        up     = {cam_up_used.tolist()}")

    by_name = {o["name"]: o for o in scene["objects"]}

    # World-frame 是唯一支持的路径. scene.json 必须有 "frame": "world".
    scene_frame = scene.get("frame", None)
    if scene_frame != "world":
        print(f"[step5] [fatal] scene.json frame={scene_frame!r}, 但只支持 'world'. "
              f"请用 step3c 的 world-frame 版本生成新 scene.json.")
        sys.exit(2)

    obj_poses = {}
    for obj_cfg in OBJECTS:
        src_name = obj_cfg["name"]
        dst_key  = obj_cfg.get("slot")
        if dst_key is None:
            print(f"[step5] [fatal] pipeline_config.OBJECTS entry {src_name!r} missing 'slot' field")
            sys.exit(2)
        if src_name not in by_name:
            print(f"[step5] [fatal] scene.json 没有 object '{src_name}'. 现有: {list(by_name)}")
            sys.exit(2)
        obj = by_name[src_name]
        spawn_pos, R_genesis = compose_pose(obj["rotation"], obj["translation"])
        # mesh override 优先 (用于 step3c 训练用了非 sam3d 的 mesh, 比如 tripo reference).
        if src_name in mesh_overrides:
            mesh_path = mesh_overrides[src_name]
        else:
            mesh_path = resolve_mesh(src_name)
        scale = float(obj["scale"])
        z_min, z_max = mesh_bbox_in_world(mesh_path, R_genesis, spawn_pos, scale)
        obj_poses[dst_key] = (src_name, spawn_pos, R_genesis, mesh_path, scale, z_min, z_max)

    # 算 z-offset (把最低 mesh 顶到 z=margin).
    if args.no_z_offset:
        z_offset = 0.0
    else:
        global_zmin = min(p[5] for p in obj_poses.values())
        z_offset = max(0.0, -global_zmin + args.z_floor_margin)
        if z_offset > 0:
            print(f"[step5] mug+tree 最低点 z={global_zmin:.3f}m → 整体 (objects+camera+robot) 抬 {z_offset:.3f}m 避免穿地")

    plane_z_final = float(env.get("scene", {}).get("plane_z", 0.0))
    for dst_key in obj_poses:
        (src_name, spawn_pos, R_g, mesh_path, scale, z_min, z_max) = obj_poses[dst_key]
        # debug: 最低点是否真的贴 plane_z (z_offset 加上去后)
        world_zmin_after_lift = z_min + z_offset
        float_mm = (world_zmin_after_lift - plane_z_final) * 1000.0
        print(f"[step5][debug] {dst_key} world height = {(z_max-z_min):.4f}m; "
              f"z_min after lift = {world_zmin_after_lift:.4f} (plane_z={plane_z_final:.3f}; "
              f"{'贴地 ✓' if abs(float_mm) < 1.0 else f'偏 {float_mm:+.1f}mm'})")
        spawn_pos_final = spawn_pos + np.array([0.0, 0.0, z_offset])
        quat = quat_wxyz(R_g)
        patch_object(env, dst_key, mesh_path, spawn_pos_final, quat, scale)
        print(f"[step5] {src_name} → {dst_key}:")
        print(f"        mesh   = {mesh_path}")
        print(f"        pos    = {[round(float(x), 4) for x in spawn_pos_final]}  "
              f"(z_min={z_min+z_offset:.3f}, z_max={z_max+z_offset:.3f})")
        print(f"        quat   = {[round(q, 4) for q in quat]}  (wxyz)")
        print(f"        scale  = {scale:.4f}")

    # camera + robot 跟着抬 z_offset.
    new_cam_pos    = cam_pos    + np.array([0.0, 0.0, z_offset])
    new_cam_lookat = cam_lookat + np.array([0.0, 0.0, z_offset])
    env["camera"]["single_pos"]    = new_cam_pos.tolist()
    env["camera"]["single_lookat"] = new_cam_lookat.tolist()
    env["camera"]["single_up"]     = [0.0, 0.0, 1.0]
    env["camera"]["single_fov"]    = float(cam_cfg.get("single_fov", 64.81))

    env.setdefault("robot", {})
    robot_pos = np.array(env["robot"].get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    env["robot"]["pos"] = (robot_pos + np.array([0.0, 0.0, z_offset])).tolist()

    # 关掉 GS (我们没 .ply).
    env.setdefault("gs_render", {})
    env["gs_render"]["enable"] = False
    env["include_gs_rgb"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tpl, indent=2))
    print(f"\n[step5] wrote {output_path}")


if __name__ == "__main__":
    main()
