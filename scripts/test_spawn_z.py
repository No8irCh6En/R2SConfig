"""Diagnose Genesis spawn_pos convention.

Hypothesis: Genesis 把 mesh 的"底"(file 里 scene-graph 下 mesh 的 origin) 当作 spawn_pos,
而 trimesh (我们用的) 把 mesh 的"几何中心"当 origin (因为不应用 scene graph transform).

Test: 在 gsrl_config.json 里把 tree 的 spawn_pos.z 改成几个候选值, 看哪个让树落地.

Usage:
    python scripts/test_spawn_z.py <gsrl_config_path>

会输出 3 个候选 config:
  *_centered.json  — 当前 (假设 trimesh 视角对)
  *_bottom.json    — 假设 Genesis 看到 mesh 底在 origin, spawn_pos.z = plane_z
  *_check.json     — 给 .glb 实际节点变换的差额加进去
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def inspect_glb_translation(glb_path):
    """检查 .glb 的 scene graph 有没有给主 mesh 加 translate."""
    s = trimesh.load(str(glb_path))
    print(f"\n[inspect] {glb_path}")
    print(f"  Scene? {isinstance(s, trimesh.Scene)}")
    if isinstance(s, trimesh.Scene):
        # 用 scene.dump (会应用 scene transform) vs 直接拿 geometry (不应用)
        print(f"  Scene graph nodes: {list(s.graph.nodes)}")
        print(f"  Geometry names: {list(s.geometry.keys())}")
        # 应用 scene graph 取顶点
        v_with_xform = []
        for g_name, g in s.geometry.items():
            T = s.graph.get(g_name)
            if T is not None:
                M = T[0]
                v_local = np.asarray(g.vertices)
                v_world = (M @ np.column_stack([v_local, np.ones(len(v_local))]).T).T[:, :3]
                v_with_xform.append(v_world)
                print(f"  [{g_name}] node transform M[0:3,3]={M[:3, 3].tolist()}")
            else:
                v_with_xform.append(np.asarray(g.vertices))
        if v_with_xform:
            v_all = np.concatenate(v_with_xform, axis=0)
            print(f"  WITH scene transforms: extent={ (v_all.max(0)-v_all.min(0)).round(4).tolist() }")
            print(f"                          z range=[{v_all[:,2].min():.4f}, {v_all[:,2].max():.4f}]")
            print(f"                          y range=[{v_all[:,1].min():.4f}, {v_all[:,1].max():.4f}]")

        # 不应用 scene transform
        v_no_xform = []
        for g in s.geometry.values():
            v_no_xform.append(np.asarray(g.vertices))
        v_all2 = np.concatenate(v_no_xform, axis=0)
        print(f"  WITHOUT scene transforms: extent={ (v_all2.max(0)-v_all2.min(0)).round(4).tolist() }")
        print(f"                              z range=[{v_all2[:,2].min():.4f}, {v_all2[:,2].max():.4f}]")
        print(f"                              y range=[{v_all2[:,1].min():.4f}, {v_all2[:,1].max():.4f}]")


def make_variants(config_path):
    """生成 3 个 config 测试 spawn_pos.z 候选."""
    cfg = json.loads(Path(config_path).read_text())
    src = Path(config_path)

    for key in ["mug", "tree"]:
        obj = cfg["env"].get(key)
        if obj is None:
            continue
        glb = obj.get("asset_path")
        if glb:
            inspect_glb_translation(glb)

    # variant 1: spawn_pos.z = plane_z (假设 Genesis 用 file origin = bottom)
    cfg_bottom = json.loads(json.dumps(cfg))
    plane_z = cfg_bottom["env"].get("scene", {}).get("plane_z", 0.0)
    for key in ["mug", "tree"]:
        if key in cfg_bottom["env"]:
            obj = cfg_bottom["env"][key]
            sp = obj["spawn_pos"]
            obj["spawn_pos"] = [sp[0], sp[1], plane_z + 0.005]   # 5mm buffer
    out_bot = src.with_name(src.stem + "_bottom.json")
    out_bot.write_text(json.dumps(cfg_bottom, indent=2))
    print(f"\n[wrote] {out_bot}")
    print(f"  spawn_pos.z 全设成 plane_z + 0.005 = {plane_z + 0.005}")
    print(f"  如果重渲后树/杯子贴地 → 假设成立, Genesis 用 file-origin = bottom 约定")

    # variant 2: spawn_pos.z = optimizer 输出 + 一半 mesh 高度 (offset to bottom)
    cfg_shifted = json.loads(json.dumps(cfg))
    for key in ["mug", "tree"]:
        if key in cfg_shifted["env"]:
            obj = cfg_shifted["env"][key]
            sp = obj["spawn_pos"]
            scale = obj.get("entity_kwargs", {}).get("morph_kwargs", {}).get("scale", 1.0)
            # mesh z 半高 (trimesh-view, 假设 z_min ≈ -0.5, z_max ≈ +0.5 → 半高 = 0.5)
            half_h = 0.5  # rough
            obj["spawn_pos"] = [sp[0], sp[1], sp[2] - half_h * scale]
    out_shift = src.with_name(src.stem + "_shifted.json")
    out_shift.write_text(json.dumps(cfg_shifted, indent=2))
    print(f"\n[wrote] {out_shift}")
    print(f"  spawn_pos.z 减去 (scale * 0.5)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    make_variants(args.config)
