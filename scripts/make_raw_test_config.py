#!/usr/bin/env python
"""生成 minimal gsrl_config.json: mug + tree 都放原点附近, identity quat, scale=1.
用来做"两个 mesh 在不经任何处理下, 在 PT3D 和 Genesis 里各长什么样"的对照测试.

用法:
    # 组 A: 不设 file_meshes_are_zup → PT3D + Genesis 都应用 M_LOAD (默认)
    python scripts/make_raw_test_config.py --output outputs/render_raw/with_mload/gsrl_config.json
    # 组 B: 设 file_meshes_are_zup=True → 两边都跳 M_LOAD
    python scripts/make_raw_test_config.py --output outputs/render_raw/no_mload/gsrl_config.json --zup
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.io.paths import resolve as resolve_paths
from real2sim.perception.mesh_io import find_glb_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="blue_with_mug_tree",
                   help="决定从哪 scene 的 OBJECTS 找 mesh; 默认 blue_with_mug_tree")
    p.add_argument("--output", required=True, help="生成的 gsrl_config.json 路径")
    p.add_argument("--zup", action="store_true",
                   help="给 morph_kwargs 设 file_meshes_are_zup=True (= 两边跳 M_LOAD)")
    p.add_argument("--mug-pos",  type=float, nargs=3, default=[0.30, -0.10, 0.10])
    p.add_argument("--tree-pos", type=float, nargs=3, default=[0.30, +0.15, 0.10])
    p.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()

    # 用 example/1.json 作 base, 然后覆盖 mug + tree 的 pose / scale / file
    import os
    os.environ.setdefault("SCENE", args.scene)
    paths = resolve_paths()

    template = PROJECT_ROOT / "example" / "1.json"
    cfg = json.loads(template.read_text())
    env = cfg["env"]

    # 找 mesh 路径 — 根据 scene 决定哪个 mug
    if args.scene == "blue_with_mug_tree":
        mug_name = "blue_mug"
    elif args.scene == "red_with_mug_tree":
        mug_name = "red_mug"
    elif args.scene == "white_with_mug_tree":
        mug_name = "white_mug"
    else:
        print(f"[fatal] 未知 scene: {args.scene}"); sys.exit(2)
    tree_name = "black_mug_tree"
    glb_paths = find_glb_files([mug_name, tree_name], paths=paths)

    for nm in (mug_name, tree_name):
        if nm not in glb_paths:
            print(f"[fatal] 找不到 {nm} 的 mesh, 先跑 step2"); sys.exit(2)

    def fill(slot_key, mesh_path, spawn_pos):
        env[slot_key]["asset_path"] = str(mesh_path)
        env[slot_key]["gs_path"] = None
        env[slot_key]["spawn_pos"] = list(spawn_pos)
        env[slot_key]["spawn_quat"] = [1.0, 0.0, 0.0, 0.0]   # identity (wxyz)
        morph = env[slot_key].setdefault("entity_kwargs", {}).setdefault("morph_kwargs", {})
        morph["file"] = str(mesh_path)
        morph["scale"] = float(args.scale)
        if args.zup:
            morph["file_meshes_are_zup"] = True
        else:
            morph.pop("file_meshes_are_zup", None)

    fill("mug",  glb_paths[mug_name],   args.mug_pos)
    fill("tree", glb_paths[tree_name],  args.tree_pos)

    # 关掉 GS, 用一个简单 workspace_dir (step5_render 会被 --workspace-dir 覆盖)
    env.setdefault("gs_render", {})["enable"] = False
    env["gs_render"]["workspace_dir"] = str(Path(args.output).parent.resolve())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(f"saved: {out}")
    print(f"  mug:  {glb_paths[mug_name]}  spawn_pos={args.mug_pos}  identity quat  scale=1")
    print(f"  tree: {glb_paths[tree_name]} spawn_pos={args.tree_pos} identity quat  scale=1")
    print(f"  file_meshes_are_zup = {args.zup}")


if __name__ == "__main__":
    main()
