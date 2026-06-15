#!/usr/bin/env python
"""Inspect mesh orientation: raw .glb extent vs M_LOAD'd extent vs PCA long_axis.

跑 (任何有 numpy + trimesh 的 env 都行):
    python scripts/inspect_mesh_orientation.py --mesh outputs/runs/20260604_012741/gsrl_config.json
        或
    python scripts/inspect_mesh_orientation.py --glb path/to/result.glb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


M_PT3D = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def extent_str(extent):
    return f"dx={extent[0]:.4f}  dy={extent[1]:.4f}  dz={extent[2]:.4f}  (longest: {'XYZ'[int(np.argmax(extent))]})"


def inspect(glb_path: Path):
    mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    v_raw = np.asarray(mesh.vertices, dtype=np.float64)
    print(f"\n=== {glb_path} ===")
    print(f"  raw extent (在 .glb 文件里): {extent_str(v_raw.max(0) - v_raw.min(0))}")

    # M_LOAD as PT3D does: v @ M_PT3D.T
    v_loaded = v_raw @ M_PT3D.T.astype(np.float64)
    print(f"  M_LOAD'd extent (PT3D 优化里看到的): {extent_str(v_loaded.max(0) - v_loaded.min(0))}")

    # PCA on M_LOAD'd verts
    centered = v_loaded - v_loaded.mean(0)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    long_axis = eigvecs[:, -1]
    proj = centered @ long_axis
    skew = float((proj ** 3).mean())
    print(f"  PCA long_axis (M_LOAD'd 系下): {[round(float(v), 3) for v in long_axis]}  "
          f"skew={skew:+.4f}")
    if skew < 0:
        long_axis = -long_axis
        print(f"  → 符号修正后 long_axis = {[round(float(v), 3) for v in long_axis]}  "
              f"(指向'重端' / 杯底)")
    else:
        print(f"  → 不需要符号修正, long_axis 已指向'重端' / 杯底")
    print(f"  → 物体真实 up 方向 (M_LOAD'd 系下) ≈ {[round(float(-v), 3) for v in long_axis]}  "
          f"(=base 方向取反)")

    # 验证: 是不是真的杯子高度 ≈ ±Y 方向
    abs_y = abs(long_axis[1])
    if abs_y > 0.85:
        sign = "+" if long_axis[1] > 0 else "-"
        sign_up = "-" if long_axis[1] > 0 else "+"
        print(f"  ✓ long_axis 主要沿 Y 方向 ({sign}Y is 重端 / {sign_up}Y is 杯子的 up)")
        print(f"  ✓ 如果 yaw_only 锁 mesh +Y 到 anti-gravity: "
              f"{'⚠️  锁的是杯底方向, 杯子上下颠倒' if long_axis[1] > 0 else '正确: 锁的是杯子 up'}")
    else:
        print(f"  ⚠️  long_axis 不是 ±Y 方向 — yaw_only 锁 +Y 跟杯子真实 up 不重合")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glb", help="path to a single .glb")
    p.add_argument("--config", help="gsrl_config.json, will inspect all object meshes referenced")
    args = p.parse_args()

    if args.glb:
        inspect(Path(args.glb))
    elif args.config:
        cfg = json.loads(Path(args.config).read_text())
        for key in ("mug", "tree"):
            if key in cfg.get("env", {}):
                ap = cfg["env"][key].get("asset_path")
                if ap:
                    inspect(Path(ap))
    else:
        print("用 --glb <path> 或 --config <gsrl_config.json>", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
