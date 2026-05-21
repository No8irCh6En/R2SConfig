#!/usr/bin/env python
"""compare_mug_meshes.py — 看 red/blue/white 三个 mug mesh 的原始大小.

只算 build/objects/<name>__<hash>/mesh.glb (= MV-SAM3D 输出的 raw mesh)
的 bbox extent + 对角线长度. 不读 scene.json, 不应用 pose, 不乘 scale ——
就是 .glb 自己的"内禀大小".

注意 MV-SAM3D 输出的 mesh 是相对单位 (不是米), 所以 dx/dy/dz 的数值本身
没有物理意义, 但**红蓝白三个之间的比值**才是你想验证的 (e.g. 蓝杯的
mesh 比红杯大 1.05x 之类).

--with-scale: 顺手把 outputs/build/scenes/*/prompts/*/scene.json 里 obj_name
entry 的 scale 读出来, 给出 scaled diag = raw_diag * scale (world meter).
(scene.json 现在固定写在 build_prompt_dir 下, 跨 run 共享一份, 不再每个 run
各一份, 所以是去 build/ 里找而不是 outputs/runs/.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh


PROJECT_ROOT  = Path(__file__).resolve().parent
BUILD_OBJECTS = PROJECT_ROOT / "outputs" / "build" / "objects"
BUILD_SCENES  = PROJECT_ROOT / "outputs" / "build" / "scenes"

TARGETS = ["red_mug", "blue_mug", "white_mug"]


def find_mesh(obj_name: str) -> Optional[Path]:
    """outputs/build/objects/<obj_name>__<hash>/mesh.glb"""
    cands = sorted(BUILD_OBJECTS.glob(f"{obj_name}__*/mesh.glb"))
    return cands[0] if cands else None


def mesh_extent(mesh_path: Path) -> Tuple[np.ndarray, float]:
    m = trimesh.load(str(mesh_path), force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    extent = v.max(0) - v.min(0)
    diag = float(np.linalg.norm(extent))
    return extent, diag


def read_scale(obj_name: str) -> Tuple[Optional[float], Optional[Path], str]:
    """找 outputs/build/scenes/*/prompts/*/scene.json 里 name=obj_name 的 entry.

    status:
      'ok'            — 找到 entry
      'no-file'       — 整个 build_scenes 树下没有任何 scene.json
      'wrong-content' — scene.json 存在但里面没有 name=obj_name 的 entry
                         (典型: 某个 scene 还没跑过 step3c)
    """
    cands = sorted(BUILD_SCENES.glob("*/prompts/*/scene.json"))
    if not cands:
        return None, None, "no-file"
    last_seen = None
    for sj in cands:
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        last_seen = sj
        for o in d.get("objects", []):
            if o.get("name") == obj_name:
                return float(o["scale"]), sj, "ok"
    return None, last_seen, "wrong-content"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-scale", action="store_true",
                   help="也读 scene.json 里的 scale, 报 scaled_diag = raw_diag*scale.")
    args = p.parse_args()

    rows = []
    for name in TARGETS:
        path = find_mesh(name)
        if path is None:
            print(f"[warn] no mesh for {name} under {BUILD_OBJECTS}", file=sys.stderr)
            continue
        extent, diag = mesh_extent(path)
        scale_info = read_scale(name) if args.with_scale else None    # (scale, path, status) or None
        rows.append((name, path, extent, diag, scale_info))

    if not rows:
        print("no meshes found.", file=sys.stderr)
        sys.exit(1)

    # ── raw mesh bbox ─────────────────────────────────────────────────
    print(f"\n{'object':12s}  {'dx':>9s} {'dy':>9s} {'dz':>9s}   {'diag':>9s}   (raw .glb units)")
    print("-" * 70)
    for name, path, extent, diag, _ in rows:
        dx, dy, dz = extent
        print(f"{name:12s}  {dx:9.4f} {dy:9.4f} {dz:9.4f}   {diag:9.4f}")
        print(f"             ↳ {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    # ── raw ratios (normalize to red) ─────────────────────────────────
    diag_red = next((r[3] for r in rows if r[0] == "red_mug"), None)
    if diag_red is not None:
        print(f"\nraw diag ratio (red=1.0):")
        for name, _, _, diag, _ in rows:
            print(f"  {name:12s}  {diag/diag_red:.4f}")
    print()

    # ── scaled (world meter) ──────────────────────────────────────────
    if args.with_scale:
        print(f"{'object':12s}  {'scale':>9s}  {'scaled_diag (m)':>16s}   source")
        print("-" * 90)
        scaled = []
        for name, _, _, diag, scale_info in rows:
            if scale_info is None:
                continue
            scale, sj, status = scale_info
            if status == "ok":
                scaled.append((name, diag * scale))
                print(f"{name:12s}  {scale:9.6f}  {diag*scale:16.4f}   {sj.relative_to(PROJECT_ROOT)}")
            elif status == "no-file":
                print(f"{name:12s}  {'?':>9s}  {'?':>16s}   (build/scenes/*/prompts/*/scene.json 都没找到, 先跑 step3c)")
            elif status == "wrong-content":
                rel = sj.relative_to(PROJECT_ROOT) if sj else "?"
                print(f"{name:12s}  {'?':>9s}  {'?':>16s}   {rel}: 没有 entry name='{name}' "
                      f"(对应 scene 的 step3c 还没跑过)")

        sd_red = next((sd for name, sd in scaled if name == "red_mug"), None)
        if sd_red:
            print(f"\nscaled diag ratio (red=1.0):")
            for name, sd in scaled:
                print(f"  {name:12s}  {sd/sd_red:.4f}")
        elif scaled:
            base_name, base_sd = scaled[0]
            print(f"\nscaled diag ratio ({base_name}=1.0):")
            for name, sd in scaled:
                print(f"  {name:12s}  {sd/base_sd:.4f}")
        print()


if __name__ == "__main__":
    main()
