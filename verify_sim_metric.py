#!/usr/bin/env python
"""verify_sim_metric.py — 验证当前 sim 出的 mug world size 跟 tripo 真值一不一致.

sim mug 大小 (Genesis 里实际渲染的大小, 单位米) =
    step3c_scale × raw_sam3d_mesh.bbox
其中:
    step3c_scale 从 build_prompt_dir/scene.json 读 (对应 mug entry 的 'scale')
    raw_sam3d_mesh 从 build_object_dir/mesh.glb 读
    这俩相乘 = Genesis morph_kwargs.scale × mesh.glb.bbox = 真实 spawn 大小

真实 mug 大小 (用户给的 ground truth, 单位米) =
    mug_X_tripo.bbox × scale_factor
其中:
    mug_0_tripo / mug_1_tripo 在 example/ 下, 这俩 .glb 本来就是 metric (米)
    scale_factor:
        white = 1.3 × mug_0
        red   = 1.2 × mug_1
        blue  = 1.0 × mug_1

判定:
    - sim/real 都 ≈ 1.00  → 当前 pipeline metric 已正确, 不用 Plan A ✅
    - sim/real 都 ≈ 同一个常数 k → 全局乘 1/k 就修了 (放进 PipelineConfig 或 step5)
    - sim/real 三者不一致 → 需要 Plan A: per-object real_scale

最后会给出**具体应该填的 real_scale 数字** (= real_size / current_sim_size),
对照用户的物理直觉 (1.0/1.2/1.3) 看吻合不吻合.

用任意有 numpy + trimesh 的 env 跑 (sam3d-objects / sam3 / gsrl-lc 都行):
    conda activate sam3d-objects
    python verify_sim_metric.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parent

# 每个 colored mug 怎么定位它的 sim 输出 + ground-truth tripo base
TARGETS = {
    "red_mug":   {
        "scene_id":      "red_with_mug_tree",
        "mesh":          "outputs/build/objects/red_mug__f10e76c7/mesh.glb",
        "base":          "example/mug_1.obj",
        "scale_factor":  1.2,
    },
    "blue_mug":  {
        "scene_id":      "blue_with_mug_tree",
        "mesh":          "outputs/build/objects/blue_mug__878169e6/mesh.glb",
        "base":          "example/mug_1.obj",
        "scale_factor":  1.0,
    },
    "white_mug": {
        "scene_id":      "white_with_mug_tree",
        "mesh":          "outputs/build/objects/white_mug__573efe34/mesh.glb",
        "base":          "example/mug_0.obj",
        "scale_factor":  1.3,
    },
}


def resolve_mesh_path(cfg) -> Path:
    """build/objects/<hash>/mesh.glb (step2 symlink), 不在就 fallback 到
    MV-SAM3D/visualization/<hash>/**/result.glb 里最新的那个."""
    p = PROJECT / cfg["mesh"]
    if p.exists():
        return p
    hashed = Path(cfg["mesh"]).parent.name  # e.g. "red_mug__f10e76c7"
    vis = PROJECT / "MV-SAM3D" / "visualization"
    cands = list(vis.glob(f"{hashed}/**/result.glb"))
    if cands:
        return max(cands, key=lambda x: x.stat().st_mtime)
    return p  # 不存在, 留给 caller 报错/skip


def mesh_extent(p: Path):
    """Return (dx, dy, dz, diag) of axis-aligned bbox."""
    m = trimesh.load(str(p), force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    extent = v.max(0) - v.min(0)
    diag = float(np.linalg.norm(extent))
    return (float(extent[0]), float(extent[1]), float(extent[2]), diag)


def find_scene_json(scene_id: str):
    """outputs/build/scenes/<scene_id>/prompts/*/scene.json (取最新)."""
    base = PROJECT / "outputs/build/scenes" / scene_id / "prompts"
    if not base.exists():
        return None
    cands = sorted(base.glob("*/scene.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def read_step3c_scale(scene_json_path: Path, mug_name: str):
    try:
        d = json.loads(scene_json_path.read_text())
    except Exception as e:
        print(f"[fatal] 读不出 {scene_json_path}: {e}", file=sys.stderr)
        return None
    for o in d.get("objects", []):
        if o.get("name") == mug_name:
            return float(o["scale"])
    return None


def main():
    rows = []

    # ── Step 1: 收 3 个 mug 的数据 ────────────────────────────────────
    print("=" * 100)
    print("Step 1: 收数据")
    print("=" * 100)
    for name, cfg in TARGETS.items():
        mesh_p = resolve_mesh_path(cfg)
        base_p = PROJECT / cfg["base"]
        if not mesh_p.exists():
            print(f"  [skip] {name}: 找不到 SAM3D mesh (build/objects/<hash>/mesh.glb 也没,"
                  f" MV-SAM3D/visualization/{Path(cfg['mesh']).parent.name}/ 也没 result.glb)");
            continue
        if not base_p.exists():
            print(f"  [skip] {name}: tripo base 不存在 {base_p}"); continue
        sj = find_scene_json(cfg["scene_id"])
        if sj is None:
            print(f"  [skip] {name}: 找不到 scene.json (scene_id={cfg['scene_id']}), 先跑 step3c"); continue
        scale = read_step3c_scale(sj, name)
        if scale is None:
            print(f"  [skip] {name}: scene.json 里没有 name='{name}' 的 entry"); continue

        raw      = mesh_extent(mesh_p)                  # sam3d canonical, 4-tuple
        base_ext = mesh_extent(base_p)                  # tripo metric (米), 4-tuple
        sim      = tuple(v * scale            for v in raw)        # sim world (米)
        real     = tuple(v * cfg["scale_factor"] for v in base_ext) # 真实 (米)

        rows.append({
            "name":         name,
            "scale_factor": cfg["scale_factor"],
            "base_name":    Path(cfg["base"]).stem,
            "step3c_scale": scale,
            "raw":          raw,
            "base_ext":     base_ext,
            "sim":          sim,
            "real":         real,
            "scene_json":   sj,
        })
        print(f"  {name:10s}  step3c_scale={scale:.4f}  raw_diag={raw[3]:.4f}  "
              f"base={Path(cfg['base']).stem}  base_diag={base_ext[3]:.4f}  factor={cfg['scale_factor']}")

    if not rows:
        print("\n[fatal] 没收到任何数据, 退出.")
        return

    # ── Step 2: per-mug, per-axis sim vs real ────────────────────────
    print()
    print("=" * 100)
    print("Step 2: 每个 mug 的 sim world size  vs  真实 physical size  (单位: 米)")
    print("=" * 100)
    for r in rows:
        print(f"\n  [{r['name']}]   sim = step3c_scale × sam3d_raw   |   "
              f"real = {r['base_name']} × {r['scale_factor']}")
        print(f"    {'axis':5s}  {'sim (m)':>10s}  {'real (m)':>10s}  {'sim / real':>11s}  {'(sim-real)/real':>17s}")
        labels = ["dx", "dy", "dz", "diag"]
        for i, lbl in enumerate(labels):
            s, x = r["sim"][i], r["real"][i]
            ratio = s / x if x else 0.0
            err   = (s - x) / x * 100 if x else 0.0
            print(f"    {lbl:5s}  {s:10.4f}  {x:10.4f}  {ratio:11.4f}  {err:+16.1f}%")

    # ── Step 3: 三个 mug 的 sim/real 是不是同一个数 ────────────────────
    print()
    print("=" * 100)
    print("Step 3: 各 mug 的 sim/real diag 比例 — 是不是一致")
    print("=" * 100)
    print(f"  {'name':<10s}  {'sim_diag (m)':>14s}  {'real_diag (m)':>15s}  {'sim/real':>10s}")
    diag_ratios = []
    for r in rows:
        rr = r["sim"][3] / r["real"][3]
        diag_ratios.append(rr)
        print(f"  {r['name']:<10s}  {r['sim'][3]:14.4f}  {r['real'][3]:15.4f}  {rr:10.4f}")
    rmin, rmax = min(diag_ratios), max(diag_ratios)
    rmean = sum(diag_ratios) / len(diag_ratios)
    spread_pct = (rmax - rmin) / rmin * 100
    print(f"\n  range = [{rmin:.4f}, {rmax:.4f}]   mean = {rmean:.4f}   spread = {spread_pct:.1f}%")

    # ── Step 4: 给出建议的 real_scale (精确修正值) ────────────────────
    print()
    print("=" * 100)
    print("Step 4: 建议怎么改 (3 种可能场景)")
    print("=" * 100)
    if abs(rmean - 1.0) < 0.03 and spread_pct < 3.0:
        print(f"\n  ✅ sim 跟 real 已经一致 (mean ratio {rmean:.4f}, spread {spread_pct:.1f}%)")
        print(f"     → 不需要任何修正, Plan A 也别加, 直接跑 Genesis sim 就对.")
        return

    if spread_pct < 5.0:
        k = rmean
        print(f"\n  ⚠️  3 个比例集中在 {rmean:.4f} 附近 (spread {spread_pct:.1f}%), 全局偏 {(k-1)*100:+.0f}%.")
        print(f"     → 全局乘 1/{k:.4f} = {1/k:.4f} 就修了 (1 个数, 不需要 per-object).")
        print(f"     → 不必上 Plan A, 在 step5 出 spawn 时乘个常数就行.")
        return

    # spread 大, 需要 per-object real_scale (Plan A)
    print(f"\n  ❌ 3 个比例 spread {spread_pct:.0f}% 太大, 没法用单一全局因子修.")
    print(f"     → 需要 Plan A: per-object real_scale.")
    print()
    print(f"  建议的 real_scale 值 (= real_diag / sim_diag, 直接放进 pipeline_config OBJECTS):")
    print(f"  {'name':<10s}  {'你想填的物理 ratio':>18s}  {'精确修正值 (新)':>18s}  {'差异':>10s}")
    for r, rr in zip(rows, diag_ratios):
        intended = r["scale_factor"]    # 用户最初的直觉: 1.2 / 1.0 / 1.3
        exact    = 1.0 / rr              # 真正能让 sim_diag == real_diag 的值
        diff_pct = (exact - intended) / intended * 100
        print(f"  {r['name']:<10s}  {intended:18.4f}  {exact:18.4f}  {diff_pct:+9.1f}%")

    print()
    print(f"  解读:")
    print(f"    - 用户直觉填 1.2/1.0/1.3 只修了**相对**比例 (红:蓝:白 之间),")
    print(f"      没修 step3c 的**绝对**偏移 (sim 比 real 大 {(rmean-1)*100:.0f}%).")
    print(f"    - 上面 '精确修正值' 把两件事一起摊到 real_scale 里, 一步到位.")
    print()
    print(f"  你需要决定:")
    print(f"    [A] 只要相对比例对 (Genesis 里红比蓝大 20% etc., 绝对大小不在乎)")
    print(f"        → real_scale = 1.2 / 1.0 / 1.3")
    print(f"    [B] 绝对+相对都准 (Genesis 里 mug 大小 = 真实大小)")
    print(f"        → real_scale = 上面那 3 个 '精确修正值'")


if __name__ == "__main__":
    main()
