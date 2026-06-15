"""检查 SAM3D mesh 的 z_min 是不是被杂点污染. 在 sam3d-objects env 跑.

用法:
    conda activate sam3d-objects
    python scripts/check_mesh_zmin.py <path/to/result.glb>

输出:
- raw .glb extent (M_LOAD 前)
- M_LOAD 后 extent
- z 各 percentile (p0=strict min, p0.5, p1, p5, p10, p50)
- 杂点数量: z 在 [z_min, z_min+0.01] 范围内的顶点数

判定:
- p1 - p0 < 0.005m  → z_min 干净, 飘起的不是这个原因
- p1 - p0 > 0.01m   → z_min 有杂点, 用 percentile 代替严格 min 能修
"""
import sys
import numpy as np
import trimesh


def main(glb_path):
    s = trimesh.load(glb_path)
    if isinstance(s, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in s.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        m = s
    v = np.asarray(m.vertices)
    print(f"raw .glb: {len(v)} verts")
    print(f"  extent (x,y,z): {(v.max(0) - v.min(0)).round(4).tolist()}")
    M_PT3D_T = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    v2 = v @ M_PT3D_T
    print(f"after M_LOAD: {len(v2)} verts")
    print(f"  extent (x,y,z): {(v2.max(0) - v2.min(0)).round(4).tolist()}")
    z = v2[:, 2]
    print(f"  z range: [{z.min():.4f}, {z.max():.4f}]")
    print()
    print("  z percentiles:")
    for p in [0, 0.1, 0.5, 1, 2, 5, 10, 50]:
        print(f"    p{p:>4} = {np.percentile(z, p):.4f}")
    print()
    gap_01 = float(np.percentile(z, 1) - z.min())
    gap_05 = float(np.percentile(z, 5) - z.min())
    n_within_1cm = int((z < z.min() + 0.01).sum())
    n_within_5cm = int((z < z.min() + 0.05).sum())
    print(f"  verts within 1cm of z_min: {n_within_1cm} / {len(z)}")
    print(f"  verts within 5cm of z_min: {n_within_5cm} / {len(z)}")
    print(f"  gap z_min->p1: {gap_01*100:.2f}cm")
    print(f"  gap z_min->p5: {gap_05*100:.2f}cm")
    print()
    if gap_01 < 0.005:
        print("  → z_min 干净, 飘起不是这个原因 (检查 scale 或 t_x,t_y)")
    elif gap_01 < 0.01:
        print("  → z_min 边界, 有少量杂点 (gap < 1cm), 影响微小")
    else:
        print(f"  → z_min 有杂点! 用 percentile (e.g. p1) 代替严格 min 能让模型贴地 "
              f"(差 {gap_01*100:.1f}cm)")


if __name__ == "__main__":
    main(sys.argv[1])
