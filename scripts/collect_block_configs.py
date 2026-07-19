#!/usr/bin/env python3
"""collect_block_configs.py — gather per-episode arc/rec poses into ONE config.

Reads <scene-root>/ep_NN/poses/<object>.json (step4 multi-cam output) for a range
of episodes and writes a single combined JSON + a CSV summary.

Layout of the output config:
  {
    "dataset": "rm65b_sort_v0",
    "plane_z": 0.0,
    "objects": {                          # per-object CONSTANTS (same every episode)
      "arc": {"mesh": "...glb", "mesh_prerot": [[...]], "laydown": true},
      "rec": {"mesh": "...glb", "mesh_prerot": [[1,0,0],[0,1,0],[0,0,1]], "laydown": false}
    },
    "episodes": {                         # per-episode pose (R is mesh->world, PT3D row-vec)
      "ep_00": {"arc": {"R":[[...]], "t":[...], "scale":.., "final_iou":.., "best_yaw_deg":..},
                "rec": {...}},
      ...
    }
  }

`mesh_prerot` is the --laydown / --mesh-prerot rotation (constant per object); the
per-episode `R` already has it folded in (R = mesh_prerot @ R_fit_yaw), so a Genesis
exporter can use `R` directly OR (mesh_prerot, R_fit_yaw) separately.

Usage:
  python scripts/collect_block_configs.py --scene-root outputs/rm65b_sort_v0 \
      --episodes 0-9 [--objects arc,rec] [--out outputs/rm65b_sort_v0/block_configs.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IDENT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def parse_episodes(spec: str) -> list:
    """'0-9' -> [0..9];  '0,2,5' -> [0,2,5];  '0-3,7' -> [0,1,2,3,7]."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-root", required=True, type=Path,
                   help="dir containing ep_NN/ scene dirs (e.g. outputs/rm65b_sort_v0)")
    p.add_argument("--episodes", required=True,
                   help="episode spec: '0-9' or '0,2,5' or '0-3,7'")
    p.add_argument("--objects", default="arc,rec",
                   help="comma list of object slugs to collect (default arc,rec)")
    p.add_argument("--out", default=None, type=Path,
                   help="output JSON (default <scene-root>/block_configs.json)")
    p.add_argument("--iou-warn", type=float, default=0.5,
                   help="flag episodes whose object IoU is below this (default 0.5)")
    args = p.parse_args()

    eps = parse_episodes(args.episodes)
    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    out_p = args.out or (args.scene_root / "block_configs.json")

    obj_const: dict = {}          # per-object constants (mesh, mesh_prerot, laydown)
    episodes: dict = {}
    rows = []                     # CSV rows
    missing, low = [], []

    for ep in eps:
        ep_key = f"ep_{ep:02d}"
        scene = args.scene_root / ep_key
        ep_entry = {}
        for obj in objects:
            pose_p = scene / "poses" / f"{obj}.json"
            if not pose_p.exists():
                missing.append(f"{ep_key}/{obj}")
                rows.append([ep_key, obj, "", "", "", "MISSING"])
                continue
            d = json.loads(pose_p.read_text())
            iou = float(d.get("final_iou", float("nan")))
            ep_entry[obj] = {
                "R": d["R"],
                "t": d["t"],
                "scale": float(d["scale"]),
                "final_iou": iou,
                "best_yaw_deg": d.get("best_yaw_deg"),
            }
            # per-object constants — capture once, then check consistency
            mp = d.get("mesh_prerot", IDENT)
            const = {"mesh": d.get("mesh"), "mesh_prerot": mp,
                     "laydown": bool(d.get("laydown", False))}
            if obj not in obj_const:
                obj_const[obj] = const
            elif obj_const[obj]["mesh_prerot"] != mp:
                print(f"[warn] {obj}: mesh_prerot differs at {ep_key} "
                      f"(--laydown auto-rotation not stable across episodes; "
                      f"consider an explicit --mesh-prerot in the batch)")
            status = "ok"
            if iou < args.iou_warn:
                low.append(f"{ep_key}/{obj} (iou={iou:.3f})")
                status = "low_iou"
            rows.append([ep_key, obj, f"{iou:.4f}", f"{d['scale']:.4f}",
                         str(d.get("best_yaw_deg")), status])
        if ep_entry:
            episodes[ep_key] = ep_entry

    config = {
        "dataset": args.scene_root.name,
        "plane_z": 0.0,
        "objects": obj_const,
        "episodes": episodes,
    }
    out_p.write_text(json.dumps(config, indent=2))

    csv_p = out_p.with_suffix(".csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "object", "final_iou", "scale", "best_yaw_deg", "status"])
        w.writerows(rows)

    # ── report ──
    print(f"[collect] {len(episodes)} episodes, objects={objects}")
    print(f"[collect] objects constants:")
    for o, c in obj_const.items():
        print(f"    {o}: laydown={c['laydown']}  mesh_prerot={c['mesh_prerot']}  mesh={c['mesh']}")
    print(f"\n  {'episode':9} " + "  ".join(f"{o+'_iou':>10}" for o in objects))
    for ep in eps:
        ek = f"ep_{ep:02d}"
        cells = []
        for o in objects:
            v = episodes.get(ek, {}).get(o, {}).get("final_iou")
            cells.append(f"{v:>10.3f}" if isinstance(v, float) else f"{'--':>10}")
        print(f"  {ek:9} " + "  ".join(cells))
    if missing:
        print(f"\n[warn] missing poses: {missing}")
    if low:
        print(f"[warn] low IoU (<{args.iou_warn}): {low}")
    print(f"\n[collect] wrote {out_p}")
    print(f"[collect] wrote {csv_p}")


if __name__ == "__main__":
    main()
