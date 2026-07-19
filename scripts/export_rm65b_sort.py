#!/usr/bin/env python3
"""export_rm65b_sort.py — write per-episode arc/rec poses into a rm65b sort config.

Takes a BASE config (which already carries env_class, pairings, robot, camera, and
the FIXED basket.specs) and, per episode, overwrites ONLY the target.specs poses for
arc / rec from block_configs.json. Everything else — baskets especially — is copied
through from the base config untouched (baskets don't move episode-to-episode).

Per target spec it sets exactly three fields:
    spawn_pos                       <- t           (block_configs episode pose)
    quat (wxyz)                     <- quat(R.T)   (PT3D row-vec R -> Genesis col-vec)
    entity_kwargs.morph_kwargs.scale <- scale
and forces random_xy / random_yaw_deg to 0 so the fitted pose isn't randomized on
reset. The mesh file path in the base spec is left as-is.

Object<->tag mapping: block object "arc" -> target spec tag "arc_target", etc.

Usage:
  python scripts/export_rm65b_sort.py \
      --base-config ../GSRL-exp/experiments/real2sim/rm65b/rm65b_sort_config.json \
      --block-configs outputs/rm65b_sort_v0/block_configs.json \
      --episodes 0-9 \
      --out-dir outputs/rm65b_sort_v0          # writes <out-dir>/ep_NN/sort_config.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def parse_episodes(spec: str) -> list:
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


def transpose(R):
    return [[R[j][i] for j in range(3)] for i in range(3)]


def mat_to_quat_wxyz(m):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z). Shepperd's method."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2][1] - m[1][2]) / S
        y = (m[0][2] - m[2][0]) / S
        z = (m[1][0] - m[0][1]) / S
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / S
        x = 0.25 * S
        y = (m[0][1] + m[1][0]) / S
        z = (m[0][2] + m[2][0]) / S
    elif m[1][1] > m[2][2]:
        S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / S
        x = (m[0][1] + m[1][0]) / S
        y = 0.25 * S
        z = (m[1][2] + m[2][1]) / S
    else:
        S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / S
        x = (m[0][2] + m[2][0]) / S
        y = (m[1][2] + m[2][1]) / S
        z = 0.25 * S
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return [w / n, x / n, y / n, z / n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True, type=Path,
                   help="rm65b sort base config (carries env_class/pairings/robot/"
                        "camera and the FIXED basket.specs)")
    p.add_argument("--block-configs", required=True, type=Path,
                   help="block_configs.json from collect_block_configs.py")
    p.add_argument("--episodes", required=True, help="'0-9' or '0,2,5'")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="writes <out-dir>/ep_NN/sort_config.json")
    p.add_argument("--out-name", default="sort_config.json",
                   help="filename written under each <out-dir>/ep_NN/ (default sort_config.json)")
    args = p.parse_args()

    base = json.loads(args.base_config.read_text())
    blocks = json.loads(args.block_configs.read_text())
    eps = parse_episodes(args.episodes)

    specs = base.get("env", {}).get("target", {}).get("specs")
    if not specs:
        raise SystemExit(f"[export] base config has no env.target.specs — wrong base? "
                         f"({args.base_config})")
    base_tags = [s.get("tag") for s in specs]

    written = []
    for ep in eps:
        ep_key = f"ep_{ep:02d}"
        ep_poses = blocks.get("episodes", {}).get(ep_key)
        if not ep_poses:
            print(f"[export] {ep_key}: not in block_configs — skip")
            continue

        cfg = copy.deepcopy(base)
        updated, unmatched = [], []
        for spec in cfg["env"]["target"]["specs"]:
            tag = spec.get("tag", "")
            obj = tag[:-len("_target")] if tag.endswith("_target") else tag
            if obj not in ep_poses:
                unmatched.append(tag)
                continue
            pose = ep_poses[obj]
            spec["spawn_pos"] = [float(x) for x in pose["t"]]
            spec["quat"] = mat_to_quat_wxyz(transpose(pose["R"]))
            spec.setdefault("entity_kwargs", {}).setdefault("morph_kwargs", {})
            spec["entity_kwargs"]["morph_kwargs"]["scale"] = float(pose["scale"])
            spec["random_xy"] = 0.0          # fitted pose is exact — no reset jitter
            spec["random_yaw_deg"] = 0.0
            updated.append(f"{tag}(iou={pose.get('final_iou', '?'):.3f})"
                           if isinstance(pose.get("final_iou"), float) else tag)

        out_p = args.out_dir / ep_key / args.out_name
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(cfg, indent=2))
        written.append(out_p)
        msg = f"[export] {ep_key}: updated {updated}"
        if unmatched:
            msg += f"  | base target tags NOT in block_configs (left as-is): {unmatched}"
        print(msg)

    print(f"\n[export] base target tags: {base_tags}")
    print(f"[export] baskets/robot/camera/env_class/pairings copied verbatim from "
          f"{args.base_config.name} (NOT touched)")
    print(f"[export] wrote {len(written)} configs, e.g. {written[0] if written else '(none)'}")


if __name__ == "__main__":
    main()
