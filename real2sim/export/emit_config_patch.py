#!/usr/bin/env python3
"""emit_config_patch.py — per-episode poses -> base+patch config (R2SConfig ⇄ GSRL-exp contract).

Per episode, reads our `poses/<object>.json` (R, t, scale) and emits two files:
  * `<patch-name>`  — the CONTRACT: mirrors the base structure, only the changed fields.
  * `<config-name>` — base + patch merged (directly runnable via env_cls(config=...)).

Frozen numeric core (never changes): spawn_pos = t ; quat = wxyz(Rᵀ) ; scale = scale.
(Our R is PT3D row-vec v_world = v_mesh @ R; Genesis wants col-vec = Rᵀ. No extra M_LOAD —
Genesis auto-applies y-up→z-up on .glb load.)

Orientation-KEY rule (there is NO alias/normalizer on the consumer side, so the patch key
MUST equal the base key for that object): for each object we READ the base spec — if it has
`spawn_quat` we write `spawn_quat`, elif it has `quat` we write `quat`, else (base has no
orientation key, e.g. rm65b_sort_base target specs) we fall back to the dataset's
`default_quat_key`. Position is always `spawn_pos`. A key that doesn't match the base is
silently ignored by the env (identity quat) — hence read-the-base, don't guess.

Usage:
  python -m real2sim.export.emit_config_patch \
      --dataset xarm_hang_blue_mug_v0 \
      --scene-root outputs/xarm_hang_blue_mug_v0_crop --episodes 0-34
  # writes outputs/.../ep_NN/patch.json + ep_NN/config.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # real2sim/export/.. .. == R2SConfig/
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.export.config_merge import merge_config    # vendored, frozen contract

# scale always lives here in both base configs
SCALE_PATH = ["entity_kwargs", "morph_kwargs", "scale"]

# Per-dataset object→slot map. A slot is either a dict path ("slot") or a tagged-list
# location ("list" + "tag"). `default_quat_key` is used only when the base spec carries
# no orientation key of its own.
DATASETS = {
    "xarm_hang_blue_mug_v0": {
        "base": "example/xarm_mug_tree_config.json",
        "default_quat_key": "spawn_quat",
        "objects": {
            "blue_mug": {"slot": ["env", "mug"]},
            "mug_tree": {"slot": ["env", "tree"]},
        },
    },
    "rm65b_sort_v0": {
        "base": "example/rm65b_sort_base.json",
        # rm65b_sort_base.json is the PlaceInBasketEnv family → reads `quat`; its target
        # specs carry no orientation key, so this default applies to arc/rec.
        "default_quat_key": "quat",
        "objects": {
            "arc": {"list": ["env", "target", "specs"], "tag": "arc_target"},
            "rec": {"list": ["env", "target", "specs"], "tag": "rec_target"},
        },
    },
}


# ── frozen numeric core ───────────────────────────────────────────────────────
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


def pose_to_placement(pose: dict):
    """Our pose json (R row-vec, t, scale) -> (spawn_pos, quat_wxyz, scale)."""
    spawn_pos = [float(x) for x in pose["t"]]
    quat = mat_to_quat_wxyz(transpose(pose["R"]))
    return spawn_pos, quat, float(pose["scale"])


# ── nested helpers ────────────────────────────────────────────────────────────
def get_nested(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_nested(d, path, value):
    """Set d[path...] = value, creating intermediate dicts; siblings untouched."""
    cur = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def find_tagged(base, list_path, tag):
    lst = get_nested(base, list_path, default=None)
    if isinstance(lst, list):
        for e in lst:
            if isinstance(e, dict) and e.get("tag") == tag:
                return e
    return None


def orientation_key(base_spec, default_quat_key):
    """Mirror the base's orientation key: spawn_quat > quat > dataset default."""
    if isinstance(base_spec, dict):
        if "spawn_quat" in base_spec:
            return "spawn_quat"
        if "quat" in base_spec:
            return "quat"
    return default_quat_key


# ── patch assembly ────────────────────────────────────────────────────────────
def build_patch(base, objects_map, placements, default_quat_key, provenance=None):
    """Build a mirror-base patch dict from per-object placements.

    placements: {object_name: (spawn_pos, quat_wxyz, scale)}. Only objects present in
    both `objects_map` and `placements` are emitted.
    """
    patch: dict = {}
    tagged: dict = {}      # tuple(list_path) -> [spec dicts]
    for obj, m in objects_map.items():
        if obj not in placements:
            continue
        spawn_pos, quat, scale = placements[obj]
        if "slot" in m:
            base_spec = get_nested(base, m["slot"], default={})
            qkey = orientation_key(base_spec, default_quat_key)
            node = {"spawn_pos": spawn_pos, qkey: quat}
            set_nested(node, SCALE_PATH, scale)
            set_nested(patch, m["slot"], node)
        else:                                   # tagged-list slot
            base_spec = find_tagged(base, m["list"], m["tag"]) or {}
            qkey = orientation_key(base_spec, default_quat_key)
            spec = {"tag": m["tag"], "spawn_pos": spawn_pos, qkey: quat}
            set_nested(spec, SCALE_PATH, scale)
            tagged.setdefault(tuple(m["list"]), []).append(spec)
    for list_path, specs in tagged.items():
        set_nested(patch, list(list_path), specs)
    if provenance:
        patch["_r2s"] = provenance
    return patch


def emit_episode(dataset, scene_root, ep, base=None):
    """Return (patch, merged) for one episode, or None if no poses were found."""
    ds = DATASETS[dataset]
    if base is None:
        base = json.loads((PROJECT_ROOT / ds["base"]).read_text())
    poses_dir = Path(scene_root) / f"ep_{ep:02d}" / "poses"
    placements, prov_objs = {}, {}
    for obj in ds["objects"]:
        pf = poses_dir / f"{obj}.json"
        if not pf.exists():
            continue
        pose = json.loads(pf.read_text())
        placements[obj] = pose_to_placement(pose)
        iou = pose.get("final_iou")
        prov_objs[obj] = {
            "iou": round(float(iou), 4) if isinstance(iou, (int, float)) else None,
            "mesh": pose.get("mesh"),
            "src": pf.name,
        }
    if not placements:
        return None
    provenance = {"episode": int(ep), "dataset": dataset, "objects": prov_objs}
    patch = build_patch(base, ds["objects"], placements, ds["default_quat_key"], provenance)
    merged = merge_config(base, patch)
    return patch, merged


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_episodes(spec):
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--scene-root", required=True, type=Path,
                   help="dir holding ep_NN/poses/*.json (poses are read, patch/config written here)")
    p.add_argument("--episodes", required=True, help="'0-34' or '0,2,5'")
    p.add_argument("--base", type=Path, default=None,
                   help="override base config path (default: dataset's registered base)")
    p.add_argument("--patch-name", default="patch.json")
    p.add_argument("--config-name", default="config.json")
    p.add_argument("--no-merged", action="store_true", help="emit only the patch, skip merged config")
    args = p.parse_args(argv)

    ds = DATASETS[args.dataset]
    base_path = args.base if args.base else (PROJECT_ROOT / ds["base"])
    base = json.loads(Path(base_path).read_text())
    print(f"[emit] dataset={args.dataset}  base={base_path}")

    written, skipped = 0, []
    for ep in parse_episodes(args.episodes):
        res = emit_episode(args.dataset, args.scene_root, ep, base=base)
        if res is None:
            skipped.append(ep)
            continue
        patch, merged = res
        ep_dir = args.scene_root / f"ep_{ep:02d}"
        (ep_dir / args.patch_name).write_text(json.dumps(patch, indent=2) + "\n")
        objs = list(patch.get("_r2s", {}).get("objects", {}))
        line = f"[emit] ep_{ep:02d}: patch({objs}) -> {ep_dir / args.patch_name}"
        if not args.no_merged:
            (ep_dir / args.config_name).write_text(json.dumps(merged, indent=2) + "\n")
            line += f" + {args.config_name}"
        print(line)
        written += 1

    print(f"[emit] wrote {written} episodes"
          + (f"; skipped (no poses): {skipped}" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
