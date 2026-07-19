"""Quick adapter: poses/<object>.json (multi-cam step4 output) → gsrl_config.json
for Genesis render. Bypass the older scene.json-based real2sim/export pipeline
since the multi-cam flow doesn't produce a scene.json yet.

Patches the GSRL template (example/1.json) one object slot at a time, applying
the Genesis pose-convention conversion (R_pt3d_rowvec → R_genesis_colvec = R.T,
then to wxyz quaternion).

Usage:
    python scripts/export_pose_for_genesis.py \\
        --scene-dir assets/scenes/xarm_ep110 \\
        --object blue_mug --slot mug \\
        --output assets/scenes/xarm_ep110/gsrl_config.json

Then:
    bash step5_render.sh assets/scenes/xarm_ep110/gsrl_config.json
    → render goes to assets/scenes/xarm_ep110/genesis_preview/sim_c0.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.spatial.transform import Rotation as Rscipy


def quat_wxyz(R_genesis: np.ndarray):
    xyzw = Rscipy.from_matrix(R_genesis).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def patch_object_in_env(env, slot: str, mesh_path: Path, pos, quat_wxyz_list, scale: float):
    o = env.setdefault(slot, {})
    o["asset_path"] = str(mesh_path)
    o["gs_path"] = None
    o["spawn_pos"] = [float(x) for x in pos]
    o["spawn_quat"] = list(quat_wxyz_list)
    o.setdefault("entity_kwargs", {}).setdefault("morph_kwargs", {})
    morph = o["entity_kwargs"]["morph_kwargs"]
    morph["file"] = str(mesh_path)
    morph["scale"] = float(scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True, type=Path)
    p.add_argument("--object", required=True,
                   help="Looks up <scene-dir>/poses/<object>.json")
    p.add_argument("--slot", default=None,
                   help="GSRL template slot to patch (default: same as --object). "
                        "Common slots: 'mug', 'tree'.")
    p.add_argument("--template", default=str(PROJECT_ROOT / "example" / "1.json"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workspace-dir", default=None,
                   help="Override env.gs_render.workspace_dir (where Genesis writes "
                        "intermediate artifacts). Default: <output_parent>/gs_workspace/")
    p.add_argument("--include-default-tree", action="store_true",
                   help="Keep the template's default tree object so the render still "
                        "shows a tree alongside the mug (useful when only the mug "
                        "pose is optimized).")
    args = p.parse_args()

    pose_p = args.scene_dir / "poses" / f"{args.object}.json"
    if not pose_p.exists():
        print(f"[export] [fatal] pose JSON not found: {pose_p}")
        sys.exit(1)
    pose = json.loads(pose_p.read_text())

    R = np.array(pose["R"], dtype=np.float64)
    t = np.array(pose["t"], dtype=np.float64)
    s = float(pose["scale"])
    mesh_path = Path(pose["mesh"])
    if not mesh_path.exists():
        print(f"[export] [fatal] mesh file not found: {mesh_path}")
        sys.exit(1)

    # PT3D row-vec → Genesis col-vec
    R_genesis = R.T.copy()
    quat = quat_wxyz(R_genesis)

    print(f"[export] pose source: {pose_p}")
    print(f"[export] step4 final_iou = {pose.get('final_iou', '?'):.4f}" if isinstance(pose.get('final_iou'), (int, float)) else f"[export] step4 final_iou = {pose.get('final_iou', '?')}")
    print(f"[export] mesh = {mesh_path}")
    print(f"[export] spawn_pos  = {t.tolist()}")
    print(f"[export] spawn_quat = (w,x,y,z) {quat}")
    print(f"[export] scale = {s:.4f}")

    template = json.loads(Path(args.template).read_text())
    env = template["env"]

    slot = args.slot if args.slot is not None else args.object
    patch_object_in_env(env, slot, mesh_path, t, quat, s)
    print(f"[export] patched slot '{slot}'")

    # If we only ran step4 on the mug (no tree pose), the template still has the
    # tree slot pointing at some default mesh. Optionally remove it so the
    # render is purely "mug on plane" — useful for plane_z diagnostics.
    if not args.include_default_tree and "tree" in env and slot != "tree":
        env.pop("tree", None)
        print(f"[export] dropped 'tree' slot (no pose for it; --include-default-tree to keep)")

    # workspace_dir — needed by step5_render.sh (it reads this from the config)
    if args.workspace_dir is None:
        ws = args.output.parent / "gs_workspace"
    else:
        ws = Path(args.workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    env.setdefault("gs_render", {})["workspace_dir"] = str(ws)
    print(f"[export] gs_render.workspace_dir = {ws}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2))
    print(f"\n[export] written → {args.output}")
    print(f"\n[export] next:  bash step5_render.sh {args.output}")
    print(f"        → render: {args.output.parent}/genesis_preview/sim_c0.png")


if __name__ == "__main__":
    main()
