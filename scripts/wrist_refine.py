#!/usr/bin/env python3
"""wrist_refine.py — yaw-only pose refinement against the wrist camera(s).

Post-stage AFTER step4: takes an existing `poses/<object>.json` (the ego fit, which
pins t/scale + a rough yaw), FREEZES (t, scale), and re-fits ONLY yaw against the
wrist masks — the wrist sees the object up-close from changing angles as the arm
approaches, which resolves orientation (mug handle, arch opening) far better than the
side-view ego silhouette. Full-circle multi-start + DT contour loss (not area-IoU).

Reads the SAME masks that step3 already wrote under perception/cam_<wrist>/. Writes a
new `poses/<object>_wrist.json` (or --inplace updates `<object>.json`) with the refined
R (mesh_prerot folded back in, so it's drop-in for the exporter), same t/scale.

Usage (sam3d-objects env, GPU):
    python scripts/wrist_refine.py \
        --scene-dir outputs/xarm_hang_blue_mug_v0/ep_00 \
        --object blue_mug --mesh <mesh.glb> \
        --only-cams wrist --yaw-starts 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for step4 helpers

from real2sim.io.views import build_views_for_object
from real2sim.perception.mesh_io import load_glb_as_pytorch3d
from real2sim.pose.wrist_refine import refine_yaw_wrist
from step4_multicam_pose import render_debug_overlays   # reuse QA overlay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--object", required=True, help="reads poses/<object>.json")
    p.add_argument("--label", default=None, help="SAM label (default: --object with _→space)")
    p.add_argument("--mesh", default=None, help="mesh.glb (default: from the pose json)")
    p.add_argument("--only-cams", default="wrist",
                   help="comma list of WRIST cam names (default 'wrist'; rm65b: "
                        "'wrist_right' or 'wrist_left,wrist_right')")
    # frame selection — defaults reuse whatever step3 segmented; override as needed.
    p.add_argument("--score-thr", type=float, default=0.0)
    p.add_argument("--max-per-cam", type=int, default=None)
    p.add_argument("--frame-range", default=None, metavar="LO:HI")
    p.add_argument("--instance", default=None,
                   choices=["left", "right", "top", "bottom", "largest", "smallest"])
    p.add_argument("--mask-dilate", type=int, default=0)
    p.add_argument("--cam-balance", action="store_true",
                   help="weight each view by 1/(#views from its camera) — balances "
                        "e.g. wrist_left vs wrist_right (rm65b) so a cam with more "
                        "frames doesn't dominate the yaw fit.")
    # refine knobs
    p.add_argument("--yaw-starts", type=int, default=12)
    p.add_argument("--coarse-iters", type=int, default=40)
    p.add_argument("--fine-iters", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--loss", default="tversky", choices=["tversky", "dt"],
                   help="yaw objective. 'tversky' (default): asymmetric soft-IoU "
                        "(α on green-outside-mask) — pins the handle/opening direction "
                        "and has a clean peak even on near-symmetric bodies. 'dt': "
                        "contour distance-transform — shape-sensitive but near-FLAT and "
                        "MISLEADING on symmetric mugs (verified to flip them).")
    p.add_argument("--iou-fp-weight", type=float, default=3.0,
                   help="Tversky α: penalty on rendered pixels OUTSIDE the mask (>β "
                        "pins handle direction). Default 3.0.")
    p.add_argument("--iou-fn-weight", type=float, default=0.5,
                   help="Tversky β: penalty on mask not covered. Default 0.5.")
    p.add_argument("--dt-w-out", type=float, default=1.0)
    p.add_argument("--dt-w-in", type=float, default=1.0)
    p.add_argument("--inplace", action="store_true",
                   help="overwrite poses/<object>.json instead of writing <object>_wrist.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    device = torch.device(args.device)

    # ── 1. load the ego pose (freeze t/scale; init yaw from it) ──
    pose_p = scene_dir / "poses" / f"{args.object}.json"
    if not pose_p.exists():
        raise SystemExit(f"[wrist] no ego pose to refine: {pose_p}")
    pose = json.loads(pose_p.read_text())
    # SAM label: --label > the pose json's saved label > object slug. (The object
    # SLUG, e.g. 'mug_clean_mid', is just the output name, NOT the SAM label.)
    label = args.label or pose.get("label") or args.object.replace("_", " ")
    R_full = np.asarray(pose["R"], dtype=np.float64)
    t = np.asarray(pose["t"], dtype=np.float64)
    scale = float(pose["scale"])
    mesh_path = Path(args.mesh) if args.mesh else Path(pose["mesh"])
    mesh_prerot = np.asarray(pose.get("mesh_prerot", np.eye(3)), dtype=np.float64)
    # init yaw: from the pure-yaw part (R_fit_yaw if the pose carried a laydown fold,
    # else from R_full assuming it's already a pure yaw).
    R_yaw0 = np.asarray(pose.get("R_fit_yaw", R_full), dtype=np.float64)
    init_yaw = math.atan2(R_yaw0[0, 1], R_yaw0[0, 0])
    print(f"[wrist] ego pose: scale={scale:.4f} t={[round(x,3) for x in t]} "
          f"init_yaw={math.degrees(init_yaw)%360:.1f}°  mesh={mesh_path.name}")

    # ── 2. mesh: load + apply the SAME pre-rotation the saved R was folded with ──
    verts, faces, _ = load_glb_as_pytorch3d(mesh_path, device=device)
    if not np.allclose(mesh_prerot, np.eye(3)):
        verts = verts @ torch.from_numpy(mesh_prerot).float().to(device)
        print(f"[wrist] applied mesh_prerot (laydown) before yaw refine")

    # ── 3. wrist views (multi-frame, object static + camera moving) ──
    only_cams = [c.strip() for c in args.only_cams.split(",")]
    fmin = fmax = None
    if args.frame_range:
        lo, _, hi = args.frame_range.partition(":")
        fmin = int(lo) if lo.strip() else None
        fmax = int(hi) if hi.strip() else None
    views = build_views_for_object(
        scene_dir, label, only_cams=only_cams,
        score_thr=args.score_thr, max_per_cam=args.max_per_cam,
        frame_min=fmin, frame_max=fmax,
        instance_select=args.instance, dilate_px=args.mask_dilate,
        cam_balance=args.cam_balance,
    )
    print(f"[wrist] {len(views)} wrist view(s): {[v['name'] for v in views]}")

    # ── 4. refine yaw ──
    res = refine_yaw_wrist(
        verts, faces, views,
        init_yaw=init_yaw, t=t, scale=scale, device=device,
        n_starts=args.yaw_starts, coarse_iters=args.coarse_iters,
        fine_iters=args.fine_iters, lr=args.lr,
        loss_kind=args.loss, a_fp=args.iou_fp_weight, b_fn=args.iou_fn_weight,
        dt_w_out=args.dt_w_out, dt_w_in=args.dt_w_in, verbose=True,
    )

    # ── 5. fold mesh_prerot back in and save ──
    R_out = mesh_prerot @ res["R_yaw"]
    out = dict(pose)   # start from the ego pose, override R + provenance
    out["R"] = R_out.tolist()
    out["R_fit_yaw"] = res["R_yaw"].tolist()
    out["mesh_prerot"] = mesh_prerot.tolist()
    out["wrist_refined"] = True
    out["wrist_yaw_deg"] = res["yaw_deg"]
    out["wrist_init_yaw_deg"] = res["init_yaw_deg"]
    out["wrist_delta_deg"] = res["delta_deg"]
    out["wrist_loss"] = res["loss"]
    out["wrist_loss_kind"] = res["loss_kind"]
    out["wrist_views"] = [v["name"] for v in views]
    out["wrist_per_view_iou"] = {p["name"]: p["iou"] for p in res["per_view_iou"]}
    out["wrist_all_starts"] = res["all_starts"]

    out_p = pose_p if args.inplace else scene_dir / "poses" / f"{args.object}_wrist.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\n[wrist] yaw {res['init_yaw_deg']:.1f}° → {res['yaw_deg']:.1f}° "
          f"(Δ {res['delta_deg']:+.1f}°)   {res['loss_kind']}_loss={res['loss']:.4f}")
    print(f"[wrist] saved → {out_p}")

    # ── 6. QA overlays on the wrist frames (red=mask, green=render at refined yaw) ──
    render_debug_overlays(verts, faces, views,
                          torch.from_numpy(res["R_yaw"]).float(),
                          torch.as_tensor(t).float(), scale,
                          device, scene_dir, f"{args.object}_wrist")


if __name__ == "__main__":
    main()
