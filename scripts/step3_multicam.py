#!/usr/bin/env python3
"""step3_multicam.py — run SAM3 segmentation and MoGe pointmap on a multi-cam
scene_dir produced by scripts/extract_lerobot.py.

Two modes, each requires its own env:

    --mode seg       (env: sam3)         run SAM3 per (cam, frame) pair
    --mode pointmap  (env: sam3d-objects) run MoGe on one reference frame

Output layout (under <scene_dir>/perception/):

    cam_<name>/frame_<NNNNNN>/
        masks.pt        # {masks: (N,H,W) bool, boxes, scores, labels, prompt, image_path}
        overlay.png     # boxes + labels on top of the source frame
        cutout.png      # source · union(masks); background blacked out (sanity check)
        cutout_<i>_<label>.png  # source · mask_i; one per detected object
    pointmap/
        pointmap.npy    # (H, W, 3) float32, PyTorch3D cam frame (metric)
        intrinsics.npy  # 3x3 K used (user-supplied from cameras.json["ego"])
        ref.json        # which (cam, frame) was used as the MoGe reference

Defaults per the chosen sampling policy:
    fixed cams (e.g. ego):    [0]                 (one frame, cam doesn't move)
    attached cams (e.g. wrist): [0,20,40,60,80]   (5 evenly-spaced, see options
                                                   per cam below or --wrist-frames)

Override examples:
    python scripts/step3_multicam.py --mode seg \\
        --scene-dir assets/scenes/xarm_ep110 \\
        --prompt "blue mug and mug tree" \\
        --wrist-frames 0,15,30,45,60,75

    python scripts/step3_multicam.py --mode pointmap \\
        --scene-dir assets/scenes/xarm_ep110 \\
        --moge-ref ego/0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────
# Frame-selection helpers
# ─────────────────────────────────────────────────────────────────────

def parse_frame_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s == "":
        return None
    return [int(t) for t in s.split(",") if t.strip()]


def default_frames(cam_kind: str, T: int) -> List[int]:
    """Defaults per the chosen sampling policy (see module docstring)."""
    if cam_kind == "fixed":
        return [0]
    if cam_kind == "attached":
        if T <= 1:
            return [0]
        K = min(5, T)
        # K evenly-spaced indices in [0, T-1]; first sample at 0 to align with
        # the MoGe reference frame.
        step = (T - 1) / (K - 1)
        return sorted({int(round(i * step)) for i in range(K)})
    raise ValueError(f"unknown cam kind: {cam_kind!r}")


def resolve_frames_per_cam(scene_dir: Path,
                           ego_frames: Optional[List[int]],
                           wrist_frames: Optional[List[int]],
                           every: Optional[int] = None) -> Dict[str, Tuple[str, List[int]]]:
    """Return {cam_name: (kind, [frame_indices])} for the cams in cameras.json
    that have any frames on disk.

    `wrist_frames` applies to ALL attached cams (wrist_left, wrist_right, ...),
    not just one named "wrist". `every`, if set, segments every K-th frame on
    every cam (dense pass for confidence-gated selection downstream)."""
    cams = json.loads((scene_dir / "cameras.json").read_text())["cameras"]
    traj = json.loads((scene_dir / "robot_traj.json").read_text())
    # frame count, robust to robot_traj format: FK path stores link_poses_world
    # (no ee_pose_world); both formats carry joint_angles.
    T_traj = (len(traj.get("joint_angles") or [])
              or len(traj.get("ee_pose_world") or [])
              or (len(next(iter(traj["link_poses_world"].values())))
                  if traj.get("link_poses_world") else 0))

    out = {}
    for c in cams:
        name, kind = c["name"], c["kind"]
        frames_dir = scene_dir / f"cam_{name}" / "frames"
        if not frames_dir.exists():
            print(f"[step3] skip cam {name!r}: no {frames_dir}")
            continue
        T_disk = len(list(frames_dir.glob("*.png")))
        T_use = min(T_traj, T_disk)
        if ego_frames is not None and (name == "ego" or kind == "fixed"):
            frames = ego_frames
        elif wrist_frames is not None and kind == "attached":
            frames = wrist_frames
        elif every is not None and every > 0:
            frames = list(range(0, T_use, every))
        else:
            frames = default_frames(kind, T_use)
        # filter to frames that actually exist on disk
        frames = [t for t in frames if (frames_dir / f"{t:06d}.png").exists()]
        if not frames:
            print(f"[step3] skip cam {name!r}: no requested frames on disk")
            continue
        out[name] = (kind, frames)
    return out


# ─────────────────────────────────────────────────────────────────────
# Mode: seg  (env: sam3)
# ─────────────────────────────────────────────────────────────────────

_LABEL_SAFE = str.maketrans({" ": "_", "/": "_", "\\": "_", ":": "_"})


def _save_cutouts(image, masks_bool, labels, out_dir: Path) -> None:
    """Write `cutout.png` (union) and `cutout_<i>_<label>.png` (per-object).

    Pixels outside the mask are zeroed out so you can eyeball whether the mask
    actually covers the object you wanted (vs. background bleed / wrong instance).
    """
    import numpy as np
    from PIL import Image

    arr = np.array(image)                              # (H, W, 3) uint8
    masks_np = masks_bool.numpy().astype(bool)          # (N, H, W)
    if masks_np.size == 0 or masks_np.shape[0] == 0:
        return
    union = masks_np.any(axis=0)                        # (H, W)
    Image.fromarray((arr * union[..., None]).astype(np.uint8)).save(out_dir / "cutout.png")
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        cut = (arr * m[..., None]).astype(np.uint8)
        label = labels[i].translate(_LABEL_SAFE) if labels else f"obj{i}"
        Image.fromarray(cut).save(out_dir / f"cutout_{i}_{label}.png")


def run_seg(scene_dir: Path, prompt: str,
            frame_plan: Dict[str, Tuple[str, List[int]]],
            force: bool, box_threshold: float = 0.25, max_det: int = 5) -> None:
    import torch
    from PIL import Image, ImageDraw
    from real2sim.perception.segmentation import GroundedSAM

    print(f"[seg] loading SAM3 (box_threshold={box_threshold}, max_det={max_det}) ...")
    sam = GroundedSAM(device="cuda" if torch.cuda.is_available() else "cpu",
                      box_threshold=box_threshold, max_det=max_det)

    perception = scene_dir / "perception"
    for cam_name, (kind, frames) in frame_plan.items():
        print(f"\n[seg] cam {cam_name!r} ({kind}): {len(frames)} frame(s) {frames}")
        for t in frames:
            img_p = scene_dir / f"cam_{cam_name}" / "frames" / f"{t:06d}.png"
            out_dir = perception / f"cam_{cam_name}" / f"frame_{t:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            masks_p = out_dir / "masks.pt"
            overlay_p = out_dir / "overlay.png"
            if not force and masks_p.exists() and overlay_p.exists():
                print(f"  t={t:>3}  cache hit ({masks_p})")
                continue
            image = Image.open(img_p).convert("RGB")
            seg = sam.segment(image, prompt, expand_mask_iters=0, expand_kernel=15)
            n = int(seg.masks.shape[0])
            labels_str = ", ".join(f"'{seg.labels[i]}' ({float(seg.scores[i]):.2f})"
                                   for i in range(n))
            print(f"  t={t:>3}  → {n} object(s): {labels_str if n else '(none)'}")
            torch.save(
                {
                    "image_path": str(img_p),
                    "prompt": prompt,
                    "masks": seg.masks.cpu().bool(),
                    "boxes": seg.boxes.cpu(),
                    "scores": seg.scores.cpu(),
                    "labels": seg.labels,
                },
                masks_p,
            )
            vis = image.copy()
            draw = ImageDraw.Draw(vis)
            for i, b in enumerate(seg.boxes.tolist()):
                draw.rectangle(b, outline=(255, 0, 0), width=2)
                draw.text((b[0] + 3, b[1] + 3), seg.labels[i], fill=(255, 0, 0))
            vis.save(overlay_p)
            _save_cutouts(image, seg.masks.cpu().bool(), seg.labels, out_dir)
    print(f"\n[seg] DONE → {perception}")


# ─────────────────────────────────────────────────────────────────────
# Mode: pointmap (env: sam3d-objects)
# ─────────────────────────────────────────────────────────────────────

def parse_cam_frame_ref(s: str) -> Tuple[str, int]:
    """Parse 'ego/0' → ('ego', 0)."""
    if "/" not in s:
        raise ValueError(f"--moge-ref must be 'cam_name/frame_idx', got {s!r}")
    name, t = s.split("/", 1)
    return name, int(t)


def run_pointmap(scene_dir: Path, ref: str, force: bool) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v1 import MoGeModel

    from real2sim.io.cameras import CameraSet

    cam_name, t = parse_cam_frame_ref(ref)
    img_p = scene_dir / f"cam_{cam_name}" / "frames" / f"{t:06d}.png"
    if not img_p.exists():
        raise SystemExit(f"[pointmap] no such frame: {img_p}")

    cam = CameraSet.load(scene_dir / "cameras.json").by_name(cam_name)
    out_dir = scene_dir / "perception" / "pointmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    pm_p = out_dir / "pointmap.npy"
    K_p = out_dir / "intrinsics.npy"
    ref_info_p = out_dir / "ref.json"

    if not force and pm_p.exists() and K_p.exists():
        print(f"[pointmap] cache hit: {pm_p} + {K_p} (use --force to redo)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_p).convert("RGB")
    W, H = image.size
    fx, fy, cx, cy = cam.intrinsics(W=W, H=H)
    fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    print(f"[pointmap] ref = {ref}  {W}x{H}  fx={fx:.2f} → fov_x={fov_x_deg:.2f}°")

    print("[pointmap] loading MoGe (Ruicheng/moge-vitl) ...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    img_t = (torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0).to(device)
    with torch.no_grad():
        out = moge.infer(img_t, fov_x=fov_x_deg)

    # MoGe → PT3D cam frame: (x, y, z) → (-x, -y, z). Downstream uses z only,
    # but flip x/y to keep the saved pointmap consistent with PT3D convention.
    points = out["points"].clone()
    points[..., 0] = -out["points"][..., 0]
    points[..., 1] = -out["points"][..., 1]
    np.save(pm_p, points.cpu().numpy().astype(np.float32))

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.save(K_p, K)
    print(f"[pointmap] saved {pm_p}  shape={tuple(points.shape)}  z=[{float(points[...,2].min()):.3f},{float(points[...,2].max()):.3f}]")
    print(f"[pointmap] saved {K_p}  (from cameras.json[{cam_name!r}])")

    ref_info_p.write_text(json.dumps({"cam_name": cam_name, "frame": t,
                                       "image_path": str(img_p)}, indent=2))


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["seg", "pointmap"])
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--prompt", help="SAM3 text prompt (required for --mode seg)")
    p.add_argument("--ego-frames", default=None,
                   help="comma-separated frame indices for ego cam (default: [0])")
    p.add_argument("--wrist-frames", default=None,
                   help="comma-separated frame indices for ALL attached (wrist) cams "
                        "(default: 5 evenly-spaced in trajectory)")
    p.add_argument("--every", type=int, default=None,
                   help="dense pass: segment every K-th frame on every cam "
                        "(for confidence-gated frame selection in step4). "
                        "Overridden per-cam by --ego-frames / --wrist-frames.")
    p.add_argument("--box-threshold", type=float, default=0.25,
                   help="min SAM3 detection score to keep a mask. Lower it (e.g. "
                        "0.1) for small / low-contrast objects. The seg log now "
                        "prints each prompt's raw best score so you can tune this.")
    p.add_argument("--max-det", type=int, default=5,
                   help="max distinct instances kept per prompt (SAM3 is a concept "
                        "segmenter — one prompt can return several objects, e.g. "
                        "two same-colour blocks). Default 5.")
    p.add_argument("--moge-ref", default="ego/0",
                   help="cam_name/frame_idx for the single MoGe pass (default ego/0)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if outputs exist")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not (scene_dir / "cameras.json").exists():
        raise SystemExit(f"missing {scene_dir / 'cameras.json'}")

    if args.mode == "seg":
        if not args.prompt:
            raise SystemExit("--prompt is required for --mode seg")
        plan = resolve_frames_per_cam(
            scene_dir,
            ego_frames=parse_frame_list(args.ego_frames),
            wrist_frames=parse_frame_list(args.wrist_frames),
            every=args.every,
        )
        if not plan:
            raise SystemExit("no cams with frames on disk")
        print(f"[step3] scene = {scene_dir}")
        print(f"[step3] prompt = '{args.prompt}'")
        for name, (kind, fs) in plan.items():
            print(f"[step3]   {name} ({kind}): frames {fs}")
        run_seg(scene_dir, args.prompt, plan, force=args.force,
                box_threshold=args.box_threshold, max_det=args.max_det)
    elif args.mode == "pointmap":
        run_pointmap(scene_dir, args.moge_ref, force=args.force)


if __name__ == "__main__":
    main()
