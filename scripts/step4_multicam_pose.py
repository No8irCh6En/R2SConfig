#!/usr/bin/env python3
"""step4_multicam_pose.py — multi-cam multi-frame pose refinement.

Given:
    <scene_dir>/perception/cam_*/frame_*/masks.pt   (from step3_multicam.py --mode seg)
    <scene_dir>/perception/pointmap/*.npy            (from step3_multicam.py --mode pointmap)
    <scene_dir>/cameras.json, robot_traj.json        (from extract_lerobot.py)
    a SAM3D-reconstructed mesh.glb for the object

run `PoseOptimizer.optimize_multi_view()` with the object's masks across all
(cam, frame) pairs that match a SAM label. Single shared per-object (R, t, s)
is jointly fit against every view; the MoGe pointmap depth supervises the one
view it was computed on.

Mesh convention reminder:
    `load_glb_as_pytorch3d` applies M_LOAD (y-up → z-up), so the loaded mesh
    is **canonical z-up**. The optimizer then learns mesh→world (z-up) (R, t, s).

Output:
    <scene_dir>/poses/<object>.json   { R: 3x3, t: 3, scale: 1, final_iou: ... }

Usage (sam3d-objects env, GPU required):
    python scripts/step4_multicam_pose.py \\
        --scene-dir assets/scenes/xarm_ep110 \\
        --object blue_mug \\
        --label "blue mug" \\
        --mesh outputs/build/objects/blue_mug__878169e6/mesh.glb \\
        --plane-z 0.0 \\
        --floor-lock
"""
from __future__ import annotations

import os as _os
import time as _time
_T_IMPORT_START = _time.time()

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.io.dataset import MultiCamBundle
from real2sim.io.views import build_views_for_object
from real2sim.perception.mesh_io import load_glb_as_pytorch3d
from real2sim.pose.optimizer import PoseOptimizer, _make_renderers
from real2sim.config import OptimizerConfig

# ── optional per-phase timing (set STEP4_TIMING=1) ───────────────────
_STEP4_TIMING = bool(_os.environ.get("STEP4_TIMING"))
_t_prev = _T_IMPORT_START
def _lap(label: str):
    """With STEP4_TIMING set, print wall-time (GPU-synced) since the last _lap()."""
    global _t_prev
    if not _STEP4_TIMING:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    now = _time.time()
    print(f"  [T4] {label:<18} = {now - _t_prev:6.1f}s", flush=True)
    _t_prev = now


def estimate_init_scale(verts: torch.Tensor, faces: torch.Tensor, views: list,
                        init_R: torch.Tensor, init_t: torch.Tensor,
                        device: torch.device, max_iters: int = 8) -> float:
    """Iterated area-ratio scale init using the largest-mask view.

    Same logic as the optimizer's Stage 2, but run BEFORE Stage 1 with init R/t
    fixed. Converges in ~5 iters from s=1.0 to the s where the rendered area
    matches the target mask area at the chosen view.

    Why: Stage 1 freezes scale during R/t learning. If init_scale is even 2x
    off, the rendered silhouette either saturates the frame (no IoU gradient)
    or stays a dot inside the mask — either way R/t doesn't converge. And
    Stage 2 only fixes scale, not R/t — so a bad Stage 1 stays bad.
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras as _PerspCam

    largest = max(views, key=lambda v: int(v["mask"].bool().sum()))
    h, w = int(largest["H"]), int(largest["W"])
    sil_r, _, _ = _make_renderers((h, w), sigma=1e-4, device=device,
                                  faces_per_pixel=50)
    cams = _PerspCam(
        focal_length=((float(largest["fx"]), float(largest["fy"])),),
        principal_point=((float(largest["cx"]), float(largest["cy"])),),
        image_size=((h, w),),
        R=largest["R_view"][None].to(device).float(),
        T=largest["T_view"][None].to(device).float(),
        device=device, in_ndc=False,
    )

    target_area = float(largest["mask"].bool().sum())
    verts_d = verts.to(device); faces_d = faces.to(device)
    R_d = init_R.to(device); t_d = init_t.to(device)

    s = 1.0
    print(f"[init] scale iter @ '{largest['name']}' (target={int(target_area)} px):")
    for it in range(max_iters):
        with torch.no_grad():
            mesh_world = verts_d * s @ R_d + t_d
            meshes = Meshes(verts=[mesh_world], faces=[faces_d])
            alpha = sil_r(meshes, cameras=cams)[..., 3].squeeze()
            pred_area = float((alpha > 0.5).float().sum())
        if pred_area < 1.0:
            print(f"  iter {it}: s={s:.4f}  pred=0 (off-screen) — stopping")
            return max(s, 0.01)
        ratio = (target_area / pred_area) ** 0.5
        s_new = s * ratio
        print(f"  iter {it}: s {s:.4f} → {s_new:.4f}  (pred={int(pred_area)})")
        if abs(s_new - s) / max(s, 1e-8) < 5e-3:
            return s_new
        s = s_new
    return s


def render_debug_overlays(verts: torch.Tensor, faces: torch.Tensor, views: list,
                          R: torch.Tensor, t: torch.Tensor, scale: float,
                          device: torch.device, scene_dir: Path, object_name: str,
                          ) -> None:
    """Per-view overlay: source frame (gray) + target mask (red) + rendered silhouette (green).

    Writes to <scene_dir>/_step4_debug/<object>/<view>.png. Lets you eyeball
    *where* the optimization failed: shape mismatch vs. pure translation drift.
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras as _PerspCam
    from PIL import Image

    out_dir = scene_dir / "_step4_debug" / object_name
    out_dir.mkdir(parents=True, exist_ok=True)
    verts_d = verts.to(device); faces_d = faces.to(device)
    R_d = R.to(device); t_d = t.to(device)
    mesh_world = verts_d * float(scale) @ R_d + t_d
    meshes = Meshes(verts=[mesh_world], faces=[faces_d])

    print(f"\n[debug] writing render overlays → {out_dir}")
    for v in views:
        h, w = int(v["H"]), int(v["W"])
        _, _, rast_r = _make_renderers((h, w), sigma=1e-4, device=device, faces_per_pixel=50)
        cams = _PerspCam(
            focal_length=((float(v["fx"]), float(v["fy"])),),
            principal_point=((float(v["cx"]), float(v["cy"])),),
            image_size=((h, w),),
            R=v["R_view"][None].to(device).float(),
            T=v["T_view"][None].to(device).float(),
            device=device, in_ndc=False,
        )
        with torch.no_grad():
            # Hard coverage via the nearest-face rasterizer (faces_per_pixel=1,
            # blur_radius=0): pix_to_face >= 0 ⇔ a face covers this pixel center.
            # For a watertight mesh this is a solid, hole-free silhouette. The soft
            # alpha instead aggregates the 50-nearest-by-depth faces with sigma=1e-4,
            # which on this dense/noisy SAM3D mesh evicts the true covering face and
            # leaves speckled interior holes. QA-ONLY — the loss still uses soft alpha.
            frags = rast_r(meshes, cameras=cams)
            rnd = (frags.pix_to_face[..., 0].squeeze().cpu().numpy() >= 0)

        cam_name, frame_str = v["name"].split("/")
        src_p = scene_dir / f"cam_{cam_name}" / "frames" / f"{frame_str}.png"
        if not src_p.exists():
            continue
        bg = np.array(Image.open(src_p).convert("L"))  # gray
        rgb = np.stack([bg, bg, bg], axis=-1).astype(np.float32)
        tgt = (v["mask"].cpu().numpy() > 0.5)
        rgb[tgt] = 0.5 * rgb[tgt] + np.array([128, 0, 0])      # red = target
        rgb[rnd] = 0.5 * rgb[rnd] + np.array([0, 128, 0])      # green = rendered
        Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(out_dir / f"{cam_name}_{frame_str}.png")
        inter = int((tgt & rnd).sum())
        union = int((tgt | rnd).sum()) or 1
        print(f"  {v['name']:>14}  tgt={int(tgt.sum())}  rnd={int(rnd.sum())}  "
              f"inter={inter}  iou={inter/union:.3f}")


def init_t_from_ego_pointmap(scene_dir: Path, views: list) -> torch.Tensor:
    """Backproject the ego-mask centroid through the MoGe pointmap to world.

    Less reliable than triangulate_init_t_from_masks() because MoGe metric
    depth can be off by 10-30%. Used as fallback when no view has a mask.
    """
    pm_p = scene_dir / "perception" / "pointmap" / "pointmap.npy"
    ref_p = scene_dir / "perception" / "pointmap" / "ref.json"
    if not pm_p.exists() or not ref_p.exists():
        return torch.tensor([0.0, 0.0, 1.0])
    ref = json.loads(ref_p.read_text())
    ref_view = next(
        (v for v in views if v["name"] == f"{ref['cam_name']}/{int(ref['frame']):06d}"),
        None,
    )
    if ref_view is None:
        return torch.tensor([0.0, 0.0, 1.0])
    mask = ref_view["mask"].cpu().numpy() > 0.5
    if mask.sum() < 10:
        return torch.tensor([0.0, 0.0, 1.0])
    pm = np.load(pm_p)
    if pm.shape[:2] != mask.shape:
        return torch.tensor([0.0, 0.0, 1.0])
    cam_centroid = np.median(pm[mask], axis=0)
    R = ref_view["R_view"].cpu().numpy()
    T = ref_view["T_view"].cpu().numpy()
    world_centroid = (cam_centroid - T) @ R.T
    return torch.from_numpy(world_centroid).float()


def triangulate_init_t_from_masks(views: list) -> torch.Tensor:
    """Multi-view triangulation: world point closest to all back-projected rays.

    For each view, the mask centroid pixel defines a ray from the camera center
    through that pixel. The optimal init_t is the world point minimizing the
    sum of squared perpendicular distances to those rays.

    Why this beats ego-pointmap backprojection: it uses ALL views (including
    the 5 wrist frames) and doesn't depend on MoGe's metric depth, which can
    be off by 25cm. With 6 views, the over-determined linear system averages
    out per-view noise.

    PT3D pinhole reminder (matches sanity_project.py):
        u = -fx * P_view[0] / P_view[2] + cx
    so for a unit-depth ray, P_view = (-(u-cx)/fx, -(v-cy)/fy, 1).
    World eye = -T @ R.T;  world dir = P_view @ R.T (no translation, dir-only).

    Solver: minimize ∑ᵢ ||(p - eyeᵢ) - (dᵢᵀ(p - eyeᵢ)) dᵢ||²
    Closed form:  (∑ᵢ (I - dᵢdᵢᵀ)) p = ∑ᵢ (I - dᵢdᵢᵀ) eyeᵢ
    """
    eyes = []
    dirs = []
    for v in views:
        mask = v["mask"].cpu().numpy() > 0.5
        if mask.sum() < 10:
            continue
        ys, xs = np.where(mask)
        u_c, v_c = float(xs.mean()), float(ys.mean())
        fx, fy = float(v["fx"]), float(v["fy"])
        cx, cy = float(v["cx"]), float(v["cy"])
        P_view = np.array([-(u_c - cx) / fx, -(v_c - cy) / fy, 1.0])
        R = v["R_view"].cpu().numpy()              # world→view, row-vec
        T = v["T_view"].cpu().numpy()              # (3,)
        eye = -T @ R.T                             # camera center in world
        d = P_view @ R.T                           # ray dir in world
        d = d / (np.linalg.norm(d) + 1e-12)
        eyes.append(eye); dirs.append(d)
    if not eyes:
        print("[init] no usable mask centroids; triangulation skipped")
        return None

    M = np.zeros((3, 3))
    b = np.zeros(3)
    for eye, d in zip(eyes, dirs):
        proj = np.eye(3) - np.outer(d, d)          # I - d dᵀ
        M += proj
        b += proj @ eye
    # Rank check: fixing depth needs >=2 non-parallel rays. With a single view
    # (or near-parallel rays) M = Σ(I - d dᵀ) is rank-deficient along the shared
    # ray direction. np.linalg.solve does NOT raise on it — it returns a
    # depth-degenerate point (~the camera center), planting the mesh on top of
    # the camera. Detect via the smallest singular value and bail so the caller
    # can fall back to floor-plane intersection.
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[-1] < 1e-6 * max(sv[0], 1e-12):
        print(f"[init] triangulation rank-deficient (singular values "
              f"{sv.round(4).tolist()}, {len(eyes)} usable view(s)); "
              f"need >=2 non-parallel rays — falling back")
        return None
    try:
        p = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        print("[init] triangulation singular; falling back")
        return None
    # Residual: avg perpendicular distance per view (sanity metric)
    dists = []
    for eye, d in zip(eyes, dirs):
        v = p - eye
        dists.append(float(np.linalg.norm(v - (v @ d) * d)))
    print(f"[init] triangulated init_t from {len(eyes)} views: "
          f"world={p.round(3).tolist()}  (avg ray-dist={np.mean(dists)*100:.1f}cm)")
    return torch.from_numpy(p).float()


def floor_intersect_init_t(views: list, plane_z: float = 0.0) -> torch.Tensor:
    """Single-view fallback init: bottom-of-mask ray ∩ support plane z=plane_z.

    `triangulate_init_t_from_masks()` needs >=2 non-parallel rays to fix depth.
    With one view (e.g. --only-cams ego) it's rank-deficient and returns a
    depth-degenerate point near the camera center, planting the mesh on top of
    the camera (a huge, frame-filling, edge-clipped init silhouette — which then
    breaks the area-ratio scale init too). When the object rests on a known
    floor, the one depth cue a single silhouette + floor lock actually provides
    is the footprint: intersect the bottom-of-mask ray with the support plane.

    Uses the largest-mask view; takes the centroid of the bottom 10% of mask
    rows as the floor-contact pixel, and returns that point on z=plane_z. This
    is the mesh-origin xy when the mesh's vertical axis passes through its origin
    (true for the z-up canonical meshes here); the optimizer refines xy + yaw,
    and floor-lock then sets t_z = plane_z - s·z_min.

    PT3D pinhole convention matches `triangulate_init_t_from_masks()`.
    """
    usable = [v for v in views if int(v["mask"].bool().sum()) >= 10]
    if not usable:
        print("[init] floor-intersect: no usable mask; skipped")
        return None
    v = max(usable, key=lambda x: int(x["mask"].bool().sum()))
    mask = v["mask"].cpu().numpy() > 0.5
    ys, xs = np.where(mask)
    thr = np.percentile(ys, 90)                    # bottom 10% of rows = floor contact
    sel = ys >= thr
    u_b, v_b = float(xs[sel].mean()), float(ys[sel].mean())
    fx, fy = float(v["fx"]), float(v["fy"])
    cx, cy = float(v["cx"]), float(v["cy"])
    R = v["R_view"].cpu().numpy()                   # world→view, row-vec
    T = v["T_view"].cpu().numpy()                   # (3,)
    eye = -T @ R.T                                  # camera center in world
    P_view = np.array([-(u_b - cx) / fx, -(v_b - cy) / fy, 1.0])
    d = P_view @ R.T                                # ray dir in world (dir-only)
    d = d / (np.linalg.norm(d) + 1e-12)
    if abs(d[2]) < 1e-6:
        print("[init] floor-intersect: ray parallel to plane; skipped")
        return None
    lam = (plane_z - eye[2]) / d[2]
    if lam <= 0:
        print("[init] floor-intersect: plane is behind the camera; skipped")
        return None
    p = eye + lam * d
    print(f"[init] floor-intersect init_t @ '{v['name']}' (base px "
          f"{u_b:.0f},{v_b:.0f}) ∩ z={plane_z}: world={p.round(3).tolist()}")
    return torch.from_numpy(p).float()


def _yaw_R(theta_deg: float) -> torch.Tensor:
    """PT3D row-vec yaw rotation around +Z (matches optimizer._yaw_matrix_world)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_genesis_to_R_pt3d(q):
    """Genesis quaternion (w, x, y, z) → PyTorch3D row-vec rotation matrix.

    Genesis: v_col_rotated = R_genesis @ v_col. PT3D: v_row_rotated = v_row @ R_pt3d.
    So R_pt3d = R_genesis.T.
    """
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w/n, x/n, y/n, z/n
    R_genesis = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return torch.from_numpy(R_genesis.T)


def _is_pure_yaw(R: torch.Tensor, tol: float = 1e-3) -> bool:
    """Check if R is a pure z-rotation (third row+col are world ±z)."""
    r = R.detach().cpu().numpy()
    if abs(r[2, 2] - 1.0) > tol: return False
    if abs(r[0, 2]) > tol or abs(r[1, 2]) > tol: return False
    if abs(r[2, 0]) > tol or abs(r[2, 1]) > tol: return False
    return True


def _rot_rowvec(axis: str, deg: float) -> np.ndarray:
    """Row-vec rotation P so that `verts @ P` rotates points `deg` about `axis`.

    Optimizer convention is row-vec (v_world = v_mesh @ R). The column-vec matrix
    Rc (v' = Rc·v) maps to a row-vec post-multiply as P = Rc.T.
    """
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    Rc = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis.lower()]
    return Rc.T


def _laydown_rowvec(verts: torch.Tensor) -> np.ndarray:
    """Row-vec pre-rotation that tips the mesh's THINNEST bbox axis to world +Z.

    'Lay it flat': whichever loaded axis has the smallest extent becomes vertical,
    so the object rests on its largest face. Returns a proper rotation (det +1)
    P with `verts @ P` putting the chosen axis into the z slot.
    """
    ext = (verts.max(0).values - verts.min(0).values).detach().cpu().numpy()
    a = int(np.argmin(ext))
    P = {
        0: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),  # x -> z (about y)
        1: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),  # y -> z (about x)
        2: np.eye(3),                                            # already vertical
    }[a]
    return P


def _build_prerot(verts: torch.Tensor, laydown: bool,
                  mesh_prerot: Optional[str]) -> np.ndarray:
    """Compose the mesh pre-rotation (row-vec): laydown first, then manual rots.

    `mesh_prerot` is "AXIS:DEG[,AXIS:DEG...]" e.g. "y:90" or "y:90,x:180", applied
    left→right ON TOP of laydown. Total R_pre with `verts @ R_pre`.
    """
    R_pre = np.eye(3)
    if laydown:
        R_pre = R_pre @ _laydown_rowvec(verts)
    if mesh_prerot:
        for tok in mesh_prerot.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ax, _, dg = tok.partition(":")
            R_pre = R_pre @ _rot_rowvec(ax.strip(), float(dg))
    return R_pre


def main():
    _lap("import+startup")
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--object", required=True, help="output filename slug (e.g. blue_mug)")
    p.add_argument("--label", default=None,
                   help="SAM label to match (case-insensitive, fuzzy). "
                        "Default: --object with '_' → ' '.")
    p.add_argument("--mesh", required=True, help="path to mesh.glb")
    p.add_argument("--only-cams", default=None,
                   help="comma-separated cam names to use (default: all)")
    p.add_argument("--drop-frames", nargs="*", default=None, metavar="CAM/FRAME",
                   help="exclude specific views by name, e.g. --drop-frames wrist/75 "
                        "wrist/100 wrist/174 wrist/197. Frame number is "
                        "zero-pad-insensitive. Lets you test pose sensitivity to "
                        "clipped/occluded frames WITHOUT re-segmenting (masks for all "
                        "frames already exist under perception/).")
    p.add_argument("--plane-z", type=float, default=0.0,
                   help="world z of the supporting plane (m). Used by floor lock.")
    p.add_argument("--floor-lock", action="store_true",
                   help="enforce t_z = plane_z - s · z_min(mesh +Z)")
    p.add_argument("--no-yaw-only", action="store_true",
                   help="optimize full 3DOF R (default: yaw-only around +Z)")
    p.add_argument("--freeze-scale", action="store_true",
                   help="don't optimize scale (use --init-scale verbatim)")
    p.add_argument("--no-learn-scale", action="store_true",
                   help="opt out of joint R/t/s GD in Stage 1; fall back to "
                        "Stage 1 (frozen s) + Stage 2 (area-ratio). Default: on, "
                        "because the decoupled path under-scales when near and "
                        "far cameras disagree.")
    p.add_argument("--init-scale", type=float, default=None,
                   help="override init scale. Default: auto-estimate via one-shot "
                        "area-ratio at the largest-mask view.")
    p.add_argument("--init-t", default=None,
                   help="override init t as 'x,y,z' (m). Default: multi-view "
                        "triangulation from mask centroids.")
    p.add_argument("--yaw-starts", type=int, default=8,
                   help="number of evenly-spaced yaw starts (every 360/N°). "
                        "1 = no multi-start (init yaw=0 only). Default 8. "
                        "Ignored when --init-quat is given.")
    p.add_argument("--batch-yaw", dest="batch_yaw", action="store_true", default=True,
                   help="Fit ALL yaw starts in ONE batched rasterization "
                        "(optimize_multi_view_multistart) instead of a sequential "
                        "loop — same result, fills the GPU, much faster. Default ON.")
    p.add_argument("--no-batch-yaw", dest="batch_yaw", action="store_false",
                   help="Fall back to the sequential per-yaw loop (the proven path; "
                        "use to A/B-verify the batched result).")
    p.add_argument("--crop-pad", type=float, default=0.0,
                   help="Render only the object's mask bbox, padded by this fraction "
                        "of the bbox size (0 = OFF, render full frame). The object is "
                        "often <2%% of a 320×240 wrist frame, so e.g. 0.6 cuts render "
                        "pixels ~10× with an identical IoU/pose — big speedup for "
                        "stage1/stage2/tree. Object must stay in-window; 0.6–1.0 is safe "
                        "with a good init. Verify vs --crop-pad 0 before trusting.")
    p.add_argument("--yaw-keep", type=int, default=0,
                   help="Batched multi-start: 0 (default) = refine ALL yaw starts "
                        "at full res (faithful, matches the sequential result). "
                        "Set K>0 to keep only the top-K after the coarse level — a "
                        "memory saver for big masks / many starts, but the coarse "
                        "128² IoU ranking is unreliable for small/low-contrast "
                        "objects and can drop the true winner, so use sparingly.")
    p.add_argument("--init-quat", nargs=4, type=float, default=None,
                   metavar=("W", "X", "Y", "Z"),
                   help="Genesis-convention init quaternion (w x y z). When given, "
                        "skips yaw multi-start and uses this R as the single init. "
                        "Pair with --freeze-R to lock it.")
    p.add_argument("--freeze-R", action="store_true", dest="freeze_R",
                   help="Don't optimize R; only t and s are refined. Use with "
                        "--init-quat when you know the orientation approximately.")
    p.add_argument("--auto-yaw-init", action="store_true",
                   help="Derive yaw candidates from per-mask 2D asymmetry "
                        "(handle direction) instead of uniform multi-start. "
                        "Much better init for asymmetric objects (mug, kettle, "
                        "spoon). Ignored if --init-quat is given.")
    p.add_argument("--dt-loss", action="store_true",
                   help="Use distance-transform silhouette loss instead of soft-IoU. "
                        "Avoids the 'hidden feature' degenerate min that bites "
                        "asymmetric objects (mug handles get rotated behind cup body "
                        "to game pixel IoU). Pairs well with --auto-yaw-init.")
    p.add_argument("--dt-weight", type=float, default=None,
                   help="Override OptimizerConfig.stage1_dt_weight (default 2.0).")
    p.add_argument("--dt-iou-blend", type=float, default=None,
                   help="When --dt-loss is on, also add this weight × (1-tversky). "
                        "0 (default) = pure DT (scale ≈ contour match). "
                        "0.3-0.5 = boost coverage but scale gets pulled smaller.")
    p.add_argument("--iou-fp-weight", type=float, default=None,
                   help="Tversky α: penalty on rendered pixels OUTSIDE the mask "
                        "(green∖red). >1 ⇒ 'silhouette must stay inside the mask' "
                        "→ fixes handle DIRECTION. Default 1.0 (= plain IoU). "
                        "Try 2-3. TP numerator keeps scale from collapsing.")
    p.add_argument("--iou-fn-weight", type=float, default=None,
                   help="Tversky β: penalty on mask pixels NOT covered by the "
                        "render (red∖green). <1 ⇒ tolerate uncovered mask. "
                        "Default 1.0. Try 0.3-0.5 alongside --iou-fp-weight.")
    p.add_argument("--dt-w-out", type=float, default=None,
                   help="When --dt-loss: weight on green-outside-mask (precision). "
                        ">1 = handle stays inside. Pair with --no-learn-scale "
                        "(DT has no anti-collapse term).")
    p.add_argument("--dt-w-in", type=float, default=None,
                   help="When --dt-loss: weight on uncovered-mask (recall). "
                        "<1 = tolerate undercoverage.")
    p.add_argument("--refine-rounds", type=int, default=2,
                   help="after yaw sweep picks the best init, re-run "
                        "optimize_multi_view N more times using the previous "
                        "(R, t, s) as init. Each round re-fits R/t under the "
                        "latest scale, then Stage 2 re-fits scale. Default 2.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug-dir", default=None,
                   help="optional dir for per-iter render dumps")
    # ── confidence-gated, evenly-subsampled multi-cam frame selection ──
    p.add_argument("--score-thr", type=float, default=0.0,
                   help="drop a (cam,frame) view whose best matching SAM mask "
                        "scores below this — i.e. the object isn't confidently "
                        "present in that frame. Default 0 (keep all matches).")
    p.add_argument("--max-per-cam", type=int, default=None,
                   help="cap views per cam, evenly spaced among the survivors "
                        "(after --score-thr / --frame-range).")
    p.add_argument("--frame-range", default=None, metavar="LO:HI",
                   help="restrict to frame window [LO,HI] (inclusive). REQUIRED "
                        "for a MOVING object: the optimizer fits ONE pose, so pin "
                        "a static window (e.g. before the block is grasped).")
    p.add_argument("--instance", default=None,
                   choices=["left", "right", "top", "bottom", "largest", "smallest"],
                   help="when ONE label matches multiple instances per frame "
                        "(two same-colour 'gray block's = arc AND rec), pick this "
                        "instance instead of the best-scoring one. left/right/top/"
                        "bottom = IMAGE extreme (pair with --only-cams <one cam>; "
                        "position can vary per episode). largest/smallest = by mask "
                        "area — ROBUST across episodes when relative size is fixed "
                        "(arc≈4×rec → arc=largest, rec=smallest).")
    p.add_argument("--mask-dilate", type=int, default=0, metavar="PX",
                   help="grow each chosen SAM mask outward by PX pixels before the "
                        "pose fit. SAM silhouettes sit slightly inside the true edge; "
                        "1-2 px recovers the boundary so the mesh isn't shrunk to a "
                        "too-tight mask. Default 0 (raw mask).")
    p.add_argument("--cam-balance", action="store_true",
                   help="weight each view by 1/(#views from its camera) so every "
                        "CAMERA contributes equally to the summed multi-view loss. "
                        "Without it, 1 ego + N wrist frames lets wrist outvote ego "
                        "N:1 on depth/scale. Use for joint ego+wrist fits.")
    p.add_argument("--rot-cams", default=None, metavar="CAM[,CAM]",
                   help="restrict the ROTATION gradient to these cameras; other "
                        "views render with R detached (still drive t/scale). E.g. "
                        "'--only-cams ego,wrist --cam-balance --rot-cams wrist' = "
                        "t/scale from ego+wrist (better depth), yaw from wrist only "
                        "(the close wrist resolves orientation better than ego).")
    p.add_argument("--laydown", action="store_true",
                   help="pre-rotate the mesh so its THINNEST axis points up (rests on "
                        "its largest face) BEFORE the yaw-only fit. Use when the .glb's "
                        "canonical orientation stands the object up but it lies flat in "
                        "the scene (e.g. the arch block). The pre-rotation is folded "
                        "into the saved R, so export is unchanged.")
    p.add_argument("--mesh-prerot", default=None, metavar="AXIS:DEG[,AXIS:DEG]",
                   help="explicit mesh pre-rotation, applied ON TOP of --laydown, "
                        "left→right. E.g. 'y:90' or 'y:90,x:180' to flip which face is "
                        "down / which way the opening points. Folded into saved R.")
    p.add_argument("--init-pose-from", default=None, metavar="POSE.json",
                   help="load init R/t/scale from an existing pose json (single-run, no "
                        "yaw sweep). Pair with --freeze-R to KEEP that rotation and "
                        "refine only t/scale — e.g. stage-2 'wrist-yaw frozen, ego+wrist "
                        "fix depth'. (Assumes the same mesh orientation as that fit; for "
                        "--laydown objects re-pass --laydown so prerot matches.)")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    label = args.label if args.label is not None else args.object.replace("_", " ")
    only_cams = [s.strip() for s in args.only_cams.split(",")] if args.only_cams else None
    rot_cams = [s.strip() for s in args.rot_cams.split(",")] if args.rot_cams else None
    pose_from = None
    if args.init_pose_from:
        pose_from = json.loads(Path(args.init_pose_from).read_text())
        print(f"[init] --init-pose-from {args.init_pose_from}: R/t/scale as init "
              f"(single-run; add --freeze-R to keep R and only refine t/scale)")

    # ── 1. Load multi-cam bundle (for sanity prints) ─────────────────
    bundle = MultiCamBundle.load(scene_dir)
    print(f"[step4] scene = {scene_dir}")
    print(f"[step4] cams  = {[c.name for c in bundle.cameras.cameras]}")
    print(f"[step4] T     = {bundle.T} frames")
    print(f"[step4] object='{args.object}', label='{label}', mesh={args.mesh}")
    _lap("args+bundle load")

    # ── 2. Load mesh (M_LOAD → z-up canonical, simplified to 5k tris) ──
    device = torch.device(args.device)
    verts, faces, _ = load_glb_as_pytorch3d(Path(args.mesh), device=device)

    # ── mesh pre-rotation (lay flat) ─────────────────────────────────
    # The optimizer fits a pure-yaw R on `verts`; if the object lies flat in the
    # scene but the .glb stands it up, we tip it here FIRST so yaw-only+floor-lock
    # is valid, then fold R_pre into the saved R (verts·R_pre·R_yaw = verts·R_save).
    R_pre = _build_prerot(verts, args.laydown, args.mesh_prerot)
    if not np.allclose(R_pre, np.eye(3)):
        ext0 = (verts.max(0).values - verts.min(0).values).detach().cpu().numpy()
        verts = verts @ torch.from_numpy(R_pre).float().to(device)
        ext1 = (verts.max(0).values - verts.min(0).values).detach().cpu().numpy()
        print(f"[step4] mesh pre-rotation (laydown={args.laydown}, "
              f"mesh_prerot={args.mesh_prerot!r}):")
        print(f"        loaded dims X,Y,Z {ext0.round(4).tolist()} → "
              f"{ext1.round(4).tolist()}  (Z={ext1[2]*100:.1f}cm vertical)")

    z_min = float(verts[:, 2].min().item())
    z_ext = float(verts[:, 2].max().item() - z_min)
    print(f"[step4] mesh verts={tuple(verts.shape)}  z_min={z_min:.4f}  z_ext={z_ext:.4f}")
    _lap("mesh load+prerot")

    # ── 3. Build views ───────────────────────────────────────────────
    frame_min = frame_max = None
    if args.frame_range:
        lo, _, hi = args.frame_range.partition(":")
        frame_min = int(lo) if lo.strip() else None
        frame_max = int(hi) if hi.strip() else None
    views = build_views_for_object(
        scene_dir, label, only_cams=only_cams,
        score_thr=args.score_thr, max_per_cam=args.max_per_cam,
        frame_min=frame_min, frame_max=frame_max,
        instance_select=args.instance, dilate_px=args.mask_dilate,
        cam_balance=args.cam_balance,
    )
    print(f"[step4] {len(views)} views matched label='{label}' "
          f"(score_thr={args.score_thr}, max_per_cam={args.max_per_cam}, "
          f"frame_range={args.frame_range}, instance={args.instance})")
    if args.drop_frames:
        drop = set()
        for spec in args.drop_frames:
            cam, _, fr = spec.partition("/")
            drop.add((cam.strip(), int(fr)))
        kept = [v for v in views
                if (v["name"].split("/")[0], int(v["name"].split("/")[1])) not in drop]
        print(f"[step4] --drop-frames {args.drop_frames}: "
              f"{len(views)}→{len(kept)} views; kept {[v['name'] for v in kept]}")
        views = kept
        if not views:
            raise SystemExit("[step4] all views dropped — check --drop-frames")

    # ── 4. Initialize pose ──────────────────────────────────────────
    if args.init_t is not None:
        init_t = torch.tensor([float(x) for x in args.init_t.split(",")]).float().to(device)
        print(f"[init] user-set init_t={init_t.cpu().tolist()}")
    elif pose_from is not None:
        init_t = torch.tensor([float(x) for x in pose_from["t"]]).float().to(device)
        print(f"[init] init_t from pose_from = {init_t.cpu().tolist()}")
    else:
        _lap("build views")
        init_t = triangulate_init_t_from_masks(views)
        if init_t is None:
            # Single-view / degenerate triangulation: pin the footprint on the
            # support plane instead of letting the mesh land on the camera.
            init_t = floor_intersect_init_t(views, plane_z=args.plane_z)
        if init_t is None:
            init_t = init_t_from_ego_pointmap(scene_dir, views)
            print(f"[init] triangulation + floor-intersect failed; "
                  f"ego-pointmap fallback: {init_t.tolist()}")
        init_t = init_t.to(device)

    # init_R seed for scale estimation — yaw doesn't really change projected area much
    init_R_seed = torch.eye(3, device=device)
    if args.init_scale is not None:
        init_scale = float(args.init_scale)
        print(f"[init] user-set init_scale={init_scale:.4f}")
    elif pose_from is not None:
        init_scale = float(pose_from["scale"])
        print(f"[init] init_scale from pose_from = {init_scale:.4f}")
    else:
        init_scale = estimate_init_scale(verts, faces, views, init_R_seed, init_t, device)

    _lap("init t+scale")
    # Render init pose for debug BEFORE running optimization — lets us see
    # whether init was already off vs Stage 1 making it worse.
    render_debug_overlays(verts, faces, views, init_R_seed, init_t, init_scale,
                          device, scene_dir, f"{args.object}_init")
    _lap("init QA overlay")

    floor = {"y_min": z_min} if args.floor_lock else None

    # ── 5. Optimize ─────────────────────────────────────────────────
    cfg = OptimizerConfig()
    # Asymmetric silhouette (Tversky) — applies to BOTH soft-IoU and the DT blend.
    if args.iou_fp_weight is not None:
        cfg.stage1_iou_fp_weight = float(args.iou_fp_weight)
    if args.iou_fn_weight is not None:
        cfg.stage1_iou_fn_weight = float(args.iou_fn_weight)
    if cfg.stage1_iou_fp_weight != 1.0 or cfg.stage1_iou_fn_weight != 1.0:
        print(f"[step4] asymmetric silhouette (Tversky): "
              f"α(green-outside-red)={cfg.stage1_iou_fp_weight}, "
              f"β(red-uncovered)={cfg.stage1_iou_fn_weight}")
        if not args.no_learn_scale and not args.freeze_scale:
            print(f"[step4]   note: learn_scale is ON; with α>β scale tends to land "
                  f"a touch small (mug fits inside mask). Add --no-learn-scale to "
                  f"let area-ratio set scale independently of the asymmetry.")
    if args.dt_loss:
        cfg.use_dt_loss = True
        if args.dt_weight is not None:
            cfg.stage1_dt_weight = float(args.dt_weight)
        if args.dt_iou_blend is not None:
            cfg.stage1_dt_iou_blend = float(args.dt_iou_blend)
        if args.dt_w_out is not None:
            cfg.stage1_dt_w_out = float(args.dt_w_out)
        if args.dt_w_in is not None:
            cfg.stage1_dt_w_in = float(args.dt_w_in)
        print(f"[step4] DT silhouette loss enabled "
              f"(stage1_dt_weight={cfg.stage1_dt_weight}, "
              f"stage1_dt_iou_blend={cfg.stage1_dt_iou_blend}, "
              f"w_out={cfg.stage1_dt_w_out}, w_in={cfg.stage1_dt_w_in})")
    opt = PoseOptimizer(config=cfg, device=str(device))

    # Decide yaw_only: if user provides a quat that isn't pure z-rotation,
    # we MUST go full 3DOF, otherwise the yaw_only parameterization silently
    # drops the off-axis components.
    yaw_only = not args.no_yaw_only
    R_user = None
    if args.init_quat is not None:
        R_user = _quat_genesis_to_R_pt3d(args.init_quat).to(device)
    elif pose_from is not None:
        R_user = torch.tensor(pose_from["R"], dtype=torch.float32, device=device)
    if R_user is not None and yaw_only and not _is_pure_yaw(R_user):
        print(f"[step4] [warn] init R is not a pure z-rotation; forcing --no-yaw-only.")
        yaw_only = False

    best = None
    all_results = []

    if R_user is not None:
        # Single-run mode: use user's quat / pose-from R as init; optionally freeze it.
        print(f"\n[step4] single-run mode (init R from "
              f"{'pose-from' if pose_from is not None else 'init-quat'}): skipping yaw "
              f"sweep (freeze_R={args.freeze_R}, yaw_only={yaw_only})")
        print(f"        R_init =\n{R_user.cpu().numpy()}")
        r = opt.optimize_multi_view(
            mesh_verts=verts, mesh_faces=faces, views=views,
            plane_z=float(args.plane_z),
            init_R=R_user, init_t=init_t, init_scale=init_scale,
            floor_constraint=floor,
            yaw_only=yaw_only,
            freeze_scale=args.freeze_scale,
            learn_scale=(not args.no_learn_scale and not args.freeze_scale),
            freeze_R=args.freeze_R,
            rot_cams=rot_cams,
            crop_pad=args.crop_pad,
            verbose=True,
            debug_save_dir=Path(args.debug_dir) if args.debug_dir else None,
            debug_name=f"{args.object}_initquat",
        )
        iou = float(r.get("final_iou", 0.0))
        print(f"[step4] init-quat run: final_iou={iou:.4f}")
        all_results.append({"yaw_deg": None, "iou": iou,
                            "scale": float(r["scale"]),
                            "t": r["t"].detach().cpu().tolist(),
                            "source": "init-quat"})
        best = {"yaw_deg": None, **r, "final_iou": iou}
    else:
        # Multi-start yaw sweep — either uniform OR derived from mask asymmetry.
        if args.auto_yaw_init:
            from real2sim.pose.yaw_init import propose_yaw_candidates
            print(f"\n[step4] computing yaw candidates from mask asymmetry "
                  f"(level 1: mask-derived multi-start)…")
            cands = propose_yaw_candidates(views, verts, init_t, verbose=True)
            if not cands:
                print("[step4] [warn] mask-derived yaw init returned nothing; "
                      "falling back to uniform sweep.")
                yaws_deg = [round(i * 360.0 / int(args.yaw_starts), 1)
                            for i in range(max(1, int(args.yaw_starts)))]
            else:
                yaws_deg = [round(c["yaw_deg"], 1) for c in cands]
                print(f"[step4] mask-derived yaw candidates: "
                      f"{[(yd, c['source']) for yd, c in zip(yaws_deg, cands)]}")
        else:
            yaw_starts_n = max(1, int(args.yaw_starts))
            yaws_deg = [round(i * 360.0 / yaw_starts_n, 1) for i in range(yaw_starts_n)]
            print(f"\n[step4] uniform multi-start yaw sweep: {yaws_deg}")
        yaw_starts = len(yaws_deg)
        if args.batch_yaw and yaw_only:
            # Batched: fit ALL yaw candidates in ONE rasterization (fast).
            print(f"\n[step4] batched multi-start: {yaw_starts} yaws in one pass "
                  f"(yaw-keep={args.yaw_keep})")
            res = opt.optimize_multi_view_multistart(
                mesh_verts=verts, mesh_faces=faces, views=views,
                plane_z=float(args.plane_z),
                init_yaws_deg=yaws_deg, init_t=init_t, init_scale=init_scale,
                floor_constraint=floor,
                freeze_scale=args.freeze_scale,
                learn_scale=(not args.no_learn_scale and not args.freeze_scale),
                rot_cams=rot_cams,
                keep_top=int(args.yaw_keep),
                crop_pad=args.crop_pad,
                verbose=True,
                debug_save_dir=Path(args.debug_dir) if args.debug_dir else None,
                debug_name=f"{args.object}_batchyaw",
            )
            best = {"yaw_deg": res["best_yaw_deg"], **res}
            best["final_iou"] = float(res["final_iou"])
            all_results = list(res.get("all_starts", []))
            print(f"\n[step4] best yaw = {best['yaw_deg']}° "
                  f"(final_iou={best['final_iou']:.4f})")
            print(f"[step4] all yaws: " +
                  ", ".join(f"{a['yaw_deg']}°→{a['iou']:.3f}" for a in all_results))
        else:
            if args.batch_yaw and not yaw_only:
                print("[step4] [note] --batch-yaw needs yaw_only; using the "
                      "sequential loop (full-3DOF R from each yaw init).")
            for yi, yaw_deg in enumerate(yaws_deg):
                print(f"\n[step4] === yaw {yaw_deg}° ({yi+1}/{yaw_starts}) ===")
                init_R = _yaw_R(yaw_deg).to(device)
                r = opt.optimize_multi_view(
                    mesh_verts=verts, mesh_faces=faces, views=views,
                    plane_z=float(args.plane_z),
                    init_R=init_R, init_t=init_t, init_scale=init_scale,
                    floor_constraint=floor,
                    yaw_only=yaw_only,
                    freeze_scale=args.freeze_scale,
                    learn_scale=(not args.no_learn_scale and not args.freeze_scale),
                    freeze_R=args.freeze_R,
                    rot_cams=rot_cams,
                    crop_pad=args.crop_pad,
                    verbose=True,
                    debug_save_dir=Path(args.debug_dir) if args.debug_dir else None,
                    debug_name=f"{args.object}_yaw{int(yaw_deg)}",
                )
                iou = float(r.get("final_iou", 0.0))
                print(f"[step4] yaw {yaw_deg}°: final_iou={iou:.4f}")
                all_results.append({"yaw_deg": yaw_deg, "iou": iou,
                                    "scale": float(r["scale"]),
                                    "t": r["t"].detach().cpu().tolist()})
                if best is None or iou > best["final_iou"]:
                    best = {"yaw_deg": yaw_deg, **r}
                    best["final_iou"] = iou
            print(f"\n[step4] best yaw = {best['yaw_deg']}° (final_iou={best['final_iou']:.4f})")
            print(f"[step4] all yaws: " +
                  ", ".join(f"{a['yaw_deg']}°→{a['iou']:.3f}" for a in all_results))
    result = best

    # ── 5b. Refine rounds: re-warm-start to coupled-refine R/t & scale ─
    for round_i in range(int(args.refine_rounds)):
        print(f"\n[step4] === refine round {round_i+1}/{args.refine_rounds} "
              f"(init: R/t/s from best so far, iou={result['final_iou']:.4f}) ===")
        r = opt.optimize_multi_view(
            mesh_verts=verts, mesh_faces=faces, views=views,
            plane_z=float(args.plane_z),
            init_R=result["R"].to(device),
            init_t=result["t"].to(device),
            init_scale=float(result["scale"]),
            floor_constraint=floor,
            yaw_only=yaw_only,
            freeze_scale=args.freeze_scale,
            learn_scale=(not args.no_learn_scale and not args.freeze_scale),
            freeze_R=args.freeze_R,
            rot_cams=rot_cams,
            crop_pad=args.crop_pad,
            verbose=True,
            debug_save_dir=Path(args.debug_dir) if args.debug_dir else None,
            debug_name=f"{args.object}_refine{round_i+1}",
        )
        iou = float(r.get("final_iou", 0.0))
        if iou > result["final_iou"]:
            print(f"[refine] round {round_i+1}: {result['final_iou']:.4f} → "
                  f"{iou:.4f}  ✓ improved")
            r["yaw_deg"] = best["yaw_deg"]   # preserve provenance for payload
            r["final_iou"] = iou
            result = r
        else:
            print(f"[refine] round {round_i+1}: {iou:.4f} (no improvement over "
                  f"{result['final_iou']:.4f}); keeping previous result.")

    # ── 6. Save ─────────────────────────────────────────────────────
    _lap("optimize")
    out_dir = scene_dir / "poses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"{args.object}.json"

    def _tolist(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    payload = {
        "object": args.object,
        "label": label,
        "instance": args.instance,
        "mesh": str(Path(args.mesh).resolve()),
        "plane_z": float(args.plane_z),
        "floor_lock": bool(args.floor_lock),
        "yaw_only": yaw_only,
        "freeze_scale": bool(args.freeze_scale),
        "freeze_R": bool(args.freeze_R),
        "rot_cams": rot_cams,
        "init_quat": list(args.init_quat) if args.init_quat is not None else None,
        "loss_type": "dt" if args.dt_loss else "soft_iou",
        "auto_yaw_init": bool(args.auto_yaw_init),
        "init_scale": init_scale,
        "init_t_world": _tolist(init_t),
        "best_yaw_deg": (None if result["yaw_deg"] is None
                         else float(result["yaw_deg"])),
        "all_yaws": all_results,
        "views": [v["name"] for v in views],
        # R_pre (mesh pre-rotation, e.g. --laydown) is folded into the saved R so
        # downstream export (verts·R) reproduces the laid-down + yawed pose with no
        # changes. R_fit_yaw / mesh_prerot kept for provenance.
        "R": _tolist(np.asarray(R_pre) @ np.asarray(_tolist(result["R"]))),
        "R_fit_yaw": _tolist(result["R"]),
        "mesh_prerot": _tolist(np.asarray(R_pre)),
        "laydown": bool(args.laydown),
        "t": _tolist(result["t"]),
        "scale": float(result["scale"]),
        "final_iou": float(result.get("final_iou", float("nan"))),
        "per_view_iou": {x["name"]: float(x["iou"]) for x in result.get("per_view_iou", [])},
    }
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"\n[step4] saved → {out_p}")
    print(f"        R = {payload['R']}")
    print(f"        t = {payload['t']}")
    print(f"        scale = {payload['scale']:.4f}   final_iou = {payload['final_iou']:.3f}")

    # Always emit per-view render overlays for QA — cheap and invaluable when IoU is low.
    render_debug_overlays(verts, faces, views,
                          torch.as_tensor(result["R"]), torch.as_tensor(result["t"]),
                          float(result["scale"]), device, scene_dir, args.object)
    _lap("save+final QA")


if __name__ == "__main__":
    main()
