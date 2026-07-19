"""Yaw-only pose refinement against the wrist camera(s) — a post-stage after the ego fit.

The ego fit pins (t, scale) well, but the wrist camera — close-up, high angular
resolution, and viewing the object from CHANGING angles as the arm approaches —
resolves the object's YAW far better (e.g. the mug-handle direction). This stage
FREEZES (t, scale) and optimizes ONLY yaw θ about world +Z, against a multi-wrist-
frame distance-transform (contour-chamfer) loss, with a FULL-CIRCLE multi-start so it
can escape the handle-flip / 180° local minima that single-view silhouette IoU can't.

Why not IoU here: area-IoU is silhouette-degenerate and shape-blind near the optimum
(flat gradient); the contour DT is sensitive to where the boundary is, which is exactly
the rotation signal a protrusion (mug handle) or concavity (arch opening) carries.

Reuses `_yaw_matrix_world` and `_make_renderers` and the SAME DT-loss form as
`PoseOptimizer.optimize_multi_view()`, so behaviour matches step4 `--dt-loss`.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras

from .optimizer import _yaw_matrix_world, _make_renderers


def _prep_view(v: dict, size: int, sigma: float, device: torch.device) -> dict:
    """Precompute renderer + camera + target mask + DT maps for one view at a
    downscaled resolution (long side = `size`), matching optimize_multi_view()."""
    H0, W0 = int(v["H"]), int(v["W"])
    f = size / max(H0, W0)
    h, w = max(1, int(round(H0 * f))), max(1, int(round(W0 * f)))
    sh, sw = h / H0, w / W0

    tgt = F.interpolate(v["mask"][None, None].float(), size=(h, w),
                        mode="bilinear", align_corners=False)[0, 0]
    tgt_b = (tgt > 0.5).float().to(device)

    sil_r, _, _ = _make_renderers((h, w), sigma, device, faces_per_pixel=50)
    cams = PerspectiveCameras(
        focal_length=((float(v["fx"]) * sw, float(v["fy"]) * sh),),
        principal_point=((float(v["cx"]) * sw, float(v["cy"]) * sh),),
        image_size=((h, w),),
        R=v["R_view"][None].to(device).float(),
        T=v["T_view"][None].to(device).float(),
        device=device, in_ndc=False,
    )

    import scipy.ndimage as ndi
    tn = (tgt_b > 0.5).cpu().numpy()
    if tn.sum() == 0 or tn.sum() == tn.size:
        dt_out = dt_in = np.zeros_like(tn, dtype=np.float32)
    else:
        dt_out = ndi.distance_transform_edt(~tn).astype(np.float32)
        dt_in = ndi.distance_transform_edt(tn).astype(np.float32)
    norm = float(max(h, w))
    return {
        "name": v.get("name"), "weight": float(v.get("weight", 1.0)),
        "tgt": tgt_b, "sil_r": sil_r, "cams": cams,
        "dt_out": torch.from_numpy(dt_out / norm).to(device),
        "dt_in": torch.from_numpy(dt_in / norm).to(device),
    }


def _dt_loss(alpha: torch.Tensor, ld: dict, w_out: float, w_in: float) -> torch.Tensor:
    """Same balanced DT loss as optimize_multi_view(): precision (green outside mask,
    weighted by distance-to-mask) + recall (mask not covered, weighted by depth-in)."""
    tb = ld["tgt"] > 0.5
    n_in = tb.float().sum().clamp(min=1.0)
    n_out = (~tb).float().sum().clamp(min=1.0)
    loss_in = ((1.0 - alpha) * ld["dt_in"] * tb.float()).sum() / n_in
    loss_out = (alpha * ld["dt_out"] * (~tb).float()).sum() / n_out
    return w_out * loss_out + w_in * loss_in


def _tversky_loss(alpha: torch.Tensor, tgt: torch.Tensor,
                  a_fp: float, b_fn: float) -> torch.Tensor:
    """1 - Tversky. a_fp>b_fn penalizes rendered pixels OUTSIDE the mask harder →
    "the silhouette must stay INSIDE the observed mask", which pins an asymmetric
    feature's DIRECTION (mug handle / arch opening). For a near-symmetric body the
    contour-DT has almost no yaw signal AND points the wrong way (verified on the
    mug); this Tversky has a clean peak at the correct handle yaw."""
    tp = (alpha * tgt).sum()
    fp = (alpha * (1 - tgt)).sum()
    fn = ((1 - alpha) * tgt).sum()
    return 1.0 - tp / (tp + a_fp * fp + b_fn * fn + 1e-6)


def _soft_iou(alpha: torch.Tensor, tgt: torch.Tensor) -> float:
    tp = (alpha * tgt).sum(); fp = (alpha * (1 - tgt)).sum(); fn = ((1 - alpha) * tgt).sum()
    return float(tp / (tp + fp + fn + 1e-6))


def refine_yaw_wrist(
    mesh_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    views: List[dict],
    *,
    init_yaw: float,
    t,
    scale: float,
    device,
    n_starts: int = 12,
    coarse_res: int = 192,
    fine_res: int = 384,
    coarse_iters: int = 40,
    fine_iters: int = 120,
    lr: float = 0.02,
    loss_kind: str = "tversky",     # "tversky" (default, pins handle) or "dt"
    a_fp: float = 3.0,              # Tversky α (penalty on green-outside-mask)
    b_fn: float = 0.5,             # Tversky β (penalty on red-uncovered)
    dt_w_out: float = 1.0,
    dt_w_in: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Refine yaw-only (θ about world +Z) against `views` with (t, scale) FROZEN.

    Full-circle multi-start (n_starts evenly spaced from init_yaw) at coarse res to
    pick the basin, then a fine-res GD refine of the winner. `mesh_verts` must already
    be in the frame the saved yaw applies to (i.e. mesh_prerot already applied by the
    caller); the returned yaw is a PURE yaw — re-fold mesh_prerot @ R_yaw(θ) to save.
    """
    device = torch.device(device)
    mv = mesh_verts.to(device)
    mf = mesh_faces.to(device)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device)

    def _view_loss(alpha, ld):
        if loss_kind == "dt":
            return _dt_loss(alpha, ld, dt_w_out, dt_w_in)
        return _tversky_loss(alpha, ld["tgt"], a_fp, b_fn)

    def run(theta0: float, lds: List[dict], iters: int):
        # On a near-symmetric object the yaw landscape is shallow and GD wanders;
        # track the BEST-loss θ seen, not the last step (that's what flipped 160°→243°).
        theta = torch.tensor(float(theta0), device=device,
                             dtype=torch.float32, requires_grad=True)
        opt = Adam([theta], lr=lr)
        best_l, best_th = float("inf"), float(theta0)
        for _ in range(iters):
            opt.zero_grad()
            cur = float(theta.detach())
            R = _yaw_matrix_world(theta)
            meshes = Meshes(verts=[mv * scale @ R + t_t], faces=[mf])
            loss = torch.zeros((), device=device)
            for ld in lds:
                alpha = ld["sil_r"](meshes, cameras=ld["cams"])[..., 3].squeeze()
                loss = loss + ld["weight"] * _view_loss(alpha, ld)
            l = float(loss.detach())
            if l < best_l:
                best_l, best_th = l, cur
            loss.backward(); opt.step()
        return best_th, best_l

    # ── coarse multi-start (escape handle-flip local minima) ──
    if verbose:
        print(f"[wrist] {len(views)} view(s); loss={loss_kind}"
              f"{f' (α={a_fp},β={b_fn})' if loss_kind=='tversky' else ''}; "
              f"{n_starts}-way full-circle yaw multi-start (coarse {coarse_res}px × {coarse_iters} it)")
    coarse_lds = [_prep_view(v, coarse_res, 1e-2, device) for v in views]
    starts = [init_yaw + 2 * math.pi * i / max(1, n_starts) for i in range(max(1, n_starts))]
    results = []
    for i, th0 in enumerate(starts):
        th, l = run(th0, coarse_lds, coarse_iters)
        results.append((l, th, th0))
        if verbose:
            print(f"  start {i+1:2d}/{n_starts}  yaw0={math.degrees(th0)%360:5.0f}° "
                  f"→ {math.degrees(th)%360:5.0f}°  loss={l:.4f}")
    results.sort(key=lambda r: r[0])
    best_l, best_th, _ = results[0]

    # ── fine refine the winner ──
    fine_lds = [_prep_view(v, fine_res, 1e-4, device) for v in views]
    fine_th, fine_l = run(best_th, fine_lds, fine_iters)
    if verbose:
        print(f"[wrist] best coarse yaw {math.degrees(best_th)%360:.0f}° "
              f"→ fine {math.degrees(fine_th)%360:.0f}°  loss {best_l:.4f}→{fine_l:.4f}")

    # per-view IoU at the final yaw (QA only)
    R = _yaw_matrix_world(torch.tensor(fine_th, device=device))
    meshes = Meshes(verts=[mv * scale @ R + t_t], faces=[mf])
    per_view = []
    for ld in fine_lds:
        with torch.no_grad():
            alpha = ld["sil_r"](meshes, cameras=ld["cams"])[..., 3].squeeze()
        per_view.append({"name": ld["name"], "iou": _soft_iou(alpha, ld["tgt"])})
    if verbose:
        print("[wrist] per-view IoU: " +
              ", ".join(f"{p['name']}={p['iou']:.3f}" for p in per_view))

    return {
        "yaw": float(fine_th),
        "yaw_deg": math.degrees(fine_th) % 360,
        "init_yaw_deg": math.degrees(init_yaw) % 360,
        "delta_deg": ((math.degrees(fine_th - init_yaw) + 180) % 360) - 180,
        "loss": float(fine_l),
        "loss_kind": loss_kind,
        "R_yaw": _yaw_matrix_world(torch.tensor(fine_th)).cpu().numpy(),
        "per_view_iou": per_view,
        "n_starts": n_starts,
        "all_starts": [
            {"yaw0_deg": math.degrees(t0) % 360,
             "yaw_deg": math.degrees(th) % 360, "loss": l}
            for l, th, t0 in results
        ],
    }
