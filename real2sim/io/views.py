"""Build per-object view lists for `PoseOptimizer.optimize_multi_view()`.

Scans `<scene_dir>/perception/cam_<name>/frame_<NNNNNN>/masks.pt`, picks the
mask whose label matches `label_pattern` for each (cam, frame), and pairs it
with the world→view (R, T) from `MultiCamBundle.world_camera_at()`.

Returns a list of view dicts exactly in the shape `optimize_multi_view()`
expects:

    {
      "name":   "ego/000000",
      "mask":   (H, W) torch float (binary 0/1),
      "R_view": (3, 3) torch float,
      "T_view": (3,)   torch float,
      "fx", "fy", "cx", "cy": float,
      "H", "W": int,
      "depth":  (H, W) torch float OR None,    # only on the pointmap ref view
    }

The single MoGe-pointmap view (typically `ego/0`) gets `depth` attached: the
PT3D-cam-frame z channel of `pointmap.npy`, masked to the object so the
optimizer's depth-loss isn't pulled by background pixels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from .dataset import MultiCamBundle


def _norm_label(s: str) -> str:
    """lowercase, strip, '_'→' ', collapse whitespace. Used for fuzzy matching."""
    return " ".join(s.lower().replace("_", " ").split())


def _label_matcher(pattern: Union[str, Callable[[str], bool]]) -> Callable[[str], bool]:
    if callable(pattern):
        return pattern
    p = _norm_label(pattern)

    def _m(lbl: str) -> bool:
        n = _norm_label(lbl)
        return p in n or n in p
    return _m


def _pick_mask(masks: torch.Tensor, scores: torch.Tensor, labels: List[str],
               matcher: Callable[[str], bool],
               instance_select: Optional[str] = None) -> Optional[int]:
    """Index of the chosen matching mask.

    `instance_select`:
        None                  → highest-scoring matching mask (default).
        'left' / 'right'      → matching mask with min / max centroid-x.
        'top'  / 'bottom'     → matching mask with min / max centroid-y.
        'largest' / 'smallest'→ matching mask with most / fewest pixels.

    The spatial modes (left/right/top/bottom) compare IMAGE position, so they're
    only meaningful within a single camera view (use with `only_cams=[<one cam>]`)
    AND only when the instances keep that arrangement. For objects whose RELATIVE
    SIZE is invariant but whose position varies per episode (arc ≈4× rec), prefer
    'largest'/'smallest' — robust across episodes.
    """
    cand = [i for i, lbl in enumerate(labels) if matcher(lbl)]
    if not cand:
        return None
    if instance_select is None:
        return max(cand, key=lambda i: float(scores[i]) if i < len(scores) else 0.0)
    if instance_select in ("largest", "smallest"):
        areas = {i: int(masks[i].bool().sum()) for i in cand}
        areas = {i: a for i, a in areas.items() if a > 0}
        if not areas:
            return None
        return (max if instance_select == "largest" else min)(areas, key=areas.get)
    cents = {}
    for i in cand:
        ys, xs = torch.where(masks[i].bool())
        if len(ys):
            cents[i] = (float(xs.float().mean()), float(ys.float().mean()))
    if not cents:
        return None
    keyfn = {
        "left":   lambda i: cents[i][0],
        "right":  lambda i: -cents[i][0],
        "top":    lambda i: cents[i][1],
        "bottom": lambda i: -cents[i][1],
    }[instance_select]
    return min(cents, key=keyfn)


def _even_subsample(items: list, k: Optional[int]) -> list:
    """Keep at most k roughly-evenly-spaced items (order preserved)."""
    if k is None or len(items) <= k:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int).tolist()))
    return [items[i] for i in idx]


def build_views_for_object(
    scene_dir: Path,
    label_pattern: Union[str, Callable[[str], bool]],
    *,
    only_cams: Optional[List[str]] = None,
    score_thr: float = 0.0,
    max_per_cam: Optional[int] = None,
    frame_min: Optional[int] = None,
    frame_max: Optional[int] = None,
    instance_select: Optional[str] = None,
    dilate_px: int = 0,
    cam_balance: bool = False,
    verbose: bool = True,
) -> List[dict]:
    """Assemble the `views` list for `optimize_multi_view()`.

    Scans `<scene_dir>/perception/cam_<name>/frame_<NNNNNN>/masks.pt` (written by
    `scripts/step3_multicam.py --mode seg`). Returns one view dict per kept
    (cam, frame) pair, in deterministic (cam-name, frame-idx) order.

    `label_pattern`:
        str: fuzzy substring match on the SAM label (case-insensitive, '_'~' ').
        callable: receives the raw label, returns bool.

    `only_cams`: restrict to these cam names (e.g. ["ego", "wrist_left"]).

    Frame selection (applied in this order, per cam):
      - `frame_min`/`frame_max`: restrict to a frame window. The optimizer solves
        ONE shared pose, so for a MOVING object you MUST pin a static window here —
        otherwise views straddle the object's motion and the fit is meaningless.
      - `score_thr`: drop a (cam, frame) whose best matching mask scores below this
        (SAM isn't confident the object is present in that frame).
      - `max_per_cam`: of the survivors, keep this many, evenly spaced.

    `instance_select` ('left'/'right'/'top'/'bottom'): when one text label
    matches MULTIPLE instances per frame (two same-colour blocks → arc AND rec),
    pick the instance at that image extreme instead of the highest-scoring one.
    Image-relative, so pin to a single cam via `only_cams`.
    """
    if (instance_select in ("left", "right", "top", "bottom")
            and (only_cams is None or len(only_cams) != 1)):
        print(f"[views] [warn] instance_select={instance_select!r} compares IMAGE "
              f"position and is camera-dependent; you passed only_cams={only_cams}. "
              f"Restrict to ONE cam (e.g. only_cams=['ego']) or the same instance "
              f"won't be picked across cams. (Size modes 'largest'/'smallest' are "
              f"more robust when position varies.)")
    scene_dir = Path(scene_dir)
    bundle = MultiCamBundle.load(scene_dir)
    matcher = _label_matcher(label_pattern)

    # Load pointmap once if present — pin depth to the (cam, frame) it was
    # computed for so the optimizer's depth-loss has correct geometry.
    pm_ref_p = scene_dir / "perception" / "pointmap" / "ref.json"
    pm_p     = scene_dir / "perception" / "pointmap" / "pointmap.npy"
    depth_cam_name, depth_frame, depth_arr = None, None, None
    if pm_ref_p.exists() and pm_p.exists():
        ref = json.loads(pm_ref_p.read_text())
        depth_cam_name, depth_frame = ref["cam_name"], int(ref["frame"])
        pm = np.load(pm_p)                                  # (H, W, 3) PT3D cam
        depth_arr = torch.from_numpy(pm[..., 2]).float()    # z channel only

    perception = scene_dir / "perception"
    if not perception.exists():
        raise FileNotFoundError(f"no perception/ under {scene_dir}; "
                                f"run scripts/step3_multicam.py --mode seg first")

    views: List[dict] = []
    for cam_dir in sorted(perception.glob("cam_*")):
        cam_name = cam_dir.name[len("cam_"):]
        if only_cams is not None and cam_name not in only_cams:
            continue
        try:
            cam = bundle.cameras.by_name(cam_name)
        except KeyError:
            if verbose:
                print(f"[views] cam {cam_name!r} not in cameras.json; skip")
            continue

        cam_views: List[dict] = []
        for frame_dir in sorted(cam_dir.glob("frame_*")):
            try:
                t = int(frame_dir.name[len("frame_"):])
            except ValueError:
                continue
            if frame_min is not None and t < frame_min:
                continue
            if frame_max is not None and t > frame_max:
                continue
            mask_p = frame_dir / "masks.pt"
            if not mask_p.exists():
                continue
            md = torch.load(mask_p, weights_only=False, map_location="cpu")
            masks  = md["masks"]                  # (N, H, W) bool
            scores = md.get("scores", torch.zeros(masks.shape[0]))
            labels = md.get("labels", [])

            i = _pick_mask(masks, scores, labels, matcher,
                           instance_select=instance_select)
            if i is None:
                if verbose:
                    print(f"[views] skip {cam_name}/{t:06d}: no label matches "
                          f"(found {labels})")
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < score_thr:
                if verbose:
                    print(f"[views] skip {cam_name}/{t:06d}: score {score:.2f} "
                          f"< thr {score_thr:.2f}")
                continue
            mb = masks[i].bool()                  # (H, W)
            if dilate_px and dilate_px > 0:
                # SAM silhouettes sit a touch INSIDE the true object edge; a small
                # outward dilation recovers the boundary pixels so the rendered mesh
                # isn't pushed smaller to fit a too-tight mask. kernel=3 ⇒ ~1px/iter.
                from ..viz.utils import dilate_mask
                mb = dilate_mask(mb, kernel_size=3, iters=int(dilate_px))
            mask = mb.float()                     # (H, W) binary

            R_np, T_np = bundle.world_camera_at(cam, t)
            R = torch.from_numpy(R_np).float()
            T = torch.from_numpy(T_np).float()
            H, W = int(mask.shape[0]), int(mask.shape[1])
            fx, fy, cx, cy = cam.intrinsics(W=W, H=H)

            depth = None
            if (depth_arr is not None
                    and cam_name == depth_cam_name
                    and t == depth_frame
                    and depth_arr.shape == mask.shape):
                # Restrict depth supervision to the object's mask so the loss
                # isn't dominated by background pixels of MoGe's pointmap.
                depth = torch.where(mask > 0.5, depth_arr, torch.zeros_like(depth_arr))

            cam_views.append({
                "name":   f"{cam_name}/{t:06d}",
                "mask":   mask,
                "R_view": R, "T_view": T,
                "fx": float(fx), "fy": float(fy),
                "cx": float(cx), "cy": float(cy),
                "H": H, "W": W,
                "depth": depth,
                "score": score,
                "label": labels[i] if i < len(labels) else None,
            })
            if verbose:
                d = "+depth" if depth is not None else "mask-only"
                print(f"[views] {cam_name}/{t:06d}  '{labels[i]}' "
                      f"(score={score:.2f})  mask_px={int(mask.sum())}  {d}")

        kept = _even_subsample(cam_views, max_per_cam)
        if verbose and len(kept) < len(cam_views):
            print(f"[views] cam {cam_name}: {len(cam_views)}→{len(kept)} views "
                  f"(even subsample to max_per_cam={max_per_cam}; "
                  f"frames {[v['name'].split('/')[1] for v in kept]})")
        views.extend(kept)

    if not views:
        raise RuntimeError(
            f"no views matched label_pattern={label_pattern!r} under {perception}. "
            f"Check the SAM labels in masks.pt — perhaps adjust the pattern.")

    # `cam_balance`: per-view weight = 1 / (#views from that camera), so every CAMERA
    # contributes equally to the SUMMED multi-view loss regardless of how many frames
    # it has (the optimizer sums w·loss over views). Without this, 1 ego + N wrist
    # frames lets the wrist outvote the ego N:1 on depth/scale; a single ego is itself
    # depth-underconstrained, so a balanced ego+wrist fit triangulates better.
    if cam_balance:
        from collections import Counter
        cam_of = lambda v: v["name"].split("/")[0]
        counts = Counter(cam_of(v) for v in views)
        for v in views:
            v["weight"] = 1.0 / counts[cam_of(v)]
        if verbose:
            print(f"[views] cam-balance ON: per-view weights "
                  f"{ {c: round(1.0 / n, 3) for c, n in counts.items()} } "
                  f"(each of {len(counts)} cameras totals weight 1.0)")
    return views
