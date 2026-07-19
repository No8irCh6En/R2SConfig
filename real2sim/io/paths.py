#!/usr/bin/env python3
"""pipeline_paths.py — central path/ID resolver for the real2sim pipeline.

Layout
------
    assets/
      scenes/<scene_id>/scene.jpg
      objects/<obj_name>/images/*.jpg
    outputs/
      build/                              # expensive + deterministic, cross-run
        objects/<obj_name>__<obj_hash>/
          masks/                          # step1 (SAM3 per-view mask)
          mesh.glb                        # symlink → MV-SAM3D/visualization/.../result.glb
          prompt.json                     # (name, prompt) audit trail
        scenes/<scene_id>/
          moge/                           # step3b, scene-only (no prompt dep)
            scene_pointmap.npy
            scene_intrinsics.npy
          prompts/<prompt_id>/
            prompt.json                   # (OBJECTS, SCENE_PROMPT, hash inputs)
            scene_masks.pt                # step3a
            scene_masks_overlay.png
            inpainted.png
            scene_sam3d/<slug>/...        # step3d
            init_pose/init_pose_<obj>.npz # step3e
            scene.json                    # step3c  — pose optimization result
            pipeline_results.json         # step3c
            pt3d_intermediate/            # step3c debug viz
      runs/                               # per-invocation, viz/QA-only
        <ts>[__<run_tag>]/
          manifest.json                   # which scene_id/prompt_id this run uses
          per_object/<name>_eval.png      # step3c per-object eval viz
          comparison/                     # step4 (visualization)
          gsrl_config.json                # step5a (can be hand-edited per run)
          genesis_preview/                # step5b
          step5_render.log
        latest -> <ts>...                 # symlink

Why is scene.json under build/ and not run_dir?
    It's a deterministic function of (scene_id, prompt_id, mesh hashes, optimizer
    hyperparams). All those are captured by build/ paths. Putting it in run_dir
    used to require fast_start.sh to `cp -al` it from latest into new run_dirs,
    which caused a hardlink-clobber bug (Python's write_text uses O_TRUNC, which
    modifies the inode in place, so subsequent step3c writes corrupted prior
    runs' scene.json). Keeping scene.json in build avoids the seed-and-clobber
    pattern entirely.

Env vars (env > default)
------------------------
    SCENE         single subdir of assets/scenes/ if unset (error on 0 or 2+)
    PROMPT_TAG    "auto"
    RUN_TAG       ""
    RUN           absolute/relative run dir; overrides latest-resolution
    REBUILD       1 → ignore build-cache hits

CLI
---
    python -m real2sim.io.paths --field <name> [--obj <obj_name>]
    python -m real2sim.io.paths --json
    python -m real2sim.io.paths --new-run-dir       # allocate fresh ts dir + write manifest
    python -m real2sim.io.paths --update-latest <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .scenes import OBJECTS, SCENE_PROMPT, MASK_CONFIDENCE


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # real2sim/io/.. .. == R2SConfig/


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:8]


def obj_hash(name: str, prompt: str, mask_confidence: float = MASK_CONFIDENCE) -> str:
    return _sha(name, prompt, mask_confidence)


def prompt_hash() -> str:
    objs = tuple((o["name"], o["prompt"]) for o in OBJECTS)
    return _sha(SCENE_PROMPT, objs, MASK_CONFIDENCE)


@dataclass
class PipelinePaths:
    scene_id: str
    prompt_tag: str
    prompt_id: str
    run_tag: str
    run_dir: Path

    # ── assets ──
    @property
    def assets_root(self) -> Path:
        return PROJECT_ROOT / "assets"

    @property
    def scene_dir(self) -> Path:
        return self.assets_root / "scenes" / self.scene_id

    @property
    def scene_image(self) -> Path:
        return self.scene_dir / "scene.jpg"

    def object_images_dir(self, obj_name: str) -> Path:
        return self.assets_root / "objects" / obj_name / "images"

    # ── build (cross-run) ──
    @property
    def build_root(self) -> Path:
        return PROJECT_ROOT / "outputs" / "build"

    @property
    def build_scene_dir(self) -> Path:
        return self.build_root / "scenes" / self.scene_id

    @property
    def build_moge_dir(self) -> Path:
        return self.build_scene_dir / "moge"

    @property
    def build_prompt_dir(self) -> Path:
        return self.build_scene_dir / "prompts" / self.prompt_id

    def build_object_dir(self, obj_name: str, obj_prompt: str) -> Path:
        return self.build_root / "objects" / f"{obj_name}__{obj_hash(obj_name, obj_prompt)}"

    # ── derived: build_moge ──
    @property
    def scene_pointmap_path(self) -> Path:
        return self.build_moge_dir / "scene_pointmap.npy"

    @property
    def scene_intrinsics_path(self) -> Path:
        return self.build_moge_dir / "scene_intrinsics.npy"

    # ── derived: build_prompt ──
    @property
    def scene_masks_path(self) -> Path:
        return self.build_prompt_dir / "scene_masks.pt"

    @property
    def scene_masks_overlay_path(self) -> Path:
        return self.build_prompt_dir / "scene_masks_overlay.png"

    @property
    def inpainted_image_path(self) -> Path:
        return self.build_prompt_dir / "inpainted.png"

    @property
    def scene_sam3d_dir(self) -> Path:
        return self.build_prompt_dir / "scene_sam3d"

    @property
    def init_pose_dir(self) -> Path:
        return self.build_prompt_dir / "init_pose"

    def init_pose_path(self, obj_name: str) -> Path:
        return self.init_pose_dir / f"init_pose_{obj_name}.npz"

    # ── derived: step3c outputs (moved here from run_dir to avoid the
    # `cp -al` hardlink-clobber bug; see module docstring) ──
    @property
    def scene_json_path(self) -> Path:
        return self.build_prompt_dir / "scene.json"

    @property
    def pipeline_results_path(self) -> Path:
        return self.build_prompt_dir / "pipeline_results.json"

    @property
    def pt3d_intermediate_dir(self) -> Path:
        return self.build_prompt_dir / "pt3d_intermediate"

    # ── derived: build_object ──
    def object_images_link_dir(self, obj_name: str, obj_prompt: str) -> Path:
        """Mirrored (hardlinked) copy of assets/objects/<obj>/images/ under build dir,
        because SAM3D's --input_path expects sibling `images/` and `<slug>/` subdirs."""
        return self.build_object_dir(obj_name, obj_prompt) / "images"

    def object_masks_dir(self, obj_name: str, obj_prompt: str) -> Path:
        """SAM3 mask subdir; name = sanitize_filename(prompt) so SAM3D's --mask_prompt
        flag (the slug) lines up."""
        return self.build_object_dir(obj_name, obj_prompt) / sanitize_filename(obj_prompt)

    def object_mesh_link(self, obj_name: str, obj_prompt: str) -> Path:
        """Symlink → MV-SAM3D/visualization/.../result.glb (set after step2)."""
        return self.build_object_dir(obj_name, obj_prompt) / "mesh.glb"

    # ── runs ──
    @property
    def runs_root(self) -> Path:
        return PROJECT_ROOT / "outputs" / "runs"

    @property
    def runs_latest(self) -> Path:
        return self.runs_root / "latest"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def comparison_dir(self) -> Path:
        return self.run_dir / "comparison"

    @property
    def per_object_dir(self) -> Path:
        """step3c viz: per_object/<name>_eval.png. Lives in run_dir (not build_) so
        each run has a self-contained QA folder."""
        return self.run_dir / "per_object"

    @property
    def gsrl_config_path(self) -> Path:
        return self.run_dir / "gsrl_config.json"

    @property
    def genesis_preview_dir(self) -> Path:
        return self.run_dir / "genesis_preview"

    @property
    def step5_render_log(self) -> Path:
        return self.run_dir / "step5_render.log"

    # ── manifest write ──
    def write_manifest(self, extra: Optional[dict] = None) -> None:
        data = {
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "scene_id": self.scene_id,
            "prompt_tag": self.prompt_tag,
            "prompt_id": self.prompt_id,
            "run_tag": self.run_tag,
            "scene_prompt": SCENE_PROMPT,
            "mask_confidence": MASK_CONFIDENCE,
            "objects": [
                {
                    "name": o["name"],
                    "prompt": o["prompt"],
                    "obj_hash": obj_hash(o["name"], o["prompt"]),
                }
                for o in OBJECTS
            ],
            "build_paths": {
                "scene_moge": str(self.build_moge_dir.relative_to(PROJECT_ROOT)),
                "scene_prompt": str(self.build_prompt_dir.relative_to(PROJECT_ROOT)),
                "objects": {
                    o["name"]: str(
                        self.build_object_dir(o["name"], o["prompt"]).relative_to(PROJECT_ROOT)
                    )
                    for o in OBJECTS
                },
            },
        }
        if extra:
            data.update(extra)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w") as f:
            json.dump(data, f, indent=2)


def _resolve_scene_id() -> str:
    sid = os.environ.get("SCENE", "").strip()
    if sid:
        return sid
    scenes_dir = PROJECT_ROOT / "assets" / "scenes"
    if not scenes_dir.is_dir():
        raise RuntimeError(
            "No SCENE env var set and assets/scenes/ does not exist. "
            "Create assets/scenes/<scene_id>/scene.jpg."
        )
    subs = sorted(d for d in scenes_dir.iterdir() if d.is_dir())
    if len(subs) == 1:
        return subs[0].name
    if not subs:
        raise RuntimeError("assets/scenes/ is empty. Create assets/scenes/<scene_id>/scene.jpg.")
    raise RuntimeError(
        f"assets/scenes/ has multiple subdirs ({[d.name for d in subs]}); "
        "set SCENE=<scene_id> to choose."
    )


def _new_run_dir_path(run_tag: str) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}__{run_tag}" if run_tag else ts
    return PROJECT_ROOT / "outputs" / "runs" / name


def resolve(*, allocate_new_run: bool = False) -> PipelinePaths:
    """Read env, return PipelinePaths.

    allocate_new_run=True  → fresh timestamped run_dir (caller will populate).
    allocate_new_run=False → RUN env if set, else latest symlink, else fresh ts.
    """
    scene_id = _resolve_scene_id()
    prompt_tag = os.environ.get("PROMPT_TAG", "auto").strip() or "auto"
    run_tag = os.environ.get("RUN_TAG", "").strip()
    prompt_id = f"{prompt_tag}__{prompt_hash()}"

    explicit = os.environ.get("RUN", "").strip()
    if explicit:
        p = Path(explicit)
        run_dir = p if p.is_absolute() else (PROJECT_ROOT / p)
    elif allocate_new_run:
        run_dir = _new_run_dir_path(run_tag)
    else:
        latest = PROJECT_ROOT / "outputs" / "runs" / "latest"
        if latest.exists() or latest.is_symlink():
            run_dir = latest.resolve()
        else:
            run_dir = _new_run_dir_path(run_tag)

    return PipelinePaths(
        scene_id=scene_id,
        prompt_tag=prompt_tag,
        prompt_id=prompt_id,
        run_tag=run_tag,
        run_dir=run_dir,
    )


def update_latest_symlink(run_dir: Path) -> None:
    runs_root = run_dir.parent
    latest = runs_root / "latest"
    runs_root.mkdir(parents=True, exist_ok=True)
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name, target_is_directory=True)


def should_rebuild() -> bool:
    return os.environ.get("REBUILD", "0").strip().lower() in ("1", "true", "yes")


# ── object build dir bootstrap ────────────────────────────────────────

def write_object_prompt_audit(obj_name: str, obj_prompt: str, paths: PipelinePaths) -> None:
    """Drop a prompt.json into build/objects/<obj>/ so the dir is self-describing."""
    d = paths.build_object_dir(obj_name, obj_prompt)
    d.mkdir(parents=True, exist_ok=True)
    audit = {
        "name": obj_name,
        "prompt": obj_prompt,
        "obj_hash": obj_hash(obj_name, obj_prompt),
        "mask_confidence": MASK_CONFIDENCE,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (d / "prompt.json").write_text(json.dumps(audit, indent=2))


def write_prompt_audit(paths: PipelinePaths) -> None:
    """Drop a prompt.json into build/scenes/<scene>/prompts/<prompt_id>/."""
    d = paths.build_prompt_dir
    d.mkdir(parents=True, exist_ok=True)
    audit = {
        "scene_id": paths.scene_id,
        "prompt_tag": paths.prompt_tag,
        "prompt_id": paths.prompt_id,
        "scene_prompt": SCENE_PROMPT,
        "mask_confidence": MASK_CONFIDENCE,
        "objects": [
            {"name": o["name"], "prompt": o["prompt"], "obj_hash": obj_hash(o["name"], o["prompt"])}
            for o in OBJECTS
        ],
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (d / "prompt.json").write_text(json.dumps(audit, indent=2))


# ── CLI ──────────────────────────────────────────────────────────────

_SIMPLE_FIELDS = [
    "scene_id", "prompt_tag", "prompt_id", "run_tag",
    "scene_image", "scene_dir",
    "build_root", "build_scene_dir", "build_moge_dir", "build_prompt_dir",
    "scene_pointmap_path", "scene_intrinsics_path",
    "scene_masks_path", "scene_masks_overlay_path", "inpainted_image_path",
    "scene_sam3d_dir", "init_pose_dir",
    "scene_json_path", "pipeline_results_path", "pt3d_intermediate_dir",
    "runs_root", "runs_latest", "run_dir",
    "manifest_path", "comparison_dir", "per_object_dir",
    "gsrl_config_path", "genesis_preview_dir", "step5_render_log",
]


def _cli():
    ap = argparse.ArgumentParser(description="Pipeline path/id resolver.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--field", help="Print one field (see --json for list).")
    g.add_argument("--json", action="store_true", help="Print all resolved paths as JSON.")
    g.add_argument("--new-run-dir", action="store_true",
                   help="Allocate fresh ts run dir, write manifest, print path.")
    g.add_argument("--update-latest", metavar="RUN_DIR",
                   help="Update outputs/runs/latest symlink to point at RUN_DIR.")
    ap.add_argument("--obj", help="Object name (for object-keyed fields).")
    args = ap.parse_args()

    if args.update_latest:
        update_latest_symlink(Path(args.update_latest))
        print(args.update_latest)
        return

    if args.new_run_dir:
        paths = resolve(allocate_new_run=True)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        paths.write_manifest()
        print(paths.run_dir)
        return

    paths = resolve()

    if args.json:
        out = {}
        for k in _SIMPLE_FIELDS:
            v = getattr(paths, k)
            out[k] = str(v) if isinstance(v, Path) else v
        if args.obj:
            obj = next((o for o in OBJECTS if o["name"] == args.obj), None)
            if obj is not None:
                out["object_images_dir"] = str(paths.object_images_dir(obj["name"]))
                out["build_object_dir"] = str(paths.build_object_dir(obj["name"], obj["prompt"]))
                out["object_images_link_dir"] = str(paths.object_images_link_dir(obj["name"], obj["prompt"]))
                out["object_masks_dir"] = str(paths.object_masks_dir(obj["name"], obj["prompt"]))
                out["object_mesh_link"] = str(paths.object_mesh_link(obj["name"], obj["prompt"]))
                out["init_pose_path"] = str(paths.init_pose_path(obj["name"]))
                out["obj_hash"] = obj_hash(obj["name"], obj["prompt"])
                out["prompt_slug"] = sanitize_filename(obj["prompt"])
                out["obj_prompt"] = obj["prompt"]
        print(json.dumps(out, indent=2))
        return

    if args.field:
        if args.obj:
            obj = next((o for o in OBJECTS if o["name"] == args.obj), None)
            if obj is None:
                print(f"unknown obj '{args.obj}'", file=sys.stderr); sys.exit(2)
            obj_keyed = {
                "object_images_dir": paths.object_images_dir(obj["name"]),
                "build_object_dir": paths.build_object_dir(obj["name"], obj["prompt"]),
                "object_images_link_dir": paths.object_images_link_dir(obj["name"], obj["prompt"]),
                "object_masks_dir": paths.object_masks_dir(obj["name"], obj["prompt"]),
                "object_mesh_link": paths.object_mesh_link(obj["name"], obj["prompt"]),
                "init_pose_path": paths.init_pose_path(obj["name"]),
                "obj_hash": obj_hash(obj["name"], obj["prompt"]),
                "prompt_slug": sanitize_filename(obj["prompt"]),
                "obj_prompt": obj["prompt"],
            }
            if args.field in obj_keyed:
                v = obj_keyed[args.field]
                print(str(v) if isinstance(v, Path) else v)
                return
        v = getattr(paths, args.field, None)
        if v is None:
            print(f"unknown field '{args.field}'", file=sys.stderr); sys.exit(2)
        print(str(v) if isinstance(v, Path) else v)


if __name__ == "__main__":
    _cli()
