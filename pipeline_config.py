"""pipeline_config.py — declarative config shared by all pipeline steps.

Anything in this file feeds into prompt_id / obj_hash. Bumping a value here
forces the corresponding build/ directory to be rebuilt (because its hash
changes), without touching previous runs.

OBJECTS is resolved from SCENE env var at import time (looks up
_OBJECTS_BY_SCENE). All pipeline scripts read SCENE the same way (or have
fast_start.sh export it for them), so import order doesn't matter.
"""

import os
from typing import Dict, List

# Per-scene OBJECTS table. To add a new scene, drop another entry here.
# Each entry: dict with {"name", "prompt", "slot"}.
#   name   = identifier; also the asset dir name (assets/objects/<name>/images/)
#            and the build dir prefix (outputs/build/objects/<name>__<hash>/)
#   prompt = SAM3 text prompt for that object in scene+per-view masks
#   slot   = key under GSRL template env.* that this object fills.
#            example/1.json hard-codes "mug" and "tree"; step5 uses this to
#            patch env[<slot>].asset_path / spawn_pos / spawn_quat etc.
#            (hash-neutral — slot does not enter obj_hash / prompt_hash.)
_OBJECTS_BY_SCENE: Dict[str, List[Dict[str, str]]] = {
    "red_with_mug_tree": [
        {"name": "red_mug",        "prompt": "mug",                   "slot": "mug"},
        {"name": "black_mug_tree", "prompt": "black straight object", "slot": "tree"},
    ],
    "blue_with_mug_tree": [
        {"name": "blue_mug",       "prompt": "mug",                   "slot": "mug"},
        {"name": "black_mug_tree", "prompt": "black straight object", "slot": "tree"},
    ],
    "white_with_mug_tree": [
        {"name": "white_mug",      "prompt": "mug",                   "slot": "mug"},
        {"name": "black_mug_tree", "prompt": "black straight object", "slot": "tree"},
    ],
}

# Fallback if SCENE env var is empty / unknown. Picked so a casual
# `python -m pipeline_paths --json` doesn't crash.
_DEFAULT_SCENE = "red_with_mug_tree"


def _resolve_scene_for_objects() -> str:
    sid = os.environ.get("SCENE", "").strip()
    return sid if sid in _OBJECTS_BY_SCENE else _DEFAULT_SCENE


OBJECTS: List[Dict[str, str]] = _OBJECTS_BY_SCENE[_resolve_scene_for_objects()]

# SAM3 text prompt for the whole scene (step3a/3d). Keep aligned with the
# per-object prompts so the labels returned by SAM3 can be matched to meshes.
SCENE_PROMPT = "mug. black straight object."

# SAM3 mask confidence threshold (step1). Different value → different masks →
# different downstream meshes/poses → different build dir.
MASK_CONFIDENCE = 0.1
