"""config_merge.py — base + per-episode patch -> runnable config.

⚠️ VENDORED MIRROR of GSRL-exp `experiments/real2sim/config_merge.py`. The merge rules are
FROZEN by the cross-repo contract (docs/superpowers/specs/2026-07-10-base-patch-config-merge
in GSRL-exp). Keep this byte-for-byte behaviorally identical — do NOT diverge. R2SConfig uses
it only to emit the "merged" preview; GSRL-exp runs the same logic on its side.

Rules (see the spec):
  * dict + dict          -> deep-merge (patch overrides, base-only kept, patch-only appended)
  * scalar / scalar list -> patch replaces wholesale (never element-wise)
  * tagged dict-list     -> merge BY `tag` (same tag deep-merges, new tag appended, base order)
  * other list / type change -> patch replaces wholesale
  * `_`-prefixed keys    -> merged like any key; env ignores unknown keys via .get()
Inputs are never mutated; base key order preserved, patch-only keys appended.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

TAG_KEY = "tag"


def _is_tagged_dict_list(value: Any) -> bool:
    """True iff `value` is a non-empty list whose every element is a dict with a `tag`."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(e, dict) and TAG_KEY in e for e in value)
    )


def _merge_tagged_lists(base_list: list, patch_list: list) -> list:
    """Merge two tag-keyed spec lists (base order kept; same tag deep-merges; new tag appended)."""
    merged = [copy.deepcopy(e) for e in base_list]
    index = {e[TAG_KEY]: i for i, e in enumerate(merged)}
    for patch_entry in patch_list:
        tag = patch_entry[TAG_KEY]
        if tag in index:
            merged[index[tag]] = deep_merge(merged[index[tag]], patch_entry)
        else:
            index[tag] = len(merged)
            merged.append(copy.deepcopy(patch_entry))
    return merged


def _merge_value(base_value: Any, patch_value: Any) -> Any:
    """Combine one base value with the corresponding patch value (see module docstring)."""
    if isinstance(base_value, dict) and isinstance(patch_value, dict):
        return deep_merge(base_value, patch_value)
    if _is_tagged_dict_list(base_value) and _is_tagged_dict_list(patch_value):
        return _merge_tagged_lists(base_value, patch_value)
    return copy.deepcopy(patch_value)


def deep_merge(base: dict, patch: dict) -> dict:
    """Return a new dict = `base` with `patch` deep-merged onto it (inputs untouched)."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        raise TypeError(
            f"deep_merge expects two dicts, got {type(base).__name__} and {type(patch).__name__}"
        )
    merged: dict = {}
    for key, base_value in base.items():
        if key in patch:
            merged[key] = _merge_value(base_value, patch[key])
        else:
            merged[key] = copy.deepcopy(base_value)
    for key, patch_value in patch.items():
        if key not in base:
            merged[key] = copy.deepcopy(patch_value)
    return merged


def merge_config(base: dict, patch: dict) -> dict:
    """Apply an episode `patch` onto a `base` config dict; return the runnable config."""
    return deep_merge(base, patch)


def merge_config_files(base_path: str | Path, patch_path: str | Path) -> dict:
    """Load a base config + an episode patch from disk and return the merged config dict."""
    base = json.loads(Path(base_path).read_text())
    patch = json.loads(Path(patch_path).read_text())
    return merge_config(base, patch)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge a per-episode patch onto a base config.")
    ap.add_argument("--base", required=True, type=Path, help="base config JSON (full template)")
    ap.add_argument("--patch", required=True, type=Path, help="per-episode patch JSON (override)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write merged config here (default: print to stdout)")
    args = ap.parse_args(argv)
    merged = merge_config_files(args.base, args.patch)
    text = json.dumps(merged, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
        print(f"[config_merge] {args.base} + {args.patch} -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
