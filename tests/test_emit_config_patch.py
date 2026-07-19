"""CPU unit tests for the pose->patch emitter + vendored merge (pure dict/json, no torch)."""
import copy
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from real2sim.export import emit_config_patch as e   # lazy real2sim __init__, CPU-safe
from real2sim.export.config_merge import merge_config


def approx(a, b, tol=1e-6):
    if isinstance(a, list):
        return len(a) == len(b) and all(approx(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) <= tol


# ── synthetic bases mirroring the two real shapes ──────────────────────────────
def xarm_base():
    return {
        "env": {
            "robot": {"pos": [0.0, 0.0, 0.0], "type": "xarm"},
            "mug": {
                "asset_path": "assets/mug.glb",
                "spawn_pos": [0.1, 0.2, 0.3],
                "spawn_quat": [1.0, 0.0, 0.0, 0.0],
                "handle_offset": [0.0, -0.066, 0.03],
                "entity_kwargs": {"material_kwargs": {"friction": 1.0},
                                  "morph_kwargs": {"file": "m.glb", "scale": 0.22}},
            },
            "tree": {
                "asset_path": "assets/tree.glb",
                "spawn_pos": [0.6, 0.0, 0.24],
                "spawn_quat": [1.0, 0.0, 0.0, 0.0],
                "entity_kwargs": {"morph_kwargs": {"file": "t.glb", "scale": 0.38}},
            },
        }
    }


def rm65b_base():
    return {
        "env": {
            "target": {"specs": [
                {"tag": "arc_target", "spawn_pos": [0.45, -0.15, -0.44],
                 "random_yaw_deg": 0.0,
                 "entity_kwargs": {"morph_kwargs": {"file": "arc.glb", "scale": 1.0}}},
                {"tag": "rec_target", "spawn_pos": [0.45, 0.15, -0.44],
                 "random_yaw_deg": 0.0,
                 "entity_kwargs": {"morph_kwargs": {"file": "rec.glb", "scale": 1.0}}},
            ]},
            "basket": {"specs": [
                {"tag": "orange_basket_left", "spawn_pos": [0.51, 0.43, -0.415],
                 "quat": [0.707, 0.0, 0.0, 0.707]},
            ]},
            "pairings": [{"arm": "right", "target_tag": "arc_target"}],
        }
    }


# ── frozen numeric core ────────────────────────────────────────────────────────
def test_quat_identity():
    assert approx(e.mat_to_quat_wxyz([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [1, 0, 0, 0])


def test_quat_yaw90_colvec():
    # column-vector R_z(90°): maps +x->+y
    m = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert approx(e.mat_to_quat_wxyz(m), [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], 1e-6)


def test_transpose():
    assert e.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]


def test_pose_to_placement_transposes_R():
    # our R is row-vec yaw-90; transpose -> col-vec R_z(90) -> quat [.707,0,0,.707]
    pose = {"R": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "t": [1.0, 2.0, 3.0], "scale": 0.5}
    pos, quat, scale = e.pose_to_placement(pose)
    assert pos == [1.0, 2.0, 3.0]
    assert approx(quat, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert scale == 0.5


# ── orientation-key mirror rule ────────────────────────────────────────────────
def test_orientation_key_precedence():
    assert e.orientation_key({"spawn_quat": [1, 0, 0, 0]}, "quat") == "spawn_quat"
    assert e.orientation_key({"quat": [1, 0, 0, 0]}, "spawn_quat") == "quat"
    assert e.orientation_key({"spawn_pos": [0, 0, 0]}, "quat") == "quat"        # no key -> default
    assert e.orientation_key({"spawn_quat": 1, "quat": 1}, "x") == "spawn_quat"  # spawn_quat wins


# ── nested helpers ─────────────────────────────────────────────────────────────
def test_set_nested_keeps_siblings():
    d = {}
    e.set_nested(d, ["env", "mug"], {"a": 1})
    e.set_nested(d, ["env", "tree"], {"b": 2})
    assert d == {"env": {"mug": {"a": 1}, "tree": {"b": 2}}}


def test_find_tagged():
    base = rm65b_base()
    assert e.find_tagged(base, ["env", "target", "specs"], "rec_target")["spawn_pos"] == [0.45, 0.15, -0.44]
    assert e.find_tagged(base, ["env", "target", "specs"], "nope") is None


# ── patch assembly: dict-slot (xarm) ───────────────────────────────────────────
def test_build_patch_dict_slot():
    base = xarm_base()
    placements = {"blue_mug": ([1, 2, 3], [0, 0, 0, 1], 0.11),
                  "mug_tree": ([4, 5, 6], [1, 0, 0, 0], 0.34)}
    patch = e.build_patch(base, e.DATASETS["xarm_hang_blue_mug_v0"]["objects"],
                          placements, "spawn_quat")
    assert patch["env"]["mug"]["spawn_pos"] == [1, 2, 3]
    assert patch["env"]["mug"]["spawn_quat"] == [0, 0, 0, 1]           # base uses spawn_quat
    assert patch["env"]["mug"]["entity_kwargs"]["morph_kwargs"]["scale"] == 0.11
    assert patch["env"]["tree"]["spawn_pos"] == [4, 5, 6]
    # patch must NOT carry base-only fields (handle_offset, asset_path)
    assert "handle_offset" not in patch["env"]["mug"]
    assert "asset_path" not in patch["env"]["mug"]


# ── patch assembly: tagged-list (rm65b) ────────────────────────────────────────
def test_build_patch_tagged_list_uses_default_key():
    base = rm65b_base()
    placements = {"arc": ([1, 1, 1], [1, 0, 0, 0], 0.9),
                  "rec": ([2, 2, 2], [0, 1, 0, 0], 0.8)}
    patch = e.build_patch(base, e.DATASETS["rm65b_sort_v0"]["objects"],
                          placements, "quat")           # target specs have no orientation key
    specs = patch["env"]["target"]["specs"]
    assert {s["tag"] for s in specs} == {"arc_target", "rec_target"}
    arc = next(s for s in specs if s["tag"] == "arc_target")
    assert arc["spawn_pos"] == [1, 1, 1]
    assert "quat" in arc and "spawn_quat" not in arc      # mirrors default (no base key)
    assert arc["entity_kwargs"]["morph_kwargs"]["scale"] == 0.9


def test_build_patch_mirrors_existing_base_key():
    # base target spec ALREADY has spawn_quat -> patch must use spawn_quat despite default=quat
    base = rm65b_base()
    base["env"]["target"]["specs"][0]["spawn_quat"] = [1, 0, 0, 0]
    patch = e.build_patch(base, {"arc": {"list": ["env", "target", "specs"], "tag": "arc_target"}},
                          {"arc": ([1, 1, 1], [0, 0, 0, 1], 0.9)}, "quat")
    arc = patch["env"]["target"]["specs"][0]
    assert "spawn_quat" in arc and "quat" not in arc


# ── merge correctness (uses the frozen vendored merge) ─────────────────────────
def test_merge_dict_slot_overrides_and_keeps():
    base = xarm_base()
    patch = e.build_patch(base, e.DATASETS["xarm_hang_blue_mug_v0"]["objects"],
                          {"blue_mug": ([9, 9, 9], [0, 0, 0, 1], 0.5),
                           "mug_tree": ([8, 8, 8], [1, 0, 0, 0], 0.3)}, "spawn_quat",
                          provenance={"episode": 5})
    merged = merge_config(base, patch)
    assert merged["env"]["mug"]["spawn_pos"] == [9, 9, 9]                  # overridden
    assert merged["env"]["mug"]["spawn_quat"] == [0, 0, 0, 1]
    assert merged["env"]["mug"]["entity_kwargs"]["morph_kwargs"]["scale"] == 0.5
    assert merged["env"]["mug"]["handle_offset"] == [0.0, -0.066, 0.03]    # base-only kept
    assert merged["env"]["mug"]["entity_kwargs"]["morph_kwargs"]["file"] == "m.glb"  # sibling kept
    assert merged["env"]["robot"] == {"pos": [0.0, 0.0, 0.0], "type": "xarm"}  # untouched
    assert merged["_r2s"] == {"episode": 5}                                # provenance added


def test_merge_tagged_list_by_tag_keeps_rest():
    base = rm65b_base()
    patch = e.build_patch(base, e.DATASETS["rm65b_sort_v0"]["objects"],
                          {"arc": ([1, 1, 1], [1, 0, 0, 0], 0.9),
                           "rec": ([2, 2, 2], [0, 1, 0, 0], 0.8)}, "quat")
    merged = merge_config(base, patch)
    specs = merged["env"]["target"]["specs"]
    assert [s["tag"] for s in specs] == ["arc_target", "rec_target"]       # base order kept
    arc = specs[0]
    assert arc["spawn_pos"] == [1, 1, 1] and arc["quat"] == [1, 0, 0, 0]
    assert arc["random_yaw_deg"] == 0.0                                    # base field kept
    assert arc["entity_kwargs"]["morph_kwargs"]["file"] == "arc.glb"       # sibling kept
    assert merged["env"]["basket"]["specs"][0]["tag"] == "orange_basket_left"  # basket untouched
    assert merged["env"]["pairings"] == [{"arm": "right", "target_tag": "arc_target"}]


def test_scalar_list_replaced_not_merged():
    base = {"env": {"mug": {"spawn_pos": [1, 2, 3, 4], "spawn_quat": [1, 0, 0, 0]}}}
    patch = {"env": {"mug": {"spawn_pos": [9, 9]}}}                        # different length
    merged = merge_config(base, patch)
    assert merged["env"]["mug"]["spawn_pos"] == [9, 9]                     # wholesale replace


def test_inputs_never_mutated():
    base = xarm_base()
    base_snapshot = copy.deepcopy(base)
    patch = e.build_patch(base, e.DATASETS["xarm_hang_blue_mug_v0"]["objects"],
                          {"blue_mug": ([9, 9, 9], [0, 0, 0, 1], 0.5)}, "spawn_quat")
    _ = merge_config(base, patch)
    assert base == base_snapshot                                          # base untouched


# ── emit_episode end-to-end from a temp poses dir ──────────────────────────────
def _write_pose(d, R, t, scale, mesh="m.glb", iou=0.9):
    d.write_text(json.dumps({"R": R, "t": t, "scale": scale, "mesh": mesh, "final_iou": iou}))


def test_emit_episode_roundtrip(tmp_path):
    ep = tmp_path / "ep_07" / "poses"
    ep.mkdir(parents=True)
    _write_pose(ep / "blue_mug.json", [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [1, 2, 3], 0.5)
    _write_pose(ep / "mug_tree.json", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [4, 5, 6], 0.3)
    base = xarm_base()
    patch, merged = e.emit_episode("xarm_hang_blue_mug_v0", tmp_path, 7, base=base)
    assert patch["env"]["mug"]["spawn_pos"] == [1, 2, 3]
    assert approx(patch["env"]["mug"]["spawn_quat"], [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert patch["_r2s"]["episode"] == 7 and patch["_r2s"]["dataset"] == "xarm_hang_blue_mug_v0"
    assert patch["_r2s"]["objects"]["blue_mug"]["iou"] == 0.9
    assert merged["env"]["mug"]["spawn_pos"] == [1, 2, 3]
    assert merged["env"]["mug"]["handle_offset"] == [0.0, -0.066, 0.03]   # base preserved


def test_emit_episode_missing_object(tmp_path):
    ep = tmp_path / "ep_00" / "poses"
    ep.mkdir(parents=True)
    _write_pose(ep / "blue_mug.json", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 1, 1], 0.4)
    # no mug_tree.json
    patch, merged = e.emit_episode("xarm_hang_blue_mug_v0", tmp_path, 0, base=xarm_base())
    assert "mug" in patch["env"] and "tree" not in patch["env"]           # only present object
    assert list(patch["_r2s"]["objects"]) == ["blue_mug"]


def test_emit_episode_none_when_no_poses(tmp_path):
    (tmp_path / "ep_03" / "poses").mkdir(parents=True)
    assert e.emit_episode("xarm_hang_blue_mug_v0", tmp_path, 3, base=xarm_base()) is None
