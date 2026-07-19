#!/usr/bin/env python3
"""extract_lerobot.py — extract one episode of a LeRobot v2/v3 dataset to the
                       multi-cam pipeline layout.

Camera rig parameters (ego pose, fovs, wrist offset) come from a Genesis
template (e.g. example/1.json). Per-frame TCP pose + joints come from the
dataset's `observation.state`.

Output layout (consumed by real2sim/io/{cameras,robot,dataset}.py):

    <output>/
      cameras.json
      robot_traj.json
      cam_ego/   frames/000000.png ... 0000NN.png
      cam_wrist/ frames/000000.png ...
      scene.jpg                    # symlink to cam_ego/frames/000000.png

Handles both:
  - v2 layout (meta/episodes.jsonl + single mp4 per cam)
  - v3 layout (meta/episodes/chunk-XXX/file-NNN.parquet + multi-mp4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────

def quat_wxyz_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n2 = w * w + x * x + y * y + z * z
    if n2 < 1e-12:
        return np.eye(3, dtype=np.float64)
    n = n2 ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_se3(R, t):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = t
    return M


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────

def discover_snapshot(snap: Path) -> dict:
    """Print + return a structured map of meta/, data/, videos/ contents."""
    summary = {"meta_files": [], "data_files": [], "video_files_by_cam": {}}

    meta = snap / "meta"
    if meta.exists():
        for p in sorted(meta.rglob("*")):
            if p.is_file():
                summary["meta_files"].append(str(p.relative_to(snap)))

    data = snap / "data"
    if data.exists():
        for p in sorted(data.rglob("*.parquet")):
            summary["data_files"].append(str(p.relative_to(snap)))

    videos = snap / "videos"
    if videos.exists():
        cam_dirs = [d for d in videos.rglob("observation.images.*") if d.is_dir()]
        for d in cam_dirs:
            cam_name = d.name.split("observation.images.", 1)[1]
            mp4s = sorted(d.rglob("*.mp4"))
            summary["video_files_by_cam"].setdefault(cam_name, []).extend(
                str(p.relative_to(snap)) for p in mp4s
            )

    print("[discover] meta files:")
    for p in summary["meta_files"]:
        print(f"    {p}")
    print(f"[discover] data parquet files ({len(summary['data_files'])}):")
    for p in summary["data_files"]:
        print(f"    {p}")
    print("[discover] video files per cam:")
    for cam, mp4s in summary["video_files_by_cam"].items():
        print(f"    {cam}: {len(mp4s)} mp4")
        for p in mp4s:
            print(f"        {p}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Episode boundary resolution
# ─────────────────────────────────────────────────────────────────────

def _ep_bounds_from_jsonl(eps_p: Path, episode: int):
    """v2 layout: meta/episodes.jsonl, one episode per line."""
    cursor = 0
    last_ep = None
    with eps_p.open() as f:
        for line in f:
            ep = json.loads(line)
            last_ep = ep
            idx = int(ep.get("episode_index", -1))
            if "dataset_from_index" in ep and "dataset_to_index" in ep:
                if idx == episode:
                    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"]), ep
            length = int(ep.get("length", ep.get("num_frames", 0)))
            if idx == episode:
                return cursor, cursor + length, ep
            cursor += length
    return None


def _ep_bounds_from_meta_parquet(snap: Path, episode: int):
    """v3 layout: meta/episodes/{chunk-X/file-N,}.parquet."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    candidates = []
    for pat in ["meta/episodes/**/*.parquet", "meta/episodes.parquet"]:
        candidates += list(snap.glob(pat))
    if not candidates:
        return None

    tbls = [pq.read_table(p) for p in sorted(set(candidates))]
    tbl = pa.concat_tables(tbls) if len(tbls) > 1 else tbls[0]
    df = tbl.to_pandas()
    print(f"[meta] episodes parquet: {len(df)} rows, columns={list(df.columns)}")

    if "episode_index" not in df.columns:
        return None
    mask = df["episode_index"] == episode
    if not mask.any():
        return None
    row = df[mask].iloc[0].to_dict()

    for k_from, k_to in [
        ("dataset_from_index", "dataset_to_index"),
        ("from_index", "to_index"),
    ]:
        if k_from in row and k_to in row:
            return int(row[k_from]), int(row[k_to]), row

    # accumulate `length` to derive bounds
    if "length" in df.columns:
        df_sorted = df.sort_values("episode_index").reset_index(drop=True)
        total = 0
        for _, r in df_sorted.iterrows():
            length = int(r["length"])
            if int(r["episode_index"]) == episode:
                return total, total + length, r.to_dict()
            total += length
    return None


def _ep_bounds_from_data_scan(snap: Path, episode: int):
    """Last resort: scan data/*.parquet's episode_index column for the boundary."""
    import pyarrow.parquet as pq

    pqs = sorted((snap / "data").rglob("*.parquet"))
    if not pqs:
        return None
    cursor = 0
    for p in pqs:
        try:
            tbl = pq.read_table(p, columns=["episode_index"])
        except Exception:
            tbl = pq.read_table(p)
            if "episode_index" not in tbl.column_names:
                return None
        ep_arr = np.asarray(tbl.column("episode_index").to_pylist())
        idxs = np.where(ep_arr == episode)[0]
        if len(idxs) > 0:
            return cursor + int(idxs[0]), cursor + int(idxs[-1] + 1), {"_from_data_scan": True}
        cursor += len(ep_arr)
    return None


def find_episode_range(snap: Path, episode: int):
    """Return (global_from, global_to, episode_metadata_dict)."""
    jsonl = snap / "meta" / "episodes.jsonl"
    if jsonl.exists():
        result = _ep_bounds_from_jsonl(jsonl, episode)
        if result is not None:
            return result
        print(f"[warn] {jsonl} present but episode {episode} not found; trying parquet fallbacks", file=sys.stderr)

    result = _ep_bounds_from_meta_parquet(snap, episode)
    if result is not None:
        return result

    print(f"[warn] no usable meta/episodes.*; scanning data parquet", file=sys.stderr)
    result = _ep_bounds_from_data_scan(snap, episode)
    if result is not None:
        return result

    raise ValueError(f"could not resolve episode {episode} bounds via any strategy")


# ─────────────────────────────────────────────────────────────────────
# Video read plan (multi-mp4 aware)
# ─────────────────────────────────────────────────────────────────────

def _list_mp4s(snap: Path, cam_name: str):
    cam_dir = next(
        (d for d in (snap / "videos").rglob(f"observation.images.{cam_name}") if d.is_dir()),
        None,
    )
    if cam_dir is None:
        raise FileNotFoundError(f"no observation.images.{cam_name}/ under videos/")
    mp4s = sorted(cam_dir.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"no mp4 in {cam_dir}")
    return mp4s


def plan_video_reads(snap: Path, cam_name: str, g_from: int, g_to: int):
    """Figure out which mp4(s) cover [g_from, g_to), and the local offsets.

    Strategy: assume mp4s are sorted-order concatenated and together cover the
    full global frame range. We probe each mp4 for its frame count and slice
    the requested range against the cumulative cursor.
    """
    import cv2

    plan = []
    cursor = 0
    for p in _list_mp4s(snap, cam_name):
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        local_lo = max(g_from - cursor, 0)
        local_hi = min(g_to - cursor, n)
        if local_hi > local_lo:
            plan.append({
                "video": p,
                "local_start": int(local_lo),
                "local_end":   int(local_hi),
                "W": W, "H": H, "fps": fps, "video_frames": n,
            })
        cursor += n
        if cursor >= g_to:
            break
    if not plan:
        raise RuntimeError(f"could not cover [{g_from}, {g_to}) with any mp4 of cam '{cam_name}' "
                           f"(total mp4 frames seen = {cursor})")
    return plan


def extract_frames(plan, out_dir: Path):
    """Run a video-read plan, save consecutive PNGs starting at 000000."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for step in plan:
        cap = cv2.VideoCapture(str(step["video"]))
        try:
            # POS_FRAMES seek may snap to keyframe; for first step we accept that
            # (LeRobot encodes with all-I or low GOP typically). For sequential
            # steps we naturally pick up at frame 0.
            cap.set(cv2.CAP_PROP_POS_FRAMES, step["local_start"])
            for _ in range(step["local_end"] - step["local_start"]):
                ok, frame = cap.read()
                if not ok:
                    print(f"[warn] {step['video'].name}: capture ended early at saved={saved}",
                          file=sys.stderr)
                    return saved
                cv2.imwrite(str(out_dir / f"{saved:06d}.png"), frame)
                saved += 1
        finally:
            cap.release()
    return saved


# ─────────────────────────────────────────────────────────────────────
# Data parquet slice for the chosen range
# ─────────────────────────────────────────────────────────────────────

def load_parquet_slice(snap: Path, g_from: int, g_to: int):
    import pyarrow.parquet as pq
    import pyarrow as pa

    pqs = sorted((snap / "data").rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError("no parquet under data/")

    cursor = 0
    slices = []
    for p in pqs:
        t = pq.read_table(p)
        n = t.num_rows
        lo = max(g_from - cursor, 0)
        hi = min(g_to - cursor, n)
        if hi > lo:
            slices.append(t.slice(lo, hi - lo))
        cursor += n
        if cursor >= g_to:
            break
    if not slices:
        raise RuntimeError(f"empty parquet slice for [{g_from}, {g_to})")
    return pa.concat_tables(slices) if len(slices) > 1 else slices[0]


# ─────────────────────────────────────────────────────────────────────
# Camera config + robot traj
# ─────────────────────────────────────────────────────────────────────

# Which template key holds the mount-link name for each wrist-cam stream.
_WRIST_LINK_KEY = {
    "wrist":       "wrist_link_name",          # single arm (xArm)
    "wrist_left":  "left_wrist_link_name",     # bimanual (rm65b)
    "wrist_right": "right_wrist_link_name",
}


def build_camera_set(template_path: Path, W: int, H: int, cam_names) -> dict:
    cfg = json.loads(template_path.read_text())["env"]["camera"]
    out = []
    for cam_name in cam_names:
        if cam_name == "ego":
            out.append({
                "name": "ego", "kind": "fixed",
                "fov_deg": float(cfg["ego_fov"]),
                "W": int(W), "H": int(H),
                "world_pos":    list(cfg["ego_pos"]),
                "world_lookat": list(cfg["ego_lookat"]),
                "world_up":     list(cfg["ego_up"]),
            })
        elif cam_name in _WRIST_LINK_KEY:
            # Wrist cam attaches directly to its arm's mount link (link7 for xArm,
            # l_link6/r_link6 for rm65b). The per-frame world pose of that link is
            # FK'd from joints into robot_traj.json::link_poses_world; the offset
            # `wrist_offset_T` (link→cam, as authored in the Genesis template) is
            # applied as-is by world_camera_at().
            out.append({
                "name": cam_name, "kind": "attached",
                "fov_deg": float(cfg["wrist_fov"]),
                "W": int(W), "H": int(H),
                "link_name": cfg[_WRIST_LINK_KEY[cam_name]],
                "link_offset_T": [list(row) for row in cfg["wrist_offset_T"]],
            })
        else:
            print(f"[warn] cam '{cam_name}' not in template; skipping config block "
                  f"(frames will still be extracted)", file=sys.stderr)
    return {"cameras": out}


def build_robot_traj(states: np.ndarray, fps: float) -> dict:
    n = states.shape[0]
    tcp_pos = states[:, 0:3]
    tcp_quat = states[:, 3:7]
    joints = states[:, 7:14]
    gripper = states[:, 14]

    ee_pose = [make_se3(quat_wxyz_to_R(tcp_quat[i]), tcp_pos[i]).tolist()
               for i in range(n)]

    return {
        "urdf_path": None,
        "joint_names": [f"joint.{i}" for i in range(7)],
        "ee_link": "link_tcp",
        "fps": float(fps),
        "tcp_quat_convention": "wxyz",
        "joint_angles": joints.tolist(),
        "gripper": gripper.tolist(),
        "ee_pose_world": ee_pose,
        "_note": (
            "ee_pose_world[t] is world_T_link_tcp (LeRobot 'tcp.*' columns). "
            "xArm URDF: link_tcp = link7 origin + (0, 0, 0.172) in link7 frame. "
            "wrist cam offset has already been re-expressed in link_tcp frame "
            "by extract_lerobot.py, so world_T_wristcam = ee_pose_world @ "
            "cameras.json::wrist.link_offset_T directly."
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Forward kinematics: per-frame world pose of each wrist-mount link
# ─────────────────────────────────────────────────────────────────────
#
# Unified path for BOTH xArm (single arm) and rm65b (bimanual). We never use the
# recorded `tcp` columns to place wrist cameras: `tcp` is the controller/SDK tool
# frame, which (verified empirically against FK) need not coincide with the sim
# link the camera mounts on — for rm65b it sits ~0.167 m off l_link6, for the real
# xArm it can carry the A103 controller tool offset. Instead we do exactly what the
# Genesis sim does (eval_smolvla_sim / preprocess_workspace read link poses off the
# built robot): FK the mount link from the recorded JOINT angles through the arm
# URDF, then place the arm's URDF root in the sim-world frame via its base pose.


def _resolve_genesis_asset(path) -> Path:
    """Resolve a URDF path from the template. Absolute paths are used as-is;
    relative paths resolve against the installed `genesis` package's assets dir
    (so we don't hard-bind to one checkout), with $GENESIS_ASSETS_ROOT as override."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = os.environ.get("GENESIS_ASSETS_ROOT")
    if root:
        return Path(root) / p
    import importlib.util
    spec = importlib.util.find_spec("genesis")
    if spec is not None and spec.origin:
        return Path(spec.origin).parent / "assets" / p
    raise FileNotFoundError(
        f"cannot resolve relative URDF {path!r}: `genesis` is not importable and "
        "$GENESIS_ASSETS_ROOT is unset. Set one, or use an absolute path in the template.")


def arms_from_template(template_path: Path):
    """Per-arm FK specs from the Genesis template, or None if it has no
    env.robot.urdf (→ caller falls back to the legacy tcp path).

    One spec for a single arm (xArm), two for a bimanual rig (rm65b). Each spec:
        {cam_name, urdf(Path), mount_link, n_joints, joint_cols,
         base_pos, base_quat, joint_unit}
    """
    env = json.loads(Path(template_path).read_text())["env"]
    robot, cam = env["robot"], env["camera"]
    urdf = robot.get("urdf")
    if not urdf:
        return None
    junit = robot.get("joint_unit", "rad")          # xArm: rad, rm65b: deg

    if robot.get("use_bimanual"):
        n = int(robot.get("n_arm_joints", 6))
        return [
            dict(cam_name="wrist_left", urdf=_resolve_genesis_asset(urdf["left"]),
                 mount_link=cam.get("left_wrist_link_name", "l_link6"), n_joints=n,
                 joint_cols=[f"left.joint.{i}" for i in range(n)],
                 base_pos=robot["pos_left"], base_quat=robot["quat_left"], joint_unit=junit),
            dict(cam_name="wrist_right", urdf=_resolve_genesis_asset(urdf["right"]),
                 mount_link=cam.get("right_wrist_link_name", "r_link6"), n_joints=n,
                 joint_cols=[f"right.joint.{i}" for i in range(n)],
                 base_pos=robot["pos_right"], base_quat=robot["quat_right"], joint_unit=junit),
        ]

    n = int(robot.get("n_arm_joints", 7))
    upath = urdf["path"] if isinstance(urdf, dict) else urdf
    return [dict(cam_name="wrist", urdf=_resolve_genesis_asset(upath),
                 mount_link=cam.get("wrist_link_name", "link7"), n_joints=n,
                 joint_cols=[f"joint.{i}" for i in range(n)],
                 base_pos=robot.get("pos", [0.0, 0.0, 0.0]),
                 base_quat=robot.get("quat", [1.0, 0.0, 0.0, 0.0]), joint_unit=junit)]


def _fk_mount_link_world(urdf_path, mount_link, n_joints, base_pos, base_quat_wxyz,
                         joints, joint_unit):
    """Per-frame world_T_mount_link via pinocchio FK on a single-arm URDF.

        world_T_link(t) = se3(base_quat, base_pos) @ FK(joints[t], mount_link)

    `joints` is (T, n_joints). The first `n_joints` movable URDF joints (the arm
    chain — gripper joints come after it in tree order) are set from each row,
    after converting `joint_unit` ('deg' | 'rad') to radians.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    if model.njoints - 1 < n_joints:
        raise ValueError(f"{Path(urdf_path).name}: {model.njoints - 1} movable joints "
                         f"< requested n_joints={n_joints}")
    fid = model.getFrameId(mount_link)
    if fid >= model.nframes:
        raise KeyError(f"{Path(urdf_path).name}: no frame named {mount_link!r}")
    idx_q = [model.joints[j].idx_q for j in range(1, n_joints + 1)]   # arm joints, in order
    world_T_base = make_se3(quat_wxyz_to_R(base_quat_wxyz),
                            np.asarray(base_pos, dtype=np.float64))
    J = np.asarray(joints, dtype=np.float64)
    if joint_unit == "deg":
        J = np.deg2rad(J)

    out = []
    for row in J:
        q = pin.neutral(model)
        for k, iq in enumerate(idx_q):
            q[iq] = float(row[k])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        out.append((world_T_base @ data.oMf[fid].homogeneous).tolist())
    return out


def build_robot_traj_fk(states: np.ndarray, state_names, fps: float, arms,
                        robot_type=None) -> dict:
    """Unified robot_traj: bake per-frame world_T_mount_link for every arm via FK."""
    if not state_names:
        raise ValueError("FK path needs observation.state feature names (from "
                         "meta/info.json) to map joint columns.")
    names = list(state_names)
    if len(names) != states.shape[1]:
        raise ValueError(f"observation.state has {states.shape[1]} cols but "
                         f"{len(names)} names; layout mismatch.")
    idx = {n: i for i, n in enumerate(names)}

    link_poses, all_cols, all_joints = {}, [], []
    for arm in arms:
        J = states[:, [idx[c] for c in arm["joint_cols"]]]
        print(f"[extract] FK {arm['cam_name']} → {arm['mount_link']} "
              f"({arm['n_joints']}j {arm['joint_unit']})  {Path(arm['urdf']).name}")
        link_poses[arm["mount_link"]] = _fk_mount_link_world(
            arm["urdf"], arm["mount_link"], arm["n_joints"],
            arm["base_pos"], arm["base_quat"], J, arm["joint_unit"])
        all_cols += arm["joint_cols"]
        all_joints.append(J)
    joint_angles = (np.concatenate(all_joints, axis=1).tolist()
                    if all_joints else [])
    grippers = {c: states[:, idx[c]].tolist() for c in names if c.endswith("gripper")}

    return {
        "robot_type": robot_type,
        "urdf_path": {a["cam_name"]: str(a["urdf"]) for a in arms},
        "joint_names": all_cols,
        "joint_unit": arms[0]["joint_unit"],
        "ee_link": None,
        "fps": float(fps),
        "joint_angles": joint_angles,
        "grippers": grippers,
        "link_poses_world": link_poses,
        "_note": (
            "link_poses_world[<mount_link>][t] = se3(base_quat,base_pos) @ "
            "FK(joints[t], mount_link); wrist cam world pose = world_T_mount_link @ "
            "cameras.json::<wrist>.link_offset_T. Joints come from observation.state "
            "(NOT tcp) — this matches how the Genesis sim places the wrist cameras."
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--template", required=True,
                   help="Genesis cam template JSON (e.g. example/1.json)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=100,
                   help="extract first N frames of the episode (default 100)")
    p.add_argument("--output", required=True,
                   help="output scene dir (e.g. assets/scenes/xarm_ep110)")
    p.add_argument("--cam-names", default="ego,wrist",
                   help="comma-separated camera names (default: ego,wrist)")
    p.add_argument("--discover-only", action="store_true",
                   help="just print snapshot layout and exit")
    args = p.parse_args()

    snap = Path(args.snapshot_dir).resolve()
    template = Path(args.template).resolve()
    out = Path(args.output).resolve()
    cam_names = [c.strip() for c in args.cam_names.split(",") if c.strip()]

    if not snap.exists():
        raise FileNotFoundError(snap)
    if not template.exists():
        raise FileNotFoundError(template)

    print(f"[extract] snapshot = {snap}")
    print(f"[extract] template = {template}")
    print(f"[extract] output   = {out}")
    discover_snapshot(snap)
    if args.discover_only:
        return

    # 1) episode range
    g_from, g_to, ep_meta = find_episode_range(snap, args.episode)
    n = min(args.end_frame, g_to - g_from)
    print(f"\n[extract] episode {args.episode}: global rows [{g_from}, {g_to}), using first {n}")
    print(f"[extract] ep meta: {ep_meta}")

    # 2) plan + extract video frames per cam
    out.mkdir(parents=True, exist_ok=True)
    cam_W = cam_H = cam_fps = None
    n_saved = {}
    for cam_name in cam_names:
        plan = plan_video_reads(snap, cam_name, g_from, g_from + n)
        # use first plan step's W/H/fps as canonical for this cam
        first = plan[0]
        if cam_name == "ego" or cam_W is None:
            cam_W, cam_H, cam_fps = first["W"], first["H"], first["fps"]
        out_dir = out / f"cam_{cam_name}" / "frames"
        saved = extract_frames(plan, out_dir)
        n_saved[cam_name] = saved
        spans = " + ".join(f"{p['video'].name}[{p['local_start']}:{p['local_end']})" for p in plan)
        print(f"[extract] cam_{cam_name}: saved {saved} frames from  {spans}")
        print(f"          → {out_dir.relative_to(out.parent)}")

    # 3) cameras.json
    cam_set = build_camera_set(template, cam_W, cam_H, cam_names)
    (out / "cameras.json").write_text(json.dumps(cam_set, indent=2))
    print(f"\n[extract] cameras.json (W={cam_W}, H={cam_H})")

    # 4) parquet → robot_traj.json
    tbl = load_parquet_slice(snap, g_from, g_from + n)
    df = tbl.to_pandas()
    print(f"[extract] parquet slice: {len(df)} rows, columns={list(df.columns)}")
    if "observation.state" not in df.columns:
        raise KeyError("parquet missing 'observation.state' column")
    states = np.stack([np.asarray(s, dtype=np.float64) for s in df["observation.state"]])

    # observation.state column names (for joint→column mapping) from meta/info.json
    state_names = None
    info_p = snap / "meta" / "info.json"
    if info_p.exists():
        feats = json.loads(info_p.read_text()).get("features", {})
        state_names = (feats.get("observation.state") or {}).get("names")

    # Unified FK-from-joints path (xArm + rm65b) when the template declares URDFs;
    # else fall back to the legacy single-arm tcp builder.
    arms = arms_from_template(template)
    if arms is not None:
        robot_type = json.loads(template.read_text())["env"]["robot"].get("type")
        print(f"[extract] FK-from-joints: robot '{robot_type}', {len(arms)} arm(s), "
              f"mount links {[a['mount_link'] for a in arms]}")
        traj = build_robot_traj_fk(states, state_names, cam_fps, arms, robot_type=robot_type)
    else:
        if states.shape[1] != 15:
            print(f"[warn] observation.state is {states.shape[1]}-d, not 15, and template "
                  "has no env.robot.urdf for FK. Legacy tcp layout may be off.",
                  file=sys.stderr)
        traj = build_robot_traj(states, cam_fps)

    if "task_index" in df.columns:
        traj["task_indices"] = [int(x) for x in df["task_index"].tolist()]
    (out / "robot_traj.json").write_text(json.dumps(traj, indent=2))
    if traj.get("ee_pose_world"):
        p0 = traj["ee_pose_world"][0]
        print(f"[extract] robot_traj.json: {len(traj['ee_pose_world'])} frames, "
              f"ee_pose[0] pos=({p0[0][3]:.3f},{p0[1][3]:.3f},{p0[2][3]:.3f})")
    elif traj.get("link_poses_world"):
        lp = traj["link_poses_world"]
        link0 = next(iter(lp))
        p0 = lp[link0][0]
        print(f"[extract] robot_traj.json: {len(lp[link0])} frames, "
              f"links={list(lp)}, {link0}[0] pos="
              f"({p0[0][3]:.3f},{p0[1][3]:.3f},{p0[2][3]:.3f})")

    # 5) scene.jpg symlink
    first_ego = out / "cam_ego" / "frames" / "000000.png"
    if first_ego.exists():
        scene_jpg = out / "scene.jpg"
        if scene_jpg.is_symlink() or scene_jpg.exists():
            scene_jpg.unlink()
        scene_jpg.symlink_to(first_ego.resolve())
        print(f"[extract] scene.jpg → cam_ego/frames/000000.png")

    print(f"\n[extract] DONE  →  {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
