"""Robot trajectory + FK for attached-camera resolution.

Loaded from `assets/scenes/<scene_id>/robot_traj.json`. Records per-frame joint
state so the multi-cam pipeline can compute, for each frame t and each
attached-cam c, the world-frame pose of c via FK(joints[t], link_c) @ offset_c.

Schema:
    {
      "urdf_path":   "assets/robot/xarm.urdf",
      "joint_names": ["joint1", ..., "jointN"],
      "ee_link":     "link_ee",
      "fps":         30,
      "joint_angles":  [[θ1_t0, ..., θN_t0], ...]      # (T, N), radians
      "ee_pose_world": [[[..4x4..]], ...]              # (T, 4, 4), OPTIONAL
                                                       # if present, can skip FK
    }

`ee_pose_world` is included as a convenience for datasets that already report
the EE pose directly (no URDF / pinocchio dependency at solve time). When it's
present and the only attached cam is on `ee_link`, we can drop FK entirely.

FK back-ends in priority order (configurable at solve time):
  1. pinocchio       (preferred — fast + accurate)
  2. pybullet        (fallback, also fine)
  3. yourdfpy + manual chain (last resort, no external sim dep)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RobotTrajectory:
    urdf_path: Optional[str]
    joint_names: List[str]
    ee_link: Optional[str]
    fps: float
    joint_angles: List[List[float]]                     # (T, n_joints)
    ee_pose_world: Optional[List[List[List[float]]]] = None  # (T, 4, 4)
    # Bimanual / multi-link: precomputed world pose for each named link, (T, 4, 4)
    # per link. When present, fk_link_pose() indexes this directly (no FK backend
    # at solve time) — this is how the bimanual extractor bakes the per-frame
    # world_T_{l,r}_link6 poses the wrist cameras mount on.
    link_poses_world: Optional[Dict[str, List[List[List[float]]]]] = None

    @classmethod
    def load(cls, path: Path) -> "RobotTrajectory":
        d = json.loads(Path(path).read_text())
        return cls(
            urdf_path=d.get("urdf_path"),
            joint_names=d.get("joint_names", []),
            ee_link=d.get("ee_link"),
            fps=float(d.get("fps", 30.0)),
            joint_angles=d.get("joint_angles", []),
            ee_pose_world=d.get("ee_pose_world"),
            link_poses_world=d.get("link_poses_world"),
        )

    @property
    def T(self) -> int:
        if self.joint_angles:
            return len(self.joint_angles)
        if self.link_poses_world:
            return len(next(iter(self.link_poses_world.values())))
        if self.ee_pose_world is not None:
            return len(self.ee_pose_world)
        return 0


_XARM_LINK7_TO_LINK_TCP_Z = 0.172  # URDF: link_tcp = link7 + (0,0,+0.172) in link7 frame


def fk_link_pose(traj: RobotTrajectory, frame_t: int, link_name: str):
    """Forward-kinematics: pose of `link_name` in world frame at frame_t.

    Returns 4x4 numpy array. Raises if no FK backend is available.

    Short-circuit paths:
      1. link_name == traj.ee_link → index ee_pose_world directly.
      2. xArm-specific: link7 ↔ link_tcp differ by a pure (0,0,+0.172) z-translation
         in link7's local frame. If we have one and need the other, shift along
         the link's own +z axis (the third column of the rotation block).
    """
    import numpy as np
    # Precomputed multi-link poses (bimanual extractor bakes l_link6/r_link6 here).
    if traj.link_poses_world is not None and link_name in traj.link_poses_world:
        return np.asarray(traj.link_poses_world[link_name][frame_t], dtype=np.float64)
    if traj.ee_pose_world is None:
        raise NotImplementedError(
            "FK backend not wired up yet. Pass traj with ee_pose_world or "
            "extend fk_link_pose() with pinocchio/pybullet."
        )

    M = np.asarray(traj.ee_pose_world[frame_t], dtype=np.float64)
    if link_name == traj.ee_link:
        return M

    pair = (traj.ee_link, link_name)
    if pair == ("link_tcp", "link7"):
        # link7 = link_tcp shifted by -0.172 along link_tcp's +z (= link7's +z)
        M_out = M.copy()
        M_out[:3, 3] = M[:3, 3] - _XARM_LINK7_TO_LINK_TCP_Z * M[:3, 2]
        return M_out
    if pair == ("link7", "link_tcp"):
        M_out = M.copy()
        M_out[:3, 3] = M[:3, 3] + _XARM_LINK7_TO_LINK_TCP_Z * M[:3, 2]
        return M_out

    raise NotImplementedError(
        f"FK from ee_link={traj.ee_link!r} to link_name={link_name!r} not wired up. "
        "Extend fk_link_pose() (pinocchio/pybullet) or use a supported link."
    )
