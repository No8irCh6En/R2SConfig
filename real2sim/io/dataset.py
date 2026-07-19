"""Multi-cam dataset wrapper (the input to the new pose-estimation pipeline).

Loads an `assets/scenes/<scene_id>/` directory previously populated by
scripts/extract_lerobot.py:

    assets/scenes/<scene_id>/
      scene.jpg                  # single representative frame (for back-compat
                                 # with the old single-image steps; usually
                                 # cam_<id>/frames/000000.png of a fixed cam)
      cameras.json               # CameraSet (mixed fixed + attached)
      robot_traj.json            # RobotTrajectory (per-frame joints, optional EE)
      cam_<id>/
        frames/000000.png ... NNNNNN.png

Use as:

    bundle = MultiCamBundle.load(scene_id)
    for cam in bundle.cameras.cameras:
        for t in range(bundle.T):
            img = bundle.read_frame(cam.name, t)
            R, T_view = bundle.world_camera_at(cam, t)    # numpy, PT3D row-vec

PT3D convention reminder:
    v_view = v_world @ R + T_view
    u      = -fx * P_view[0] / P_view[2] + cx
    v      = -fy * P_view[1] / P_view[2] + cy

`fixed` cam:
    R = [cam_x | cam_y | cam_z]_world  (columns built from look_at)
    T = -eye @ R

`attached` cam (link rigid-mounted, e.g. wrist):
    M_world_cam(t) = M_world_link(t) @ link_offset_T   (4x4 SE(3))
    # ↑ link_offset_T comes from Genesis (`pos_lookat_up_to_T` uses
    #   z = pos - lookat, i.e. col 2 is BACK = OpenGL convention,
    #   cols = right, up, back). Convert to PT3D (col 2 = forward) with
    #   diag([-1, 1, -1]) — keeps det = +1 (PT3D is left-handed: cam +X
    #   ends up image-left because of the `-fx` in the pinhole formula).
    R = M_world_cam[:3, :3] @ diag([-1, 1, -1])
    T = -M_world_cam[:3, 3] @ R

Single source of truth for the multi-cam silhouette loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .cameras import CameraSet, CameraSpec
from .robot import RobotTrajectory, fk_link_pose


@dataclass
class MultiCamBundle:
    scene_dir: Path
    cameras: CameraSet
    robot: RobotTrajectory

    @classmethod
    def load(cls, scene_dir: Path) -> "MultiCamBundle":
        scene_dir = Path(scene_dir)
        return cls(
            scene_dir=scene_dir,
            cameras=CameraSet.load(scene_dir / "cameras.json"),
            robot=RobotTrajectory.load(scene_dir / "robot_traj.json"),
        )

    @property
    def T(self) -> int:
        return self.robot.T

    def frames_dir(self, cam_name: str) -> Path:
        return self.scene_dir / f"cam_{cam_name}" / "frames"

    def read_frame(self, cam_name: str, t: int):
        from PIL import Image
        p = self.frames_dir(cam_name) / f"{t:06d}.png"
        return Image.open(p).convert("RGB")

    def world_camera_at(
        self, cam: Union[str, CameraSpec], t: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World→view (R, T) in PyTorch3D row-vec form for `cam` at frame `t`.

        Returns numpy float64 arrays: R is (3,3), T is (3,). `cam` can be a
        CameraSpec or a name. For fixed cams, `t` is ignored.
        """
        if isinstance(cam, str):
            cam = self.cameras.by_name(cam)

        if cam.kind == "fixed":
            if cam.world_pos is None or cam.world_lookat is None or cam.world_up is None:
                raise ValueError(
                    f"fixed cam {cam.name!r} missing world_pos/world_lookat/world_up"
                )
            return _look_at_pt3d(
                np.asarray(cam.world_pos, dtype=np.float64),
                np.asarray(cam.world_lookat, dtype=np.float64),
                np.asarray(cam.world_up, dtype=np.float64),
            )

        if cam.kind == "attached":
            if cam.link_name is None or cam.link_offset_T is None:
                raise ValueError(
                    f"attached cam {cam.name!r} missing link_name/link_offset_T"
                )
            M_world_link = fk_link_pose(self.robot, t, cam.link_name)        # (4,4)
            M_link_cam = np.asarray(cam.link_offset_T, dtype=np.float64)     # (4,4)
            M_world_cam = M_world_link @ M_link_cam
            # Genesis cam basis (right, up, back) → PT3D (cam +X, +Y, +Z forward).
            # See Genesis/genesis/utils/geom.py::pos_lookat_up_to_T (z = pos - lookat).
            R = M_world_cam[:3, :3] @ np.diag([-1.0, 1.0, -1.0])
            eye = M_world_cam[:3, 3]
            T_view = -eye @ R
            return R, T_view

        raise ValueError(f"unknown cam kind: {cam.kind!r}")


def _look_at_pt3d(eye: np.ndarray, at: np.ndarray, up: np.ndarray):
    """Numpy reimpl of PyTorch3D look_at_view_transform.

    Returns (R, T) with v_view = v_world @ R + T. R has the camera basis as
    columns: R[:, 0] = cam +X in world, R[:, 1] = cam +Y, R[:, 2] = cam +Z
    (look direction). This is exactly what PT3D's look_at_rotation returns,
    so the result drops straight into the existing scene_geometry math.
    """
    z = at - eye
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)         # (3,3), cam basis as cols
    T = -eye @ R                            # (3,)
    return R, T
