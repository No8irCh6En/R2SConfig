"""Camera config for the multi-cam workflow (minimal schema).

Loaded from `assets/scenes/<scene_id>/cameras.json`, written by the LeRobot
extractor (scripts/extract_lerobot.py).

Two kinds:

    fixed: camera bolted to the world. World→cam constant.
        {
          "name": "ego",
          "kind": "fixed",
          "fov_deg": 64.81,                  # vertical FOV (Genesis convention)
          "W": 320, "H": 240,
          "world_pos":    [x, y, z],         # Genesis z-up world
          "world_lookat": [x, y, z],
          "world_up":     [x, y, z]
        }

    attached: camera rigidly mounted on a robot link (e.g. wrist cam).
        {
          "name": "wrist",
          "kind": "attached",
          "fov_deg": 64.5,
          "W": 320, "H": 240,
          "link_name":     "link7",                 # URDF link
          "link_offset_T": [[..4x4..]]              # link→cam SE(3)
        }

Per-frame world pose of an attached cam at frame t:
    world_T_cam(t) = world_T_link(t) @ link_offset_T
                   = robot.ee_pose_world(t) @ link_offset_T   (if link == ee_link)

Intrinsics are derived from fov_deg + (W, H) using the same formula as Genesis
gs_camera.py:187 — `fy = (H/2) / tan(fov_v/2);  fx = fy;  cx, cy = W/2, H/2`.
We deliberately do NOT store fx/fy/cx/cy: resolution may legitimately change
(e.g. dataset 320x240 vs sim render 640x480) and we want a single source of
truth (fov + actual res of the image you're loading).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class CameraSpec:
    name: str
    kind: Literal["fixed", "attached"]
    fov_deg: float
    W: int
    H: int
    # fixed-only
    world_pos: Optional[List[float]] = None
    world_lookat: Optional[List[float]] = None
    world_up: Optional[List[float]] = None
    # attached-only
    link_name: Optional[str] = None
    link_offset_T: Optional[List[List[float]]] = None  # 4x4

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSpec":
        return cls(
            name=d["name"], kind=d["kind"],
            fov_deg=float(d["fov_deg"]),
            W=int(d["W"]), H=int(d["H"]),
            world_pos=d.get("world_pos"),
            world_lookat=d.get("world_lookat"),
            world_up=d.get("world_up"),
            link_name=d.get("link_name"),
            link_offset_T=d.get("link_offset_T"),
        )

    def intrinsics(self, W: Optional[int] = None, H: Optional[int] = None):
        """Compute (fx, fy, cx, cy) for given (W, H) — defaults to stored W/H.

        Pass W/H explicitly when loading a higher-res render than the stored
        canonical (e.g. you stored 320x240 but you're now feeding 640x480).
        """
        W = self.W if W is None else W
        H = self.H if H is None else H
        fy = (H / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        fx = fy                              # square pixels (Genesis convention)
        cx, cy = W / 2.0, H / 2.0
        return fx, fy, cx, cy


@dataclass
class CameraSet:
    cameras: List[CameraSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CameraSet":
        data = json.loads(Path(path).read_text())
        return cls(cameras=[CameraSpec.from_dict(c) for c in data["cameras"]])

    def by_kind(self, kind: str) -> List[CameraSpec]:
        return [c for c in self.cameras if c.kind == kind]

    def by_name(self, name: str) -> CameraSpec:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)
