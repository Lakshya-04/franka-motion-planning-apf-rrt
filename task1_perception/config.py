"""
task1_perception/config.py
==========================
Frozen dataclasses for every tunable constant in the perception pipeline.

Grouping constants into typed, immutable config objects serves two purposes:
1. SOLID — Single Responsibility: all configuration lives here, nowhere else.
2. Safety — frozen=True prevents accidental mutation during a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Intrinsic + near/far clipping for every virtual camera in the scene."""

    width: int = 224
    height: int = 224
    near: float = 0.1
    far: float = 3.1
    fov_deg: float = 60.0

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class SceneConfig:
    """Physics simulation and scene geometry constants."""

    sim_hz: float = 240.0
    table_height: float = 0.625  # surface Z of table/table.urdf above the plane
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class RobotConfig:
    """Franka Panda-specific joint and gripper parameters."""

    ee_link: int = 11                               # panda_hand link index
    arm_joint_ids: Tuple[int, ...] = tuple(range(7))  # joints 0–6 (7-DOF arm)
    finger_joints: Tuple[int, int] = (9, 10)        # panda_finger_joint1/2

    gripper_open: float = 0.08    # m — max finger separation
    gripper_closed: float = 0.001  # m — nearly fully closed

    # Joint limits [rad] — used for IK rest-pose bias
    joint_lower: Tuple[float, ...] = (
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    )
    joint_upper: Tuple[float, ...] = (
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    )
    home_config: Tuple[float, ...] = (
        0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785
    )

    max_joint_force: float = 87.0    # N·m
    finger_force: float = 20.0       # N — normal gripper hold
    grasp_force: float = 40.0        # N — firm grasp during lift
    max_joint_vel: float = 1.2       # rad/s
    finger_vel: float = 0.3          # m/s
    finger_lateral_friction: float = 2.0


@dataclass(frozen=True)
class WorkspaceConfig:
    """Reachable workspace filter applied after back-projection."""

    reach_x: Tuple[float, float] = (0.28, 0.72)
    reach_y: Tuple[float, float] = (-0.28, 0.28)
    z_min_above_table: float = 0.01   # m above table surface
    z_max_above_table: float = 0.12   # max reasonable cube height

    pre_grasp_height: float = 0.15    # m — hover height before descend
    lift_height: float = 0.25         # m — lift height after grasp


@dataclass(frozen=True)
class ColourPalette:
    """HSV hue ranges for colour classification."""

    entries: Tuple[Tuple[str, int, int], ...] = field(default_factory=lambda: (
        ("red",    0,  10),
        ("orange", 11, 25),
        ("yellow", 26, 35),
        ("green",  36, 85),
        ("cyan",   86, 100),
        ("blue",  101, 130),
        ("purple", 131, 160),
        ("pink",  161, 180),
    ))
    min_saturation: int = 60
    min_value: int = 60
    patch_size: int = 10
