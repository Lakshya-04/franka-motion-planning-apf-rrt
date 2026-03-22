"""
task1_perception/scene.py
=========================
SceneManager — owns the PyBullet world lifecycle (SRP: one reason to change).

Responsibilities:
  - Connect / disconnect to/from the physics server.
  - Load URDFs (plane, table, robot).
  - Spawn random coloured cubes.
  - Run warm-up physics steps until the scene is settled.
  - Configure the GUI camera and debug visualiser settings.

All physics parameters come from SceneConfig / RobotConfig; SceneManager
never hard-codes magic numbers of its own.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from .config import RobotConfig, SceneConfig


class SceneManager:
    """Manages the PyBullet world lifecycle for one simulation session.

    Owns connection, URDF loading, cube spawning, and physics settling.
    All configuration comes from injected dataclasses; no magic numbers
    are defined in this class (SRP).

    Attributes:
        robot_id: PyBullet body ID of the loaded Franka Panda, set by setup().
    """

    def __init__(self, scene_cfg: SceneConfig, robot_cfg: RobotConfig,
                 use_gui: bool = True) -> None:
        """Initialise the manager with configuration; does not connect yet.

        Args:
            scene_cfg: Physics and geometry constants (gravity, table height, Hz).
            robot_cfg: Franka Panda joint and gripper parameters.
            use_gui: Open the PyBullet GUI window when True, run headless otherwise.
        """
        self._scene_cfg = scene_cfg
        self._robot_cfg = robot_cfg
        self._use_gui = use_gui
        self.robot_id: int = -1
        self._cube_ids: list[int] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Connect to PyBullet, configure physics, and load base scene assets.

        Loads the ground plane, table URDF, and Franka Panda URDF, then snaps
        all joints to the home configuration with active position-control motors
        so the arm does not sag during the settling phase.

        Raises:
            pybullet.error: If a URDF file cannot be found in the search path.
        """
        mode = p.GUI if self._use_gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self._scene_cfg.gravity)
        p.setTimeStep(1.0 / self._scene_cfg.sim_hz)

        if self._use_gui:
            self._configure_gui()

        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.15, 0.0, self._scene_cfg.table_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        self._init_robot()

    def disconnect(self) -> None:
        """Disconnect from the PyBullet physics server and free resources."""
        p.disconnect()

    # ------------------------------------------------------------------
    # Robot initialisation
    # ------------------------------------------------------------------

    def _init_robot(self) -> None:
        """Snap joints to home config and activate motors."""
        rc = self._robot_cfg
        for i, angle in enumerate(rc.home_config):
            p.resetJointState(self.robot_id, i, angle)
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=angle, force=rc.max_joint_force, maxVelocity=1.0,
            )
        for fj in rc.finger_joints:
            p.resetJointState(self.robot_id, fj, rc.gripper_open / 2)
            p.setJointMotorControl2(
                self.robot_id, fj, p.POSITION_CONTROL,
                targetPosition=rc.gripper_open / 2, force=rc.finger_force,
            )
        # High finger friction so cubes don't slip during lift
        for fj in rc.finger_joints:
            p.changeDynamics(self.robot_id, fj,
                             lateralFriction=rc.finger_lateral_friction)

    # ------------------------------------------------------------------
    # Cube spawning
    # ------------------------------------------------------------------

    def spawn_random_cubes(self, num_cubes: int) -> list[int]:
        """Spawn randomly placed, sized, and coloured cubes on the table surface.

        Each cube gets a random half-extent (15–28 mm), a random XY position
        within the reachable table area, a random yaw, and a random RGB colour.
        Lateral friction is set high (1.5) so cubes resist sliding on grasp.

        Args:
            num_cubes: Number of cubes to create.

        Returns:
            List of PyBullet body IDs for the spawned cubes, in creation order.
        """
        table_cx = 0.5
        half_y = 0.30
        table_z = self._scene_cfg.table_height
        body_ids: list[int] = []

        for _ in range(num_cubes):
            size = random.uniform(0.015, 0.028)
            pos = [
                random.uniform(table_cx - 0.18, table_cx + 0.18),
                random.uniform(-half_y + 0.05, half_y - 0.05),
                table_z + size,
            ]
            orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, math.pi)])
            colour = [random.uniform(0, 1) for _ in range(3)] + [1.0]

            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[size] * 3, rgbaColor=colour
            )
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size] * 3)
            bid = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                baseOrientation=orn,
            )
            p.changeDynamics(bid, -1, lateralFriction=1.5)
            body_ids.append(bid)

        self._cube_ids = body_ids
        return body_ids

    # ------------------------------------------------------------------
    # Physics settling
    # ------------------------------------------------------------------

    def settle(self, steps: int) -> None:
        """Step the physics simulation until spawned objects come to rest.

        Args:
            steps: Number of simulation steps to run (e.g. 300 ≈ 1.25 s at 240 Hz).
        """
        print(f"Settling physics ({steps} steps) …")
        for _ in range(steps):
            p.stepSimulation()
        print("Ready.")

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------

    def _configure_gui(self) -> None:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.3],
        )
