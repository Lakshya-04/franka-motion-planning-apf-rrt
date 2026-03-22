"""
task1_perception/scene.py
=========================
SceneManager — owns the PyBullet world lifecycle (SRP: one reason to change).

Responsibilities:
  - Connect / disconnect to/from the physics server.
  - Load URDFs (plane, table, robot).
  - Spawn random coloured objects (boxes, cylinders, spheres).
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


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (all in [0,1]) to RGB tuple without importing colorsys."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sector = i % 6
    return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[sector]


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
        # Maps body_id → shape string ("box", "cylinder", "sphere")
        self.shape_map: dict[int, str] = {}

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
    # Object spawning
    # ------------------------------------------------------------------

    def spawn_random_cubes(self, num_cubes: int) -> list[int]:
        """Spawn randomly placed, sized, and coloured cubes (kept for compatibility)."""
        return self.spawn_random_objects(num_cubes)

    def spawn_random_objects(self, num_objects: int) -> list[int]:
        """Spawn a mix of boxes, cylinders, and spheres on the table surface.

        Objects are distributed evenly across the three shape classes.  Each
        object gets a vivid random RGB colour, random size (15–28 mm footprint
        radius), and a random XY position within the reachable table area.

        Args:
            num_objects: Total number of objects to create (rounded down per shape).

        Returns:
            List of PyBullet body IDs for the spawned objects, in creation order.
        """
        table_cx = 0.5
        half_y = 0.30
        table_z = self._scene_cfg.table_height
        body_ids: list[int] = []
        self.shape_map.clear()

        shapes = ["box", "cylinder", "sphere"]
        # Distribute shapes evenly across the requested count
        shape_sequence = [shapes[i % len(shapes)] for i in range(num_objects)]
        random.shuffle(shape_sequence)

        for shape in shape_sequence:
            size = random.uniform(0.018, 0.028)
            pos_x = random.uniform(table_cx - 0.18, table_cx + 0.18)
            pos_y = random.uniform(-half_y + 0.05, half_y - 0.05)

            # Choose a vivid colour (high saturation: avoid dull near-grey)
            hue = random.uniform(0, 1)
            colour = list(_hsv_to_rgb(hue, 1.0, 0.9)) + [1.0]

            if shape == "box":
                pos = [pos_x, pos_y, table_z + size]
                orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, math.pi)])
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[size] * 3, rgbaColor=colour
                )
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size] * 3)
                bid = p.createMultiBody(
                    baseMass=0.1, baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn,
                )

            elif shape == "cylinder":
                height = size * 2.0   # height = diameter so it looks like a puck
                pos = [pos_x, pos_y, table_z + height / 2.0]
                orn = p.getQuaternionFromEuler([0, 0, 0])
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=size, length=height, rgbaColor=colour
                )
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=size, height=height
                )
                bid = p.createMultiBody(
                    baseMass=0.1, baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn,
                )

            else:  # sphere
                pos = [pos_x, pos_y, table_z + size]
                orn = p.getQuaternionFromEuler([0, 0, 0])
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=size, rgbaColor=colour
                )
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
                bid = p.createMultiBody(
                    baseMass=0.1, baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn,
                )

            p.changeDynamics(bid, -1, lateralFriction=1.5)
            body_ids.append(bid)
            self.shape_map[bid] = shape

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
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.3],
        )
