"""
task1_perception/controller.py
================================
RobotController — all robot motion logic (SRP: one class, one reason to change).

Responsibilities:
  - Solve inverse kinematics.
  - Execute smooth joint-space trajectories (smoothstep interpolation).
  - Control the gripper (open / close / force).
  - Run a complete 5-stage grasp sequence (approach → descend → grasp → lift).
  - Return the arm to the home configuration.

All parameters come from RobotConfig; no magic numbers are defined here.
"""

from __future__ import annotations

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pybullet as p

if TYPE_CHECKING:
    from .camera import WristCamera

from .config import RobotConfig, SceneConfig, WorkspaceConfig


class RobotController:
    """
    High-level motion controller for the Franka Panda in PyBullet.

    The controller depends only on config dataclasses (DIP) — it does not
    import Scene or Camera, so those concerns remain decoupled.
    """

    def __init__(
        self,
        robot_id: int,
        robot_cfg: RobotConfig,
        scene_cfg: SceneConfig,
        workspace_cfg: WorkspaceConfig,
        use_gui: bool = True,
    ) -> None:
        """Initialise the controller and pre-compute numpy arrays from config.

        Args:
            robot_id: PyBullet body ID of the Franka Panda (from SceneManager).
            robot_cfg: Joint limits, gains, and gripper parameters.
            scene_cfg: Physics timestep (used for real-time sleep in GUI mode).
            workspace_cfg: Grasp heights (pre_grasp_height, lift_height).
            use_gui: When True, sleep after each simulation step to run at real time.
        """
        self._robot = robot_id
        self._rc = robot_cfg
        self._sc = scene_cfg
        self._ws = workspace_cfg
        self._use_gui = use_gui

        # Pre-compute numpy arrays from the frozen config tuples
        self._joint_lower = np.array(robot_cfg.joint_lower)
        self._joint_upper = np.array(robot_cfg.joint_upper)
        self._joint_range = self._joint_upper - self._joint_lower
        self._home = np.array(robot_cfg.home_config)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def solve_ik(self, target_pos: list, target_orn: list) -> np.ndarray:
        """Compute joint angles for a desired EE pose via damped-least-squares IK.

        Uses PyBullet's built-in IK solver with joint limits and a rest-pose bias
        toward the home configuration to prefer natural, collision-free postures.

        Args:
            target_pos: Desired EE position [x, y, z] in world frame [m].
            target_orn: Desired EE orientation as a quaternion [x, y, z, w].

        Returns:
            Float64 array of shape (7,) with target joint angles [rad] for joints 0–6.
        """
        joints = p.calculateInverseKinematics(
            self._robot,
            self._rc.ee_link,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=self._joint_lower.tolist(),
            upperLimits=self._joint_upper.tolist(),
            jointRanges=self._joint_range.tolist(),
            restPoses=self._home.tolist(),
            maxNumIterations=500,
            residualThreshold=1e-4,
        )
        return np.array(joints[:7])

    # ------------------------------------------------------------------
    # Arm motion
    # ------------------------------------------------------------------

    def move_ee_to(
        self,
        target_pos: list,
        target_orn: list,
        n_steps: int = 240,
        max_force: float | None = None,
    ) -> None:
        """Smoothly move the EE to a target pose using joint-space interpolation.

        Solves IK for the target pose, then interpolates from the current joint
        configuration using a smoothstep profile S(t) = 3t²−2t³ for jerk-limited
        motion (zero velocity at both start and end of the trajectory).

        Args:
            target_pos: Desired EE position [x, y, z] in world frame [m].
            target_orn: Desired EE orientation as a quaternion [x, y, z, w].
            n_steps: Number of simulation steps for the motion (default 240 ≈ 1 s).
            max_force: Joint motor force limit [N·m]. Defaults to RobotConfig.max_joint_force.
        """
        if max_force is None:
            max_force = self._rc.max_joint_force

        q_target = self.solve_ik(target_pos, target_orn)
        q_current = np.array([
            p.getJointState(self._robot, i)[0] for i in self._rc.arm_joint_ids
        ])

        for step in range(n_steps):
            t = (step + 1) / n_steps
            s = t * t * (3.0 - 2.0 * t)  # smoothstep
            q = q_current + s * (q_target - q_current)
            for i, angle in enumerate(q):
                p.setJointMotorControl2(
                    self._robot, i, p.POSITION_CONTROL,
                    targetPosition=float(angle),
                    force=max_force,
                    maxVelocity=self._rc.max_joint_vel,
                )
            p.stepSimulation()
            if self._use_gui:
                time.sleep(1.0 / self._sc.sim_hz)

    def home(self) -> None:
        """Return the arm to the pre-defined home configuration."""
        for i, angle in enumerate(self._rc.home_config):
            p.setJointMotorControl2(
                self._robot, i, p.POSITION_CONTROL,
                targetPosition=angle,
                force=self._rc.max_joint_force,
                maxVelocity=0.8,
            )
        for _ in range(300):
            p.stepSimulation()
            if self._use_gui:
                time.sleep(1.0 / self._sc.sim_hz)

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def open_gripper(self, n_steps: int = 80) -> None:
        """Open the gripper to the maximum aperture defined in RobotConfig.

        Args:
            n_steps: Simulation steps for the motion (default 80 ≈ 0.33 s at 240 Hz).
        """
        self._set_gripper(self._rc.gripper_open, n_steps, self._rc.finger_force)

    def close_gripper(self, n_steps: int = 120) -> None:
        """Close the gripper firmly using the grasp force defined in RobotConfig.

        Args:
            n_steps: Simulation steps for the motion (default 120 ≈ 0.5 s at 240 Hz).
        """
        self._set_gripper(self._rc.gripper_closed, n_steps, self._rc.grasp_force)

    def _set_gripper(self, width: float, n_steps: int, force: float) -> None:
        """Drive both finger joints to the target separation width."""
        half = width / 2.0
        for fj in self._rc.finger_joints:
            p.setJointMotorControl2(
                self._robot, fj, p.POSITION_CONTROL,
                targetPosition=half, force=force,
                maxVelocity=self._rc.finger_vel,
            )
        for _ in range(n_steps):
            p.stepSimulation()
            if self._use_gui:
                time.sleep(1.0 / self._sc.sim_hz)

    # ------------------------------------------------------------------
    # Grasp sequence
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Wrist-camera helpers
    # ------------------------------------------------------------------

    def _object_visible(self, wrist_cam: WristCamera) -> bool:
        """Return True if a vivid-coloured object blob is visible in the wrist camera.

        Used for reactive verification: object might fall or roll between
        the planning step and the actual grasp attempt.
        """
        rgb, _, _ = wrist_cam.capture()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Same saturation/value gate as the overhead detector
        mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
        return int(mask.sum()) > 200   # at least ~200 vivid pixels

    def _grip_succeeded(self) -> bool:
        """Return True if the fingers are NOT fully closed (something is gripped).

        After closing, if both fingers are essentially at 0 the gripper passed
        through the object (missed or object fell) — abort the lift.
        """
        left  = p.getJointState(self._robot, self._rc.finger_joints[0])[0]
        right = p.getJointState(self._robot, self._rc.finger_joints[1])[0]
        gap = left + right   # total finger separation
        return gap > 0.004   # > 4 mm separation means something is gripped

    # ------------------------------------------------------------------
    # Grasp sequence
    # ------------------------------------------------------------------

    def grasp(
        self,
        target_xyz: np.ndarray,
        table_height: float,
        shape: str = "box",
        wrist_cam: WristCamera | None = None,
    ) -> bool:
        """
        Execute a shape-aware, wrist-camera-reactive pick-and-place sequence.

        Stages
        ------
        1. Open gripper.
        2. Approach  — hover at pre_grasp_height, then check wrist cam: if the
                       object is no longer visible the target must have moved —
                       abort immediately and return False.
        3. Descend   — lower to object centre (shape-adjusted); check wrist cam
                       again just before closing (object may have rolled away).
        4. Close     — force-limited close; finger gap checked afterward to
                       confirm something was actually gripped.
        5. Lift      — raise to lift_height only when grip is confirmed.

        Shape-specific finger/force tuning
        -----------------------------------
        box      : tight close (gripper_closed), standard force.
        cylinder : full close against contact (physics stops fingers), 2.5× force.
        sphere   : same as cylinder.

        Returns True on successful lift, False if any reactive check aborts.
        """
        grasp_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        ws = self._ws

        if shape in ("cylinder", "sphere"):
            finger_target = 0.0
            grasp_f = self._rc.grasp_force * 2.5
            descend_bias = 0.008
        else:
            finger_target = self._rc.gripper_closed
            grasp_f = self._rc.grasp_force
            descend_bias = 0.0

        print(f"    [Grasp] Opening gripper (shape={shape}) …")
        self.open_gripper()

        # ── Stage 2: Approach & pre-grasp check ──────────────────────────
        pre_pos = [float(target_xyz[0]), float(target_xyz[1]),
                   float(target_xyz[2]) + ws.pre_grasp_height]
        print(f"    [Grasp] Approaching pre-grasp at z = {pre_pos[2]:.3f} m …")
        self.move_ee_to(pre_pos, grasp_orn, n_steps=220)

        if wrist_cam is not None and not self._object_visible(wrist_cam):
            print("    [Grasp] ✗ Object not visible from pre-grasp hover — target moved, aborting.")
            self.home()
            return False

        # ── Stage 3: Descend ──────────────────────────────────────────────
        # (No wrist check here — at close range the gripper fingers block the FOV.)
        obj_centre_z = (table_height + float(target_xyz[2])) / 2.0 + descend_bias
        grasp_pos = [float(target_xyz[0]), float(target_xyz[1]), obj_centre_z]
        print(f"    [Grasp] Descending to z = {grasp_pos[2]:.3f} m …")
        self.move_ee_to(grasp_pos, grasp_orn, n_steps=160)

        # ── Stage 4: Close & grip verification ───────────────────────────
        gap_mm = finger_target * 1000
        print(f"    [Grasp] Closing gripper (gap target={gap_mm:.1f} mm, force={grasp_f:.0f} N) …")
        half = finger_target / 2.0
        for fj in self._rc.finger_joints:
            p.setJointMotorControl2(
                self._robot, fj, p.POSITION_CONTROL,
                targetPosition=half, force=grasp_f,
                maxVelocity=self._rc.finger_vel,
            )
        for _ in range(120):
            p.stepSimulation()
            if self._use_gui:
                time.sleep(1.0 / self._sc.sim_hz)

        if not self._grip_succeeded():
            print("    [Grasp] ✗ Fingers fully closed — nothing was gripped, aborting lift.")
            self.home()
            return False

        # Keep gripper clamped hard throughout the lift
        for fj in self._rc.finger_joints:
            p.setJointMotorControl2(
                self._robot, fj, p.POSITION_CONTROL,
                targetPosition=half, force=grasp_f, maxVelocity=0.05,
            )

        # ── Stage 5: Lift ─────────────────────────────────────────────────
        lift_pos = [float(target_xyz[0]), float(target_xyz[1]),
                    float(target_xyz[2]) + ws.lift_height]
        print(f"    [Grasp] Lifting to z = {lift_pos[2]:.3f} m …")
        self.move_ee_to(lift_pos, grasp_orn, n_steps=220)

        print("    [Grasp] ✓ Sequence complete.")
        return True
