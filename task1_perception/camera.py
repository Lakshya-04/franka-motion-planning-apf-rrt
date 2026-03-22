"""
task1_perception/camera.py
==========================
Camera abstraction and projection math (SRP: all camera concerns live here).

Classes
-------
Camera       — static virtual camera with a fixed view matrix.
WristCamera  — camera that follows the robot end-effector (OCP/LSP: extends
               Camera without breaking the base contract).

Standalone helpers
------------------
pixel_to_world  — back-projects a pixel + depth value to a 3-D world point.
                  This is a pure function; it does not belong to any class.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from .config import CameraConfig


# ===========================================================================
# Projection helper
# ===========================================================================

def pixel_to_world(
    view_matrix,
    proj_matrix,
    pixel_x: int,
    pixel_y: int,
    depth_img: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    """Back-project a pixel and OpenGL depth value to a 3-D world coordinate.

    Args:
        view_matrix: Column-major 4×4 view matrix from pybullet.computeViewMatrix.
        proj_matrix: Column-major 4×4 projection matrix from pybullet.computeProjectionMatrixFOV.
        pixel_x: Horizontal pixel coordinate (0 = left).
        pixel_y: Vertical pixel coordinate (0 = top).
        depth_img: Float32 H×W array of raw OpenGL depth buffer values in [0, 1].
        near: Near clipping plane distance [m].
        far: Far clipping plane distance [m].

    Returns:
        3-D world coordinate as a float64 array of shape (3,) in metres.

    Note:

        Coordinate transform chain:
            pixel → NDC → eye space (via inv projection) → world (via inv view)

    Step 1 — Linearise the non-linear OpenGL depth buffer value d_raw ∈ [0,1]:
        depth_m = far·near / (far − (far−near)·d_raw)

    Step 2 — Convert metric depth to NDC z (fixes the common "2*d−1" bug):
        z_ndc = (far+near)/(far−near)  −  2·far·near / ((far−near)·depth_m)

    Steps 3–4 — Standard homogeneous back-projection through inv(P) and inv(V).
    """
    h, w = depth_img.shape[:2]
    d_raw = float(depth_img[pixel_y, pixel_x])

    depth_m = far * near / (far - (far - near) * d_raw)
    depth_m = float(np.clip(depth_m, near + 1e-6, far - 1e-6))

    ndc_x = (2.0 * pixel_x / w) - 1.0
    ndc_y = 1.0 - (2.0 * pixel_y / h)
    ndc_z = (far + near) / (far - near) - 2.0 * far * near / ((far - near) * depth_m)

    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0])

    proj = np.array(proj_matrix).reshape(4, 4).T
    eye = np.linalg.inv(proj) @ clip
    eye /= eye[3]

    view = np.array(view_matrix).reshape(4, 4).T
    world = (np.linalg.inv(view) @ eye)[:3]
    return world


# ===========================================================================
# Camera classes
# ===========================================================================

class Camera:
    """
    A static virtual camera with a fixed view and projection matrix.

    Responsibilities (SRP):
      - Hold view + projection matrices.
      - Capture RGB-D frames from PyBullet.
      - Expose pixel_to_world back-projection.
    """

    def __init__(self, view_matrix, proj_matrix, cfg: CameraConfig) -> None:
        """Initialise with pre-computed view/projection matrices and camera config.

        Args:
            view_matrix: Column-major 4×4 view matrix (e.g. from computeViewMatrix).
            proj_matrix: Column-major 4×4 projection matrix (e.g. from computeProjectionMatrixFOV).
            cfg: Camera intrinsic parameters (resolution, near/far planes, FOV).
        """
        self._view = view_matrix
        self._proj = proj_matrix
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture one frame from the virtual camera.

        Returns:
            Tuple of (rgb, depth, seg) where:
                rgb   is uint8  H×W×3  — RGB image (no alpha channel).
                depth is float32 H×W   — raw OpenGL depth buffer values in [0, 1].
                seg   is int32  H×W   — segmentation mask (PyBullet body IDs).
        """
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self._cfg.width,
            height=self._cfg.height,
            viewMatrix=self._view,
            projectionMatrix=self._proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(
            self._cfg.height, self._cfg.width, 4
        )[:, :, :3]
        depth_arr = np.array(depth, dtype=np.float32).reshape(
            self._cfg.height, self._cfg.width
        )
        seg_arr = np.array(seg, dtype=np.int32).reshape(
            self._cfg.height, self._cfg.width
        )
        return rgb_arr, depth_arr, seg_arr

    # ------------------------------------------------------------------
    # Back-projection convenience wrapper
    # ------------------------------------------------------------------

    def pixel_to_world(self, px: int, py: int, depth_img: np.ndarray) -> np.ndarray:
        """Back-project a pixel to a 3-D world coordinate using this camera's matrices.

        Convenience wrapper around the module-level pixel_to_world function.

        Args:
            px: Horizontal pixel coordinate.
            py: Vertical pixel coordinate.
            depth_img: Float32 H×W raw OpenGL depth buffer.

        Returns:
            3-D world coordinate as a float64 array of shape (3,) in metres.
        """
        return pixel_to_world(
            self._view, self._proj, px, py, depth_img,
            self._cfg.near, self._cfg.far,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def view_matrix(self):
        return self._view

    @property
    def proj_matrix(self):
        return self._proj

    @property
    def cfg(self) -> CameraConfig:
        return self._cfg


class WristCamera(Camera):
    """
    Camera that follows the robot end-effector (LSP: fully substitutable for Camera).

    The Panda grasp-target frame has its +Z axis pointing in the APPROACH
    direction (downward when grasping from above).  The camera sits 6 cm
    behind the hand (along −Z) and looks 35 cm ahead (along +Z).
    """

    def __init__(self, robot_id: int, ee_link: int, proj_matrix, cfg: CameraConfig) -> None:
        """Initialise with a reference to the robot so the view can be updated each frame.

        Args:
            robot_id: PyBullet body ID of the robot.
            ee_link: Link index of the end-effector (e.g. 11 for panda_hand).
            proj_matrix: Shared projection matrix (same intrinsics as the overhead camera).
            cfg: Camera resolution and clipping parameters.
        """
        dummy_view = p.computeViewMatrix([0, 0, 1], [0, 0, 0], [0, 1, 0])
        super().__init__(dummy_view, proj_matrix, cfg)
        self._robot_id = robot_id
        self._ee_link = ee_link

    def update_view(self) -> None:
        """Recompute the view matrix from the current end-effector pose.

        Positions the virtual camera 6 cm behind the hand (along −Z_ee) and
        looks 35 cm ahead (along +Z_ee), which is toward the workspace when
        the gripper points downward.
        """
        state = p.getLinkState(self._robot_id, self._ee_link,
                               computeForwardKinematics=True)
        ee_pos = np.array(state[4])
        rot = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)

        eye = ee_pos + rot @ np.array([0.0, 0.0, -0.06])
        target = ee_pos + rot @ np.array([0.0, 0.0, 0.35])
        up = rot @ np.array([0.0, 1.0, 0.0])

        self._view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update view matrix from the current EE pose, then capture a frame.

        Returns:
            Tuple of (rgb, depth, seg) — same semantics as Camera.capture().
        """
        self.update_view()
        return super().capture()
