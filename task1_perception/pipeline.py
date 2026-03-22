"""
task1_perception/pipeline.py
==============================
PickAndPlacePipeline — orchestrates all subsystems (DIP: depends on abstractions).

This module is the only place that knows about the topology of the full system:
  SceneManager → overhead Camera → WristCamera → ObjectDetector → RobotController

Separation of concerns:
  - pipeline.py  → orchestration (the "glue")
  - scene.py     → world lifecycle
  - camera.py    → frame capture + back-projection
  - detection.py → vision / segmentation
  - controller.py→ robot motion

The display helper (show_cameras / init_display) also lives here because it
is the only consumer; if it were needed elsewhere it would move to its own
module.
"""

from __future__ import annotations

import numpy as np
import cv2
import pybullet as p

from .config import (
    CameraConfig,
    ColourPalette,
    RobotConfig,
    SceneConfig,
    WorkspaceConfig,
)
from .scene import SceneManager
from .camera import Camera, WristCamera
from .detection import Detection, ObjectDetector
from .controller import RobotController


# ---------------------------------------------------------------------------
# Display helpers (SRP: kept here as they are pipeline-internal concerns)
# ---------------------------------------------------------------------------

_WIN = "Perception  |  Overhead (left)   Wrist (right)"


def _init_display(cam_cfg: CameraConfig) -> None:
    """Create and size the composite OpenCV display window."""
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN, cam_cfg.width * 2 + 10, cam_cfg.height + 30)


def _show_cameras(
    cam_cfg: CameraConfig,
    overhead_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    detections: list[Detection] | None = None,
) -> bool:
    """
    Composite overhead + wrist feeds into a single OpenCV window.
    Draws detection markers with colour labels.
    Returns False when 'q' is pressed.
    """
    overhead_bgr = cv2.cvtColor(overhead_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

    if detections:
        for det in detections:
            cv2.drawMarker(overhead_bgr, (det.pixel_x, det.pixel_y),
                           (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
            label = f"{det.shape}/{det.colour}"
            cv2.putText(overhead_bgr, label,
                        (det.pixel_x + 6, det.pixel_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    label_bar = np.zeros((30, cam_cfg.width * 2 + 10, 3), dtype=np.uint8)
    for text, x in [("Overhead", 10), ("Wrist", cam_cfg.width + 20)]:
        cv2.putText(label_bar, text, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    sep = np.full((cam_cfg.height, 10, 3), 40, dtype=np.uint8)
    combined = np.vstack([label_bar, np.hstack([overhead_bgr, sep, wrist_bgr])])
    cv2.imshow(_WIN, combined)
    return cv2.waitKey(1) & 0xFF != ord("q")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PickAndPlacePipeline:
    """
    Orchestrates the full perception → planning → execution cycle.

    Dependency Inversion: the pipeline is constructed with pre-built
    subsystems, so each can be replaced independently in tests or future work.
    """

    def __init__(
        self,
        scene: SceneManager,
        overhead_cam: Camera,
        wrist_cam: WristCamera,
        detector: ObjectDetector,
        controller: RobotController,
        scene_cfg: SceneConfig,
        cam_cfg: CameraConfig,
        use_gui: bool = True,
    ) -> None:
        self._scene = scene
        self._overhead = overhead_cam
        self._wrist = wrist_cam
        self._detector = detector
        self._controller = controller
        self._scene_cfg = scene_cfg
        self._cam_cfg = cam_cfg
        self._use_gui = use_gui

    def run(self, num_cubes: int = 5, sim_steps: int = 300) -> None:
        """Spawn cubes, detect, and grasp each one in turn."""
        self._scene.spawn_random_objects(num_cubes)
        self._scene.settle(sim_steps)

        if self._use_gui:
            _init_display(self._cam_cfg)

        overhead_rgb, overhead_depth, overhead_seg = self._overhead.capture()
        wrist_rgb, _, _ = self._wrist.capture()

        table_h = self._scene_cfg.table_height
        detections = self._detector.detect(
            self._overhead, overhead_rgb, overhead_depth, table_h, overhead_seg
        )

        print(
            f"\nDetected {len(detections)} object(s) within reachable workspace:"
        )
        for i, det in enumerate(detections):
            print(
                f"  [{i}] pixel ({det.pixel_x:3d},{det.pixel_y:3d})  →  "
                f"world [{det.world_xyz[0]:.3f}, {det.world_xyz[1]:.3f}, "
                f"{det.world_xyz[2]:.3f}]  shape: {det.shape:<8s}  colour: {det.colour}"
            )

        if self._use_gui:
            _show_cameras(self._cam_cfg, overhead_rgb, wrist_rgb, detections)
            cv2.waitKey(1500)

        for idx, det in enumerate(detections):
            target = det.world_xyz.copy()
            target[2] = max(target[2], table_h + 0.015)

            print(
                f"\nGrasping object {idx} ({det.shape}/{det.colour}) at "
                f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] …"
            )

            self._controller.grasp(target, table_h, shape=det.shape,
                                   wrist_cam=self._wrist)

            if self._use_gui:
                for _ in range(120):
                    p.stepSimulation()
                    oh_rgb2, _, _ = self._overhead.capture()
                    wr_rgb2, _, _ = self._wrist.capture()
                    if not _show_cameras(self._cam_cfg, oh_rgb2, wr_rgb2):
                        break
                    import time
                    time.sleep(1.0 / self._scene_cfg.sim_hz)

            print("    Returning to home pose …")
            self._controller.open_gripper()
            self._controller.home()

        print("\nAll objects processed. Done.")
        if self._use_gui:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Factory + entry point
# ---------------------------------------------------------------------------

def build_pipeline(use_gui: bool = True) -> PickAndPlacePipeline:
    """Construct a fully wired PickAndPlacePipeline from default configs.

    Acts as the composition root: instantiates all config dataclasses and
    subsystem objects, wires them together, and returns a ready-to-run pipeline.
    Separating construction from orchestration (DIP) makes it trivial to swap
    any subsystem for a mock or alternative implementation.

    Args:
        use_gui: Open the PyBullet GUI window when True, headless otherwise.

    Returns:
        A fully initialised PickAndPlacePipeline connected to a live PyBullet session.
    """
    scene_cfg = SceneConfig()
    robot_cfg = RobotConfig()
    cam_cfg = CameraConfig()
    workspace_cfg = WorkspaceConfig()
    palette = ColourPalette()

    scene = SceneManager(scene_cfg, robot_cfg, use_gui)
    scene.setup()

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=cam_cfg.fov_deg,
        aspect=cam_cfg.aspect,
        nearVal=cam_cfg.near,
        farVal=cam_cfg.far,
    )
    overhead_view = p.computeViewMatrix(
        cameraEyePosition=[0.5, 0.0, 1.5],
        cameraTargetPosition=[0.5, 0.0, scene_cfg.table_height],
        cameraUpVector=[0.0, 1.0, 0.0],
    )
    overhead_cam = Camera(overhead_view, proj_matrix, cam_cfg)
    wrist_cam = WristCamera(scene.robot_id, robot_cfg.ee_link, proj_matrix, cam_cfg)

    detector = ObjectDetector(cam_cfg, palette, workspace_cfg)
    controller = RobotController(
        scene.robot_id, robot_cfg, scene_cfg, workspace_cfg, use_gui
    )

    return PickAndPlacePipeline(
        scene, overhead_cam, wrist_cam, detector, controller,
        scene_cfg, cam_cfg, use_gui,
    )


def main(use_gui: bool = True, num_cubes: int = 5, sim_steps: int = 300) -> None:
    """Run the full perception and pick-and-place demo.

    Args:
        use_gui: Open the PyBullet GUI and OpenCV display windows when True.
        num_cubes: Number of random cubes to spawn on the table.
        sim_steps: Physics warm-up steps before perception starts (300 ≈ 1.25 s).
    """
    pipeline = build_pipeline(use_gui)
    try:
        pipeline.run(num_cubes=num_cubes, sim_steps=sim_steps)
    finally:
        pipeline._scene.disconnect()


if __name__ == "__main__":
    main(use_gui=True, num_cubes=5)
