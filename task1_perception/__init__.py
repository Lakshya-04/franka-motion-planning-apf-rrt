"""
task1_perception
================
Franka Panda Pick-and-Place perception pipeline — SOLID architecture.

Public API
----------
The simplest way to run the demo:

    from task1_perception.pipeline import main
    main(use_gui=True, num_cubes=5)

For programmatic use with custom configs:

    from task1_perception.pipeline import build_pipeline
    pipeline = build_pipeline(use_gui=False)
    pipeline.run(num_cubes=3, sim_steps=200)

Module layout (Single Responsibility):
  config.py      — frozen dataclasses for all configuration constants
  scene.py       — SceneManager: PyBullet world lifecycle
  camera.py      — Camera, WristCamera, pixel_to_world projection math
  detection.py   — ObjectDetector, Detection dataclass (HSV segmentation)
  controller.py  — RobotController: IK, motion, gripper, grasp sequence
  pipeline.py    — PickAndPlacePipeline: orchestration + entry point
"""

from .config import (
    CameraConfig,
    ColourPalette,
    RobotConfig,
    SceneConfig,
    WorkspaceConfig,
)
from .detection import Detection
from .pipeline import PickAndPlacePipeline, build_pipeline, main

__all__ = [
    "CameraConfig",
    "ColourPalette",
    "RobotConfig",
    "SceneConfig",
    "WorkspaceConfig",
    "Detection",
    "PickAndPlacePipeline",
    "build_pipeline",
    "main",
]
