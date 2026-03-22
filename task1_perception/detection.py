"""
task1_perception/detection.py
==============================
ObjectDetector and Detection dataclass (SRP: all vision/perception logic here).

Design
------
Detection  — a plain dataclass carrying the result of one detected object.
ObjectDetector — stateless detector; accepts injected config (DIP: depends on
                 ColourPalette/WorkspaceConfig abstractions, not hard-coded values).

The detector is intentionally separated from the camera so that the detection
strategy can be swapped (OCP) — e.g. replacing HSV segmentation with a neural
detector — without touching Camera or RobotController.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig, ColourPalette, WorkspaceConfig
from .camera import Camera


@dataclass
class Detection:
    """Result of a single object detection."""

    pixel_x: int
    pixel_y: int
    world_xyz: np.ndarray  # shape (3,), metres
    colour: str


class ObjectDetector:
    """
    Detects objects in an RGB image using HSV colour segmentation.

    Responsibilities (SRP):
      - Segment the image to find candidate pixel centroids.
      - Classify the colour of each candidate.
      - Back-project pixels to 3-D world coordinates.
      - Filter detections to the reachable workspace.
    """

    def __init__(
        self,
        cam_cfg: CameraConfig,
        palette: ColourPalette,
        workspace: WorkspaceConfig,
    ) -> None:
        """Initialise the detector with injected configuration (DIP).

        Args:
            cam_cfg: Camera resolution parameters (used for patch bounds checking).
            palette: HSV hue ranges and saturation/value thresholds for colour classification.
            workspace: Reachable XYZ bounds used to filter back-projected detections.
        """
        self._cam_cfg = cam_cfg
        self._palette = palette
        self._workspace = workspace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        camera: Camera,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        table_height: float,
    ) -> list[Detection]:
        """
        Run the full detection pipeline on one overhead frame.

        Parameters
        ----------
        camera      : Camera whose matrices are used for back-projection.
        rgb_img     : uint8 H×W×3 overhead RGB image.
        depth_img   : float32 H×W depth buffer.
        table_height: Z coordinate of the table surface (used for workspace filter).

        Returns
        -------
        List of Detection objects within the reachable workspace, ordered by
        their pixel position (top-left to bottom-right).
        """
        pixel_centroids = self._segment(rgb_img)

        results: list[Detection] = []
        ws = self._workspace

        for px, py in pixel_centroids:
            xyz = camera.pixel_to_world(px, py, depth_img)
            colour = self._classify_colour(rgb_img, px, py)

            z_min = table_height + ws.z_min_above_table
            z_max = table_height + ws.z_max_above_table

            if (
                ws.reach_x[0] <= xyz[0] <= ws.reach_x[1]
                and ws.reach_y[0] <= xyz[1] <= ws.reach_y[1]
                and z_min <= xyz[2] <= z_max
            ):
                results.append(Detection(px, py, xyz, colour))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment(self, rgb_img: np.ndarray) -> list[tuple[int, int]]:
        """
        HSV colour segmentation → list of (pixel_x, pixel_y) centroids.

        Filters the neutral grey/white background and returns only vivid objects.
        Morphological open+close removes isolated noise pixels and fills small gaps.
        """
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords: list[tuple[int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    coords.append(
                        (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    )
        return coords

    def _classify_colour(self, rgb_img: np.ndarray, px: int, py: int) -> str:
        """
        Classify the dominant colour of the object centred at (px, py).

        Samples a small patch around the centroid and computes the mean HSV
        hue. Returns a colour name string (e.g. "red", "blue") or "unknown".
        """
        pal = self._palette
        h, w = rgb_img.shape[:2]
        half = pal.patch_size // 2
        x0, x1 = max(0, px - half), min(w, px + half)
        y0, y1 = max(0, py - half), min(h, py + half)
        patch = rgb_img[y0:y1, x0:x1]
        if patch.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        val = float(np.mean(hsv[:, :, 2]))

        if sat < pal.min_saturation or val < pal.min_value:
            return "white/grey"

        hue = float(np.mean(hsv[:, :, 0]))
        for name, lo, hi in pal.entries:
            if lo <= hue <= hi:
                return name
        return "unknown"
