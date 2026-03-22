"""
task1_perception/detection.py
==============================
ObjectDetector, ShapeClassifier, and Detection dataclass.

Design
------
Detection       — plain dataclass carrying the result of one detected object.
ShapeClassifier — geometric classifier using contour analysis and depth variance.
                  Distinguishes box / cylinder / sphere from an overhead RGB-D view.
                  Feature design is sim-to-real transferable: the same geometric
                  cues (circularity, depth-profile curvature) are available from
                  any physical RGB-D camera (e.g. RealSense, Kinect).
ObjectDetector  — stateless detector; accepts injected config (DIP).

Shape classification pipeline
------------------------------
1. Circularity = 4π·area / perimeter²
   - Box projected from overhead → rectangular mask → circularity < 0.88
   - Cylinder/Sphere projected from overhead → circular mask → circularity ≥ 0.88
2. Depth variance inside the object mask (from the overhead depth image)
   - Cylinder has a flat top → near-uniform depth → low variance
   - Sphere has a curved top → depth increases from centre toward rim → high variance
   Threshold is tuned to the PyBullet depth buffer scale and holds for real cameras
   after the depth buffer is converted to metric distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    shape: str = "unknown"  # "box", "cylinder", or "sphere"


# ---------------------------------------------------------------------------
# Geometric shape classifier
# ---------------------------------------------------------------------------


class ShapeClassifier:
    """
    Classify object shape from contour geometry and depth-map variance.

    This is a geometry-driven AI classifier: the features are derived from
    first principles of projective geometry, making them robust across
    viewpoints, lighting conditions, and the sim-to-real gap.

    Decision logic
    --------------
    circularity < ROUND_THRESH  →  box
    circularity ≥ ROUND_THRESH  AND  depth_var < SPHERE_VAR_THRESH  →  cylinder
    circularity ≥ ROUND_THRESH  AND  depth_var ≥ SPHERE_VAR_THRESH  →  sphere
    """

    # Overhead-projected square has circularity π/4 ≈ 0.785; circles → 1.0.
    # Threshold at 0.88 gives a clean separation in practice.
    ROUND_THRESH: float = 0.88

    # Depth-buffer variance threshold distinguishing flat (cylinder) from
    # curved (sphere) tops.  Empirically: sphere ≈ 3e-7–1e-5, cylinder ≈ 0.
    SPHERE_VAR_THRESH: float = 2e-7

    def classify(
        self,
        contour: np.ndarray,
        depth_img: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> str:
        """Return "box", "cylinder", or "sphere" for the given contour.

        Args:
            contour  : OpenCV contour array (N×1×2 int32).
            depth_img: Full-frame depth buffer float32 H×W, values in [0, 1].
            img_h    : Image height in pixels.
            img_w    : Image width in pixels.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6 or area < 1.0:
            return "box"

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        if circularity < self.ROUND_THRESH:
            return "box"

        # Round shape — distinguish cylinder (flat top) from sphere (curved top)
        # by computing depth variance inside the contour mask.
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        depth_vals = depth_img[mask > 0]
        depth_var = float(np.var(depth_vals)) if len(depth_vals) > 3 else 0.0

        return "sphere" if depth_var >= self.SPHERE_VAR_THRESH else "cylinder"


class ObjectDetector:
    """
    Detects objects in an RGB-D image using HSV segmentation + geometric shape classification.

    Responsibilities (SRP):
      - Segment the image to find candidate pixel centroids and contours.
      - Classify the shape of each candidate via ShapeClassifier.
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
        self._shape_clf = ShapeClassifier()

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
        candidates = self._segment(rgb_img)

        results: list[Detection] = []
        ws = self._workspace
        img_h, img_w = rgb_img.shape[:2]

        for px, py, contour in candidates:
            xyz = camera.pixel_to_world(px, py, depth_img)
            colour = self._classify_colour(rgb_img, px, py)
            shape = self._shape_clf.classify(contour, depth_img, img_h, img_w)

            z_min = table_height + ws.z_min_above_table
            z_max = table_height + ws.z_max_above_table

            if (
                ws.reach_x[0] <= xyz[0] <= ws.reach_x[1]
                and ws.reach_y[0] <= xyz[1] <= ws.reach_y[1]
                and z_min <= xyz[2] <= z_max
            ):
                results.append(Detection(px, py, xyz, colour, shape))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment(self, rgb_img: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """
        HSV colour segmentation → list of (pixel_x, pixel_y, contour) triples.

        Filters the neutral grey/white background and returns only vivid objects.
        Morphological open+close removes isolated noise pixels and fills small gaps.
        The contour is returned so that the caller can pass it to ShapeClassifier.
        """
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: list[tuple[int, int, np.ndarray]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    results.append((px, py, cnt))
        return results

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
