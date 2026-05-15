"""
complexity_analyzer.py
─────────────────────
Computes a scene complexity score C ∈ [0, 1] from Stage-1 detections.

Score formula:
    C = 0.35 × iou_signal + 0.30 × proximity_signal
      + 0.20 × coverage_signal + 0.15 × count_signal

Routing thresholds (calibrated to dataset; max observed = 0.50):
    C < 0.15  → Simple    → Stage 1 only
    C < 0.40  → Ambiguous → Stage 1 + 2
    C ≥ 0.40  → Complex   → Stage 1 + 2 + 3
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# ── Weights ──────────────────────────────────────────────────────────────────
W_IOU       = 0.35
W_PROXIMITY = 0.30
W_COVERAGE  = 0.20
W_COUNT     = 0.15

# ── Routing thresholds ────────────────────────────────────────────────────────
ROUTE_SIMPLE  = 0.15
ROUTE_COMPLEX = 0.40

# ── Proximity calibration ─────────────────────────────────────────────────────
PROXIMITY_PX = 100          # pixel distance below which boxes are "close"
MAX_EXPECTED_BOXES = 20     # used to normalise count signal


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])

    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _centre(box: np.ndarray) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _centre_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    cx_a, cy_a = _centre(box_a)
    cx_b, cy_b = _centre(box_b)
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


class ComplexityAnalyzer:
    """
    Compute scene complexity score from a list of Stage-1 detections.

    Parameters
    ----------
    proximity_px : int
        Pixel-distance threshold for the proximity signal.
    max_expected_boxes : int
        Upper bound used to normalise the count signal.
    """

    def __init__(
        self,
        proximity_px: int = PROXIMITY_PX,
        max_expected_boxes: int = MAX_EXPECTED_BOXES,
    ) -> None:
        self.proximity_px = proximity_px
        self.max_expected_boxes = max_expected_boxes

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        boxes: List[np.ndarray],
        image_wh: Tuple[int, int] | None = None,
    ) -> float:
        """
        Compute complexity score C ∈ [0, 1].

        Parameters
        ----------
        boxes : list of np.ndarray
            Each box is [x1, y1, x2, y2] in pixel coords.
        image_wh : (width, height) or None
            Required for coverage signal; if None, coverage = 0.

        Returns
        -------
        float
            Complexity score C.
        """
        if len(boxes) == 0:
            return 0.0

        iou_sig       = self._iou_signal(boxes)
        prox_sig      = self._proximity_signal(boxes)
        coverage_sig  = self._coverage_signal(boxes, image_wh)
        count_sig     = self._count_signal(boxes)

        C = (
            W_IOU       * iou_sig
            + W_PROXIMITY * prox_sig
            + W_COVERAGE  * coverage_sig
            + W_COUNT     * count_sig
        )
        return float(np.clip(C, 0.0, 1.0))

    def route(self, score: float) -> str:
        """Map score to route label: 'simple' | 'ambiguous' | 'complex'."""
        if score < ROUTE_SIMPLE:
            return "simple"
        if score < ROUTE_COMPLEX:
            return "ambiguous"
        return "complex"

    # ── Signal helpers ────────────────────────────────────────────────────────

    def _iou_signal(self, boxes: List[np.ndarray]) -> float:
        """Mean pairwise IoU (0 if fewer than 2 boxes)."""
        n = len(boxes)
        if n < 2:
            return 0.0
        ious = [
            _box_iou(boxes[i], boxes[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return float(np.mean(ious))

    def _proximity_signal(self, boxes: List[np.ndarray]) -> float:
        """Fraction of box pairs whose centres are within proximity_px."""
        n = len(boxes)
        if n < 2:
            return 0.0
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        close = sum(
            1 for i, j in pairs
            if _centre_distance(boxes[i], boxes[j]) < self.proximity_px
        )
        return close / len(pairs)

    def _coverage_signal(
        self,
        boxes: List[np.ndarray],
        image_wh: Tuple[int, int] | None,
    ) -> float:
        """Total box area as fraction of image area (clipped to 1)."""
        if image_wh is None:
            return 0.0
        img_area = image_wh[0] * image_wh[1]
        if img_area == 0:
            return 0.0
        total = sum(
            (b[2] - b[0]) * (b[3] - b[1]) for b in boxes
        )
        return float(np.clip(total / img_area, 0.0, 1.0))

    def _count_signal(self, boxes: List[np.ndarray]) -> float:
        """Normalised detection count (clipped to 1)."""
        return float(np.clip(len(boxes) / self.max_expected_boxes, 0.0, 1.0))
