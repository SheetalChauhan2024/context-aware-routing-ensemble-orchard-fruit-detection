"""
complexity_analyzer.py
──────────────────────
Computes a scene complexity score C ∈ [0, 1] from Stage-1 detections.

Score formula:
    C = 0.35 × S_IoU  + 0.30 × S_prox
      + 0.20 × S_cov  + 0.15 × S_count

Signal definitions:
    S_IoU  — mean pairwise IoU between detected boxes (occlusion signal)
    S_prox — proportion of box pairs whose centre-to-centre distance
             is less than twice the average fruit diameter (cluster signal)
    S_cov  — fraction of frame area covered by all bounding boxes (density signal)
    S_count — normalised detection count, saturating at 12 detections

Routing thresholds (calibrated to dataset):
    C < 0.15  → Simple    → Stage 1 only
    C < 0.40  → Ambiguous → Stage 1 + 2
    C ≥ 0.40  → Complex   → Stage 1 + 2 + 3
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


# ── Complexity signal weights ─────────────────────────────────────────────────
W_IOU       = 0.35
W_PROXIMITY = 0.30
W_COVERAGE  = 0.20
W_COUNT     = 0.15

# ── Routing thresholds ────────────────────────────────────────────────────────
ROUTE_SIMPLE  = 0.15
ROUTE_COMPLEX = 0.40

# ── Count signal calibration ──────────────────────────────────────────────────
MAX_DETECTIONS = 12   # count signal saturates at 12 detections


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


def _centre_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Euclidean distance between box centres."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


class ComplexityAnalyzer:
    """
    Compute scene complexity score from Stage-1 detections.

    Parameters
    ----------
    max_detections : int
        Detection count at which the count signal saturates (default: 12).
    """

    def __init__(self, max_detections: int = MAX_DETECTIONS) -> None:
        self.max_detections = max_detections

    def score(
        self,
        boxes: List[np.ndarray],
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Compute complexity score C ∈ [0, 1].

        Parameters
        ----------
        boxes : list of np.ndarray
            Each box is [x1, y1, x2, y2] in pixel coordinates.
        image_wh : (width, height) or None
            Required for coverage signal; if None, coverage signal = 0.

        Returns
        -------
        float
            Complexity score C.
        """
        if len(boxes) == 0:
            return 0.0

        s_iou      = self._iou_signal(boxes)
        s_prox     = self._proximity_signal(boxes)
        s_coverage = self._coverage_signal(boxes, image_wh)
        s_count    = self._count_signal(boxes)

        C = (
            W_IOU       * s_iou
            + W_PROXIMITY * s_prox
            + W_COVERAGE  * s_coverage
            + W_COUNT     * s_count
        )
        return float(np.clip(C, 0.0, 1.0))

    def route(self, score: float) -> str:
        """Map complexity score to route label."""
        if score < ROUTE_SIMPLE:
            return "simple"
        if score < ROUTE_COMPLEX:
            return "ambiguous"
        return "complex"

    # ── Signal computations ───────────────────────────────────────────────────

    def _iou_signal(self, boxes: List[np.ndarray]) -> float:
        """Mean pairwise IoU across all box pairs (0 if fewer than 2 boxes)."""
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
        """
        Fraction of box pairs whose centre-to-centre distance is less than
        twice the average fruit diameter, where diameter is estimated as the
        square root of each box area.
        """
        n = len(boxes)
        if n < 2:
            return 0.0

        diameters = [
            ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5
            for b in boxes
        ]
        avg_diameter = float(np.mean(diameters)) if diameters else 50.0
        threshold    = 2.0 * avg_diameter

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        close = sum(
            1 for i, j in pairs
            if _centre_distance(boxes[i], boxes[j]) < threshold
        )
        return close / len(pairs)

    def _coverage_signal(
        self,
        boxes: List[np.ndarray],
        image_wh: Optional[Tuple[int, int]],
    ) -> float:
        """Total bounding box area as a fraction of image area."""
        if image_wh is None:
            return 0.0
        img_area = image_wh[0] * image_wh[1]
        if img_area == 0:
            return 0.0
        total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        return float(np.clip(total_area / img_area, 0.0, 1.0))

    def _count_signal(self, boxes: List[np.ndarray]) -> float:
        """Normalised detection count, saturating at MAX_DETECTIONS (12)."""
        return float(np.clip(len(boxes) / self.max_detections, 0.0, 1.0))
