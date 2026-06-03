"""
utils.py
────────
Utility functions for the CAR Ensemble:
  • Weighted Vote Merger — combines detections from multiple stages
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Weighted Vote Merger ──────────────────────────────────────────────────────

# Stage weights
W_S1       = 1.0
W_S2       = 1.5
W_S3       = 2.0
W_S3_BONUS = 2.5   # bonus weight when S3 is the sole detector of a box

IOU_MATCH  = 0.30   # IoU threshold to consider two detections the same box


def weighted_vote_merge(
    detections_per_stage: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_match: float = IOU_MATCH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge detections from multiple stages using weighted voting.

    Parameters
    ----------
    detections_per_stage : dict mapping stage key ('s1' | 's2' | 's3') to
                           (boxes, scores, classes) tuples.
                           boxes  shape (N, 4)  — [x1, y1, x2, y2]
                           scores shape (N,)
                           classes shape (N,)
    iou_match : IoU threshold to group detections as the same physical box.

    Returns
    -------
    (merged_boxes, merged_scores, merged_classes)
    """
    stage_weights = {'s1': W_S1, 's2': W_S2, 's3': W_S3}

    # Flatten all detections, tagging each with its stage
    all_boxes, all_scores, all_classes, all_weights = [], [], [], []
    for stage, (boxes, scores, classes) in detections_per_stage.items():
        w = stage_weights.get(stage, 1.0)
        for b, s, c in zip(boxes, scores, classes):
            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)
            all_weights.append(w)

    if not all_boxes:
        empty = np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        return empty

    all_boxes   = np.array(all_boxes)
    all_scores  = np.array(all_scores)
    all_classes = np.array(all_classes, dtype=int)
    all_weights = np.array(all_weights)

    merged_boxes, merged_scores, merged_classes = [], [], []
    used = np.zeros(len(all_boxes), dtype=bool)

    for i in range(len(all_boxes)):
        if used[i]:
            continue

        # Find all detections that overlap with box i
        group = [i]
        for j in range(i + 1, len(all_boxes)):
            if not used[j] and all_classes[j] == all_classes[i]:
                if _iou(all_boxes[i], all_boxes[j]) >= iou_match:
                    group.append(j)

        # Check if S3 is the sole contributor → apply bonus weight
        group_stages = {
            stage
            for stage, (boxes, _, _) in detections_per_stage.items()
            for g in group
            if g < len(boxes)
        }
        weights = np.array([
            W_S3_BONUS if (all_weights[g] == W_S3 and len(group) == 1)
            else all_weights[g]
            for g in group
        ])

        # Weighted average of boxes and scores
        w_total = weights.sum()
        box_avg = (all_boxes[group] * weights[:, None]).sum(axis=0) / w_total
        score_avg = (all_scores[group] * weights).sum() / w_total

        merged_boxes.append(box_avg)
        merged_scores.append(score_avg)
        merged_classes.append(all_classes[i])

        for g in group:
            used[g] = True

    if not merged_boxes:
        empty = np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        return empty

    merged_boxes   = np.array(merged_boxes)
    merged_scores  = np.array(merged_scores)
    merged_classes = np.array(merged_classes, dtype=int)

    # Final sort by score descending
    order = np.argsort(-merged_scores)
    return merged_boxes[order], merged_scores[order], merged_classes[order]
