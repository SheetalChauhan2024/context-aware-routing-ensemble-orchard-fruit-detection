"""
evaluation.py
─────────────
Evaluation pipeline for the CAR Ensemble on a test split.

Computes per-image and aggregate metrics:
    • Precision, Recall, F1 at IoU@0.50
    • Route distribution
    • Per-class breakdown

Usage
-----
    python -m src.evaluation \
        --weights_s1 weights/stage1_yolo11n/best.pt \
        --weights_s2 weights/stage2_yolo11s/best.pt \
        --weights_s3 weights/stage3_yolo11m/best.pt \
        --test_dir   data/test/images \
        --label_dir  data/test/labels \
        --output     results/evaluation_output.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .car_ensemble import CAREnsemble


# ── IoU helpers ───────────────────────────────────────────────────────────────

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


def _yolo_label_to_xyxy(
    label_path: Path, img_w: int, img_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a YOLO-format .txt label file into (boxes_xyxy, classes)."""
    boxes, classes = [], []
    if not label_path.exists():
        return np.empty((0, 4)), np.empty(0, dtype=int)

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = (
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)

    if not boxes:
        return np.empty((0, 4)), np.empty(0, dtype=int)
    return np.array(boxes), np.array(classes, dtype=int)


# ── Per-image matching ────────────────────────────────────────────────────────

def match_detections(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int]:
    """
    Greedy matching of predictions to ground-truth at a given IoU threshold.

    Returns
    -------
    (TP, FP, FN)
    """
    matched_gt = set()
    tp = 0

    for i, (pb, pc) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou, best_j = 0.0, -1
        for j, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if j in matched_gt or gc != pc:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    ensemble: CAREnsemble,
    test_images: List[Path],
    label_dir: Path,
    iou_thresh: float = 0.50,
) -> Dict:
    total_tp = total_fp = total_fn = 0
    route_counts: Dict[str, int] = {}
    class_tp: Dict[int, int] = {}
    class_fp: Dict[int, int] = {}
    class_fn: Dict[int, int] = {}
    inference_times: List[float] = []

    for img_path in test_images:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue
        img_h, img_w = img.shape[:2]

        label_path = label_dir / (img_path.stem + '.txt')
        gt_boxes, gt_classes = _yolo_label_to_xyxy(label_path, img_w, img_h)

        result = ensemble.predict(img)
        inference_times.append(result['inference_ms'])

        route = result['route']
        route_counts[route] = route_counts.get(route, 0) + 1

        pred_boxes   = result['boxes']
        pred_classes = result['classes']

        tp, fp, fn = match_detections(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-class
        for cls in set(gt_classes.tolist() + pred_classes.tolist()):
            mask_pred = pred_classes == cls
            mask_gt   = gt_classes == cls
            ctp, cfp, cfn = match_detections(
                pred_boxes[mask_pred], pred_classes[mask_pred],
                gt_boxes[mask_gt],   gt_classes[mask_gt],
                iou_thresh
            )
            class_tp[cls] = class_tp.get(cls, 0) + ctp
            class_fp[cls] = class_fp.get(cls, 0) + cfp
            class_fn[cls] = class_fn.get(cls, 0) + cfn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_class = {}
    for cls in sorted(set(list(class_tp.keys()) + list(class_fp.keys()))):
        tp_ = class_tp.get(cls, 0)
        fp_ = class_fp.get(cls, 0)
        fn_ = class_fn.get(cls, 0)
        p   = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r   = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f   = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[cls] = {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4)}

    return {
        'num_images': len(test_images),
        'iou_threshold': iou_thresh,
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'avg_inference_ms': round(float(np.mean(inference_times)), 2),
        'route_distribution': route_counts,
        'per_class': per_class,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate CAR Ensemble on test set')
    parser.add_argument('--weights_s1', required=True)
    parser.add_argument('--weights_s2', required=True)
    parser.add_argument('--weights_s3', required=True)
    parser.add_argument('--test_dir',  required=True, help='Directory of test images')
    parser.add_argument('--label_dir', required=True, help='Directory of YOLO .txt labels')
    parser.add_argument('--output', default='results/evaluation_output.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    ensemble = CAREnsemble(
        weights={'s1': args.weights_s1, 's2': args.weights_s2, 's3': args.weights_s3},
        device=args.device,
        verbose=True,
    )

    test_images = sorted(Path(args.test_dir).glob('*.jpg')) + \
                  sorted(Path(args.test_dir).glob('*.png'))

    print(f"[EVAL] Found {len(test_images)} test images.")
    metrics = evaluate(ensemble, test_images, Path(args.label_dir))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\n[EVAL] Saved to {out_path}")


if __name__ == '__main__':
    main()
