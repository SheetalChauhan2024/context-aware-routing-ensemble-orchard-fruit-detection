"""
car_ensemble.py
───────────────
Context-Aware Routing (CAR) Ensemble for unripe fruit detection.

Three YOLOv11 stages are dynamically activated based on scene complexity:
    Stage 1 (YOLOv11n) → always runs
    Stage 2 (YOLOv11s) → runs if scene is ambiguous or complex
    Stage 3 (YOLOv11m) → runs if scene is complex (or upgraded)

Usage
-----
    from src.car_ensemble import CAREnsemble

    ensemble = CAREnsemble(weights={
        's1': 'weights/stage1_yolo11n/best.pt',
        's2': 'weights/stage2_yolo11s/best.pt',
        's3': 'weights/stage3_yolo11m/best.pt',
    })

    result = ensemble.predict('path/to/image.jpg')
    print(result['route'], result['boxes'])
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .complexity_analyzer import ComplexityAnalyzer
from .utils import weighted_vote_merge, _iou


# ── Default confidence thresholds per route ───────────────────────────────────
CONF_THRESHOLDS: Dict[str, Dict[str, float]] = {
    'simple':    {'s1': 0.70, 's2': 0.65, 's3': 0.60},
    'ambiguous': {'s1': 0.65, 's2': 0.60, 's3': 0.55},
    'complex':   {'s1': 0.55, 's2': 0.55, 's3': 0.50},
}

UPGRADE_RATIO = 1.5   # S2/S1 detection ratio that triggers upgrade to complex
PROXIMITY_PX  = 100   # pixel distance for Route B isolation check


def _has_nearby(box: np.ndarray, others: np.ndarray, proximity: float = PROXIMITY_PX) -> bool:
    """Return True if any other box centre is within proximity pixels of this box centre."""
    if len(others) == 0:
        return False
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    for o in others:
        if np.array_equal(o, box):
            continue
        ocx = (o[0] + o[2]) / 2.0
        ocy = (o[1] + o[3]) / 2.0
        if ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5 < proximity:
            return True
    return False

CLASS_NAMES = ['raw_apple', 'raw_plum']


class CAREnsemble:
    """
    Context-Aware Routing Ensemble.

    Parameters
    ----------
    weights : dict
        Mapping of stage key ('s1', 's2', 's3') to .pt weight file paths.
    conf_thresholds : dict, optional
        Override default confidence thresholds.
    device : str
        PyTorch device string ('cpu', 'cuda', 'cuda:0', …).
    verbose : bool
        Print routing decisions during inference.
    """

    def __init__(
        self,
        weights: Dict[str, str],
        conf_thresholds: Optional[Dict] = None,
        device: str = 'cpu',
        verbose: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.device  = device
        self.verbose = verbose
        self.conf    = conf_thresholds or CONF_THRESHOLDS

        self.analyzer = ComplexityAnalyzer()

        # Load models
        self._models: Dict[str, YOLO] = {}
        for stage, path in weights.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Weight file not found for {stage}: {path}\n"
                    f"Download from the links in weights/README.md"
                )
            self._models[stage] = YOLO(str(p))
            if self.verbose:
                print(f"[CAR] Loaded {stage}: {path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image,
        return_image_wh: bool = True,
    ) -> Dict:
        """
        Run ensemble inference on a single image.

        Parameters
        ----------
        image : str | Path | np.ndarray
            Image path or pre-loaded BGR/RGB array.

        Returns
        -------
        dict with keys:
            boxes    : np.ndarray (N, 4) — [x1, y1, x2, y2]
            scores   : np.ndarray (N,)
            classes  : np.ndarray (N,) — int class IDs
            labels   : list[str]       — class names
            route    : str             — 'simple' | 'ambiguous' | 'complex'
            complexity : float         — C score
            inference_ms : float
        """
        t0 = time.perf_counter()

        # ── Stage 1 ──────────────────────────────────────────────────────────
        s1_boxes, s1_scores, s1_classes, image_wh = self._run_stage(
            's1', image, conf=0.25   # low conf for complexity estimation
        )

        # Compute complexity
        C = self.analyzer.score(
            [b for b in s1_boxes],
            image_wh=image_wh,
        )
        route = self.analyzer.route(C)

        if self.verbose:
            print(f"[CAR] C={C:.3f} → {route} | S1 dets={len(s1_boxes)}")

        # ── Early exit: no detections ─────────────────────────────────────────
        if len(s1_boxes) == 0:
            return self._build_result(
                np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int),
                route='simple', C=C, t0=t0
            )

        # Apply route-specific confidence threshold to S1
        s1_mask = s1_scores >= self.conf[route]['s1']
        s1_boxes, s1_scores, s1_classes = (
            s1_boxes[s1_mask], s1_scores[s1_mask], s1_classes[s1_mask]
        )

        detections = {'s1': (s1_boxes, s1_scores, s1_classes)}

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if route in ('ambiguous', 'complex'):
            s2_boxes, s2_scores, s2_classes, _ = self._run_stage(
                's2', image, conf=self.conf[route]['s2']
            )

            # Dynamic upgrade
            if (
                route == 'ambiguous'
                and len(s1_boxes) > 0
                and len(s2_boxes) >= UPGRADE_RATIO * len(s1_boxes)
            ):
                route = 'complex'
                if self.verbose:
                    print(
                        f"[CAR] Dynamic upgrade → complex "
                        f"(S2={len(s2_boxes)} / S1={len(s1_boxes)})"
                    )

            if route == 'ambiguous':
                # ── Route B isolation logic ───────────────────────────────────
                # High-confidence Stage 1 detections with no nearby neighbours
                # are accepted directly without Stage 2 verification.
                # Remaining detections are verified against Stage 2.
                high_conf_boxes,  high_conf_scores,  high_conf_classes  = [], [], []
                ambiguous_boxes,  ambiguous_scores,  ambiguous_classes  = [], [], []

                for i in range(len(s1_boxes)):
                    if (s1_scores[i] >= self.conf['ambiguous']['s1']
                            and not _has_nearby(s1_boxes[i], s1_boxes)):
                        high_conf_boxes.append(s1_boxes[i])
                        high_conf_scores.append(s1_scores[i])
                        high_conf_classes.append(s1_classes[i])
                    else:
                        ambiguous_boxes.append(s1_boxes[i])
                        ambiguous_scores.append(s1_scores[i])
                        ambiguous_classes.append(s1_classes[i])

                # Verify ambiguous Stage 1 detections against Stage 2
                verified_boxes, verified_scores, verified_classes = [], [], []
                for i in range(len(ambiguous_boxes)):
                    for j in range(len(s2_boxes)):
                        if (s2_classes[j] == ambiguous_classes[i]
                                and _iou(ambiguous_boxes[i], s2_boxes[j]) >= 0.30):
                            # Weighted consensus box
                            wb = (ambiguous_boxes[i] + s2_boxes[j]) / 2.0
                            ws = (ambiguous_scores[i] + s2_scores[j]) / 2.0
                            verified_boxes.append(wb)
                            verified_scores.append(ws)
                            verified_classes.append(ambiguous_classes[i])
                            break

                # Add Stage 2 detections not matched by any Stage 1 box
                for j in range(len(s2_boxes)):
                    if not any(
                        _iou(s2_boxes[j], s1_boxes[i]) >= 0.30
                        for i in range(len(s1_boxes))
                    ):
                        verified_boxes.append(s2_boxes[j])
                        verified_scores.append(s2_scores[j])
                        verified_classes.append(s2_classes[j])

                # Combine high-confidence isolated + verified detections
                all_boxes   = high_conf_boxes  + verified_boxes
                all_scores  = high_conf_scores + verified_scores
                all_classes = high_conf_classes + verified_classes

                if all_boxes:
                    detections = {'s1': (
                        np.array(all_boxes),
                        np.array(all_scores),
                        np.array(all_classes, dtype=int),
                    )}
                else:
                    detections = {'s1': (
                        np.empty((0, 4)),
                        np.empty(0),
                        np.empty(0, dtype=int),
                    )}
            else:
                # Route upgraded to complex — pass all Stage 2 detections
                detections['s2'] = (s2_boxes, s2_scores, s2_classes)

        # ── Stage 3 ──────────────────────────────────────────────────────────
        if route == 'complex':
            s3_boxes, s3_scores, s3_classes, _ = self._run_stage(
                's3', image, conf=self.conf[route]['s3']
            )
            detections['s3'] = (s3_boxes, s3_scores, s3_classes)

        # ── Merge ────────────────────────────────────────────────────────────
        merged_boxes, merged_scores, merged_classes = weighted_vote_merge(detections)

        return self._build_result(merged_boxes, merged_scores, merged_classes, route, C, t0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage: str,
        image,
        conf: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
        """Run a single YOLO stage and return (boxes, scores, classes, image_wh)."""
        model = self._models[stage]
        results = model.predict(
            image,
            conf=conf,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        image_wh = (r.orig_shape[1], r.orig_shape[0])   # (width, height)

        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0, dtype=int),
                image_wh,
            )

        boxes   = r.boxes.xyxy.cpu().numpy()
        scores  = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        return boxes, scores, classes, image_wh

    @staticmethod
    def _build_result(
        boxes, scores, classes, route, C, t0
    ) -> Dict:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes]
        return {
            'boxes':          boxes,
            'scores':         scores,
            'classes':        classes,
            'labels':         labels,
            'route':          route,
            'complexity':     round(C, 4),
            'inference_ms':   round(elapsed_ms, 2),
            'num_detections': len(boxes),
        }
