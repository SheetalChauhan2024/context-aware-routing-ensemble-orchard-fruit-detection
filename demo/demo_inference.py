"""
demo_inference.py
─────────────────
Quick demo script for the CAR Ensemble.

Usage
-----
    python demo/demo_inference.py --image demo/sample_images/example.jpg

    # Specify weight paths explicitly:
    python demo/demo_inference.py \
        --image  path/to/image.jpg \
        --s1     weights/stage1_yolo11n/best.pt \
        --s2     weights/stage2_yolo11s/best.pt \
        --s3     weights/stage3_yolo11m/best.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.car_ensemble import CAREnsemble


CLASS_NAMES = ['raw_apple', 'raw_plum']

DEFAULT_WEIGHTS = {
    's1': 'weights/stage1_yolo11n/best.pt',
    's2': 'weights/stage2_yolo11s/best.pt',
    's3': 'weights/stage3_yolo11m/best.pt',
}


def draw_results(image_path: str, result: dict) -> None:
    """Overlay detections on the image and display / save it."""
    try:
        import cv2
    except ImportError:
        print("[DEMO] opencv-python not installed — skipping visualisation.")
        return

    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print(f"[DEMO] Could not read image: {image_path}")
        return

    colours = {0: (0, 200, 0), 1: (200, 0, 200)}  # green / purple

    for box, score, cls in zip(result['boxes'], result['scores'], result['classes']):
        x1, y1, x2, y2 = map(int, box)
        colour = colours.get(int(cls), (0, 255, 255))
        label  = f"{CLASS_NAMES[int(cls)]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Overlay route info
    info = (
        f"Route: {result['route']}  "
        f"C={result['complexity']:.3f}  "
        f"Dets={result['num_detections']}  "
        f"{result['inference_ms']:.1f} ms"
    )
    cv2.putText(img, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = Path(image_path).stem + '_car_result.jpg'
    cv2.imwrite(out_path, img)
    print(f"[DEMO] Saved annotated image → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='CAR Ensemble demo inference')
    parser.add_argument('--image',  required=True,            help='Input image path')
    parser.add_argument('--s1',     default=DEFAULT_WEIGHTS['s1'])
    parser.add_argument('--s2',     default=DEFAULT_WEIGHTS['s2'])
    parser.add_argument('--s3',     default=DEFAULT_WEIGHTS['s3'])
    parser.add_argument('--device', default='cpu',            help='cuda or cpu')
    parser.add_argument('--no_draw', action='store_true',     help='Skip drawing output')
    args = parser.parse_args()

    ensemble = CAREnsemble(
        weights={'s1': args.s1, 's2': args.s2, 's3': args.s3},
        device=args.device,
        verbose=True,
    )

    result = ensemble.predict(args.image)

    print("\n─── CAR Ensemble Result ───────────────────────────────")
    print(f"  Route       : {result['route']}")
    print(f"  Complexity  : {result['complexity']}")
    print(f"  Detections  : {result['num_detections']}")
    print(f"  Inference   : {result['inference_ms']} ms")
    print()
    for i, (lbl, score, box) in enumerate(
        zip(result['labels'], result['scores'], result['boxes'])
    ):
        print(f"  [{i+1}] {lbl:<12} conf={score:.3f}  box={box.astype(int).tolist()}")
    print("───────────────────────────────────────────────────────\n")

    if not args.no_draw:
        draw_results(args.image, result)


if __name__ == '__main__':
    main()
