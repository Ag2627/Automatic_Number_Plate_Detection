"""Compare the archived classical detector with Laplacian, Canny, and Sobel.

Example:
  python archive/classical_plate_detection/experiments/plate_detection_demo.py --image data/images/image1.jpg
"""
import argparse
from pathlib import Path
import sys

import cv2

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARCHIVE_ROOT.parent.parent
sys.path.insert(0, str(ARCHIVE_ROOT))

from plate_detection import compare_edge_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="image file to inspect")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "detection_results" / "classical"),
        help="where to save the three boxed comparison images",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = compare_edge_methods(image)
    for method in ("laplacian", "canny", "sobel"):
        candidates = comparisons[method]["candidates"]
        boxed = image.copy()
        if candidates:
            x, y, width, height = candidates[0]["bbox"]
            score = candidates[0]["score"]
            cv2.rectangle(boxed, (x, y), (x + width, y + height), (0, 200, 0), 3)
            print(
                f"{method}: x={x}, y={y}, width={width}, height={height}, "
                f"score={score:.3f} (ranking score, not confidence)"
            )
        else:
            print(f"{method}: no plate candidate found")

        output_path = output_dir / f"{image_path.stem}_{method}_boxed.jpg"
        if not cv2.imwrite(str(output_path), boxed):
            raise OSError(f"Could not save boxed image: {output_path}")
        print(f"  boxed image: {output_path.resolve()}")


if __name__ == "__main__":
    main()
