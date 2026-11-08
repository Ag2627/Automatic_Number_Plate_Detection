"""Run YOLO plate detection and record the returned boxes in the CSV."""
import argparse
import csv
import os
from pathlib import Path

import cv2

from .yolo_detector import detect_plate_boxes, load_plate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_CSV = PROJECT_ROOT / "data" / "annotations" / "plate_boxes.csv"
CSV_COLUMNS = ("image", "x", "y", "width", "height")


def image_csv_path(image_path):
    """Return an image path relative to the annotations CSV."""
    absolute_path = Path(image_path).expanduser().resolve()
    return Path(os.path.relpath(absolute_path, ANNOTATIONS_CSV.parent.resolve())).as_posix()


def update_annotations(image_reference, boxes):
    """Replace this image's old rows and write its current YOLO boxes."""
    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if ANNOTATIONS_CSV.exists():
        with ANNOTATIONS_CSV.open("r", newline="", encoding="utf-8-sig") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))

    # Re-running on the same image refreshes its rows instead of duplicating them.
    kept_rows = [row for row in existing_rows if row.get("image") != image_reference]
    for x, y, width, height in boxes:
        kept_rows.append({
            "image": image_reference,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })

    with ANNOTATIONS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Detect number plates with YOLO, draw boxes, and update the annotations CSV."
    )
    parser.add_argument("--image", required=True, help="image file to inspect")
    parser.add_argument("--weights", help="local YOLO .pt checkpoint; defaults to cached public weights")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="minimum confidence score from 0 to 1")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--output", help="boxed image path; defaults to detection_results/yolo/<name>_boxed.jpg")
    args = parser.parse_args()

    input_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    model = load_plate_model(args.weights)
    detections = detect_plate_boxes(
        image, confidence=args.confidence, image_size=args.imgsz, model=model
    )

    boxed_image = image.copy()
    for detection in detections:
        x, y, width, height = detection["bbox"]
        cv2.rectangle(boxed_image, (x, y), (x + width, y + height), (0, 200, 0), 3)

    output_path = Path(args.output).expanduser() if args.output else (
        PROJECT_ROOT / "detection_results" / "yolo" / f"{input_path.stem}_boxed.jpg"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), boxed_image):
        raise OSError(f"Could not save boxed image: {output_path}")

    image_reference = image_csv_path(input_path)
    update_annotations(image_reference, [detection["bbox"] for detection in detections])

    print(f"Image: {input_path}")
    if detections:
        for index, detection in enumerate(detections, start=1):
            x, y, width, height = detection["bbox"]
            print(
                f"Plate {index}: x={x}, y={y}, width={width}, height={height}, "
                f"confidence={detection['confidence']:.3f}"
            )
    else:
        print("No plates detected.")
    print(f"Annotations updated: {ANNOTATIONS_CSV}")
    print(f"Boxed image saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()

