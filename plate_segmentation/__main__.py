
import argparse
from pathlib import Path

import cv2

from plate_detection import load_plate_model, detect_plate_boxes
from .ocr_engine import PlateOCR
from .pipeline import process_plate_crop


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + PaddleOCR Number Plate Recognition"
    )

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output-dir",
        default="detection_results/ocr"
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)

    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(args.image)

    print("Loading YOLO plate detector...")
    model = load_plate_model(args.weights)

    print("Loading PaddleOCR...")
    ocr_engine = PlateOCR()

    detections = detect_plate_boxes(
        image,
        model=model,
        confidence=args.confidence
    )

    if not detections:
        print("No number plate detected.")
        return

    print(f"Detected {len(detections)} plate(s).")

    for index, detection in enumerate(detections, start=1):

        result = process_plate_crop(
            plate_crop=detection["crop"],
            source_image=args.image,
            source_bbox=detection["bbox"],
            output_dir=args.output_dir,
            detection_confidence=detection["confidence"],
            detection_index=index,
            ocr_engine=ocr_engine,
        )

        print(f"\nPlate {index}")
        print(
            "Registration number:",
            result["recognized_text"] or "Not recognized"
        )
        print(
            f"OCR confidence: {result['ocr_confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
