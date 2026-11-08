import argparse
import cv2

from plate_detection import (
    detect_plate_boxes,
    load_plate_model
)

from .pipeline import process_plate_crop


def main():

    parser = argparse.ArgumentParser(
        description="Person 2 + Person 3 ANPR pipeline"
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="detection_results/segmentation"
    )

    parser.add_argument(
        "--weights",
        default=None
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    # Read original image.
    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {args.image}"
        )

    # Load Person 2's YOLO model.
    model = load_plate_model(args.weights)

    # Detect number plates.
    detections = detect_plate_boxes(
        image,
        model=model,
        confidence=args.confidence
    )

    if not detections:

        print("No number plate detected.")
        return

    print(
        f"Detected {len(detections)} plate(s)"
    )

    # Process every detected plate.
    for index, detection in enumerate(
        detections,
        start=1
    ):

        result = process_plate_crop(

            plate_crop=detection["crop"],

            source_image=args.image,

            source_bbox=detection["bbox"],

            output_dir=args.output_dir,

            detection_confidence=detection["confidence"],

            detection_index=index
        )

        print("\nPlate:", index)

        print(
            "Segmentation successful:",
            result["segmentation_success"]
        )

        print(
            "Character count:",
            result["character_count"]
        )

        print(
            "Metadata:",
            result["metadata_path"]
        )


if __name__ == "__main__":
    main()