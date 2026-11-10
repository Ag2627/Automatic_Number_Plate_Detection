"""Run plate detection, character segmentation, and OCR from a vehicle image."""

import argparse
import json

from .recognition import recognize_plate


def main():
    parser = argparse.ArgumentParser(description="Recognize a number plate with EasyOCR")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="vehicle image; runs detection and segmentation first")
    source.add_argument("--metadata", help="existing plate_segmentation JSON metadata")
    parser.add_argument("--output-dir", default="detection_results/segmentation")
    parser.add_argument("--weights", help="local plate-detector weights")
    parser.add_argument("--character-weights", default="models/character_detector.pt")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="minimum plate detection confidence")
    parser.add_argument("--character-confidence", type=float, default=0.25,
                        help="minimum character detection confidence")
    parser.add_argument("--ocr-confidence", type=float, default=0.5,
                        help="OCR score below which a plate needs review")
    args = parser.parse_args()

    if args.metadata:
        result = recognize_plate(args.metadata, confidence_threshold=args.ocr_confidence)
        print(json.dumps(result, indent=2))
        return

    import cv2
    from plate_detection import detect_plate_boxes, load_plate_model
    from plate_segmentation import load_character_model, process_plate_crop

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    plate_model = load_plate_model(args.weights)
    character_model = load_character_model(args.character_weights)
    detections = detect_plate_boxes(image, model=plate_model, confidence=args.confidence)

    # Share one EasyOCR model across all plates in the image.
    reader = None
    plate_results = []
    for index, detection in enumerate(detections, start=1):
        segment = process_plate_crop(
            plate_crop=detection["crop"], source_image=args.image,
            source_bbox=detection["bbox"], output_dir=args.output_dir,
            character_model=character_model,
            detection_confidence=detection["confidence"],
            detection_index=index, character_confidence=args.character_confidence,
        )
        if reader is None and segment["character_count"]:
            from .recognition import _reader
            reader = _reader()
        plate_results.append(recognize_plate(
            segment["metadata_path"], reader=reader,
            confidence_threshold=args.ocr_confidence,
        ))

    print(json.dumps({"image": args.image, "plates": plate_results}, indent=2))


if __name__ == "__main__":
    main()
