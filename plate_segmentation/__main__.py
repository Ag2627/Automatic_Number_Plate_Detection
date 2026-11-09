import argparse
import cv2
from plate_detection import detect_plate_boxes, load_plate_model
from .character_detector import load_character_model
from .pipeline import process_plate_crop

def main():
    parser = argparse.ArgumentParser(
        description="Person 2 plate detection + Person 3 character localization"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="detection_results/segmentation")
    parser.add_argument("--weights", default=None,
                        help="Person 2 plate detector weights")
    parser.add_argument("--character-weights", default="models/character_detector.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--character-confidence", type=float, default=0.25)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    plate_model = load_plate_model(args.weights)
    character_model = load_character_model(args.character_weights)
    detections = detect_plate_boxes(image, model=plate_model,
                                    confidence=args.confidence)
    if not detections:
        print("No number plate detected.")
        return

    print(f"Detected {len(detections)} plate(s)")
    for index, detection in enumerate(detections, start=1):
        result = process_plate_crop(
            plate_crop=detection["crop"],
            source_image=args.image,
            source_bbox=detection["bbox"],
            output_dir=args.output_dir,
            character_model=character_model,
            detection_confidence=detection["confidence"],
            detection_index=index,
            character_confidence=args.character_confidence
        )
        print(f"\nPlate: {index}")
        print("Segmentation successful:", result["segmentation_success"])
        print("Character count:", result["character_count"])
        print("Metadata:", result["metadata_path"])

if __name__ == "__main__":
    main()
