
import csv
from pathlib import Path

import cv2

from plate_detection import detect_plate_boxes, load_plate_model
from plate_segmentation.ocr_engine import PlateOCR
from plate_segmentation.pipeline import process_plate_crop


# ---------------- CONFIGURATION ----------------

IMAGE_DIR = Path("data/images/test")
OUTPUT_DIR = Path("detection_results/batch_ocr")
CSV_PATH = OUTPUT_DIR / "ocr_results.csv"

PLATE_CONFIDENCE = 0.25

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ------------------------------------------------


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_DIR.resolve()}"
        )

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    print(f"Found {len(image_paths)} images.")

    # Load YOLO plate detector only once.
    print("\nLoading YOLO plate detector...")
    plate_model = load_plate_model()

    # Load PaddleOCR only once.
    print("Loading PaddleOCR...")
    ocr_engine = PlateOCR()

    rows = []

    for image_number, image_path in enumerate(image_paths, start=1):

        print("\n" + "=" * 55)
        print(f"Processing {image_number}/{len(image_paths)}: {image_path.name}")

        try:
            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError("Could not read image.")

            # 1. Detect number plates using existing YOLO model.
            detections = detect_plate_boxes(
                image,
                model=plate_model,
                confidence=PLATE_CONFIDENCE
            )

            if not detections:
                print("No number plate detected.")

                rows.append({
                    "image": image_path.name,
                    "plate_number": "",
                    "plate_confidence": "",
                    "ocr_confidence": "",
                    "status": "No plate detected"
                })
                continue

            print(f"Detected {len(detections)} plate(s).")

            # 2. Run rectification, preprocessing and OCR.
            for plate_index, detection in enumerate(
                detections, start=1
            ):

                result = process_plate_crop(
                    plate_crop=detection["crop"],
                    source_image=str(image_path),
                    source_bbox=detection["bbox"],
                    output_dir=str(OUTPUT_DIR),
                    detection_confidence=detection["confidence"],
                    detection_index=plate_index,
                    ocr_engine=ocr_engine,
                )

                recognized_text = result["recognized_text"]
                ocr_confidence = result["ocr_confidence"]

                print(f"\nPlate {plate_index}:")
                print(
                    "Recognized number:",
                    recognized_text or "Not recognized"
                )
                print(f"OCR confidence: {ocr_confidence:.4f}")

                rows.append({
                    "image": image_path.name,
                    "plate_number": recognized_text,
                    "plate_confidence": detection["confidence"],
                    "ocr_confidence": ocr_confidence,
                    "status": (
                        "Success" if recognized_text
                        else "OCR returned no text"
                    )
                })

        except Exception as error:

            print(f"Error processing {image_path.name}: {error}")

            rows.append({
                "image": image_path.name,
                "plate_number": "",
                "plate_confidence": "",
                "ocr_confidence": "",
                "status": f"Error: {error}"
            })

    # 3. Save all results to CSV.
    fieldnames = [
        "image",
        "plate_number",
        "plate_confidence",
        "ocr_confidence",
        "status"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(
        1 for row in rows
        if row["status"] == "Success"
    )

    print("\n" + "=" * 55)
    print("BATCH TEST COMPLETED")
    print(f"Images processed: {len(image_paths)}")
    print(f"Successful OCR results: {successful}")
    print(f"Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
