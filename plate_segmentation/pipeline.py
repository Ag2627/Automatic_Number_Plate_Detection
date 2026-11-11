
import json
from pathlib import Path

import cv2

from .rectification import rectify_plate
from .preprocessing import prepare_plate


def process_plate_crop(
    plate_crop,
    source_image,
    source_bbox,
    output_dir,
    detection_confidence,
    detection_index,
    ocr_engine,
):
    output_dir = Path(output_dir)

    rectified_dir = output_dir / "rectified"
    preview_dir = output_dir / "previews"

    rectified_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{Path(source_image).stem}"
        f"_plate_{detection_index:02d}"
    )

    # 1. Rectify the detected plate.
    rectified, rectification_info = rectify_plate(plate_crop)

    # 2. Preprocess the complete plate.
    prep = prepare_plate(rectified)

    enhanced = prep["enhanced"]

    # 3. Upscale the plate for OCR.
    enlarged = cv2.resize(
        enhanced,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    # PaddleOCR receives a 3-channel image.
    ocr_image = cv2.cvtColor(
        enlarged,
        cv2.COLOR_GRAY2BGR
    )

    # 4. Recognize the complete plate.
    ocr_result = ocr_engine.recognize(ocr_image)

    recognized_text = ocr_result["text"]

    # 5. Save intermediate images.
    rectified_path = rectified_dir / f"{stem}.jpg"
    enhanced_path = rectified_dir / f"{stem}_enhanced.jpg"
    preview_path = preview_dir / f"{stem}_ocr.png"

    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(enhanced_path), enhanced)

    # 6. Create OCR preview.
    preview = cv2.cvtColor(
        enlarged,
        cv2.COLOR_GRAY2BGR
    )

    display_text = (
        recognized_text
        if recognized_text
        else "No text recognized"
    )

    cv2.putText(
        preview,
        display_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(preview_path), preview)

    # 7. Save metadata.
    metadata = {
        "source_image": str(source_image),
        "plate_bbox": [int(v) for v in source_bbox],
        "plate_detection_confidence": float(
            detection_confidence
        ),
        "rectification": rectification_info,
        "recognized_text": recognized_text,
        "ocr_raw_text": ocr_result["raw_text"],
        "ocr_confidence": float(
            ocr_result["confidence"]
        ),
        "ocr_success": bool(recognized_text),
        "outputs": {
            "rectified": str(rectified_path),
            "enhanced": str(enhanced_path),
            "preview": str(preview_path),
        },
    }

    metadata_path = output_dir / f"{stem}.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            ),
        ),
        encoding="utf-8",
    )

    return metadata
