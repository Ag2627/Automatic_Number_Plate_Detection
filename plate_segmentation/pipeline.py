import json
from pathlib import Path
import cv2
from .rectification import rectify_plate
from .preprocessing import prepare_plate
from .character_detector import detect_characters

def process_plate_crop(
    plate_crop, source_image, source_bbox, output_dir, character_model,
    detection_confidence=None, detection_index=0, character_confidence=0.25
):
    """Rectify a plate, detect character boxes with trained YOLO, and save crops."""
    if plate_crop is None or plate_crop.size == 0:
        raise ValueError("Empty plate crop")
    if character_model is None:
        raise ValueError("character_model must be loaded before processing")

    output = Path(output_dir)
    prefix = f"{Path(source_image).stem}_plate_{detection_index:02d}"
    for folder in ("rectified", "grayscale", "characters", "previews"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    rectified, rectification_info = rectify_plate(plate_crop)
    prep = prepare_plate(rectified)
    detections = detect_characters(
        prep["resized"], character_model,
        confidence=character_confidence, image_size=640
    )

    rectified_path = output / "rectified" / f"{prefix}.jpg"
    enhanced_path = output / "grayscale" / f"{prefix}.png"
    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(enhanced_path), prep["enhanced"])

    annotated = cv2.cvtColor(prep["enhanced"], cv2.COLOR_GRAY2BGR)
    x0, y0, box_w, box_h = map(int, source_bbox)
    sx = box_w / max(1, plate_crop.shape[1])
    sy = box_h / max(1, plate_crop.shape[0])
    characters = []

    for index, item in enumerate(detections, start=1):
        x, y, w, h = item["bbox"]
        pad_x, pad_y = max(2, int(round(w*0.10))), max(2, int(round(h*0.10)))
        xa, ya = max(0, x-pad_x), max(0, y-pad_y)
        xb = min(prep["enhanced"].shape[1], x+w+pad_x)
        yb = min(prep["enhanced"].shape[0], y+h+pad_y)
        if xb <= xa or yb <= ya:
            continue
        crop = prep["enhanced"][ya:yb, xa:xb]
        crop_path = output / "characters" / f"{prefix}_char_{len(characters)+1:02d}.png"
        cv2.imwrite(str(crop_path), crop)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(annotated, str(index), (x,max(12,y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
        characters.append({
            "index": len(characters)+1,
            "crop_path": str(crop_path),
            "confidence": item["confidence"],
            "bbox_rectified_resized": [x,y,w,h],
            "bbox_in_original_image_approx": [
                int(x0 + xa/prep["scale_x"]*sx),
                int(y0 + ya/prep["scale_y"]*sy),
                int((xb-xa)/prep["scale_x"]*sx),
                int((yb-ya)/prep["scale_y"]*sy)
            ]
        })

    preview_path = output / "previews" / f"{prefix}_characters.png"
    cv2.imwrite(str(preview_path), annotated)
    success = len(characters) > 0
    result = {
        "source_image": str(source_image),
        "plate_index": detection_index,
        "plate_bbox_xywh_original": [x0,y0,box_w,box_h],
        "detection_confidence": detection_confidence,
        "rectification": rectification_info,
        "character_model": "YOLO one-class character detector",
        "character_confidence_threshold": character_confidence,
        "segmentation_success": success,
        "segmentation_reason": None if success else "YOLO found no character boxes",
        "character_count": len(characters),
        "characters": characters,
        "outputs": {
            "rectified_plate": str(rectified_path),
            "enhanced_grayscale": str(enhanced_path),
            "character_preview": str(preview_path)
        }
    }
    metadata_path = output / f"{prefix}.json"
    metadata_path.write_text(
    json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
    encoding="utf-8"
)
    result["metadata_path"] = str(metadata_path)
    return result
