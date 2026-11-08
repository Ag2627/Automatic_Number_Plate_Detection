import json
from pathlib import Path

import cv2
import numpy as np

from .rectification import rectify_plate
from .preprocessing import prepare_plate
from .character_segmentation import segment_characters


def process_plate_crop(
    plate_crop,
    source_image,
    source_bbox,
    output_dir,
    detection_confidence=None,
    detection_index=0
):

    if plate_crop is None or plate_crop.size == 0:
        raise ValueError("Empty plate crop")

    output = Path(output_dir)

    stem = Path(source_image).stem

    prefix = (
        f"{stem}_plate_{detection_index:02d}"
    )

    folders = [
        "rectified",
        "grayscale",
        "binary",
        "characters"
    ]

    for folder in folders:
        (output / folder).mkdir(
            parents=True,
            exist_ok=True
        )

    # STEP 1: Rectification.
    rectified, rectification_info = (
        rectify_plate(plate_crop)
    )

    # STEP 2: Preprocessing.
    prep = prepare_plate(rectified)

    # STEP 3: Character segmentation.
    segmented = segment_characters(
        prep["thresholds"]
    )

    # Save intermediate results.
    cv2.imwrite(
        str(output / "rectified" / f"{prefix}.jpg"),
        rectified
    )

    cv2.imwrite(
        str(output / "grayscale" / f"{prefix}.png"),
        prep["enhanced"]
    )

    cv2.imwrite(
        str(output / "binary" / f"{prefix}.png"),
        segmented["binary"]
    )

    x0, y0, box_w, box_h = map(
        int, source_bbox
    )

    characters = []

    annotated = cv2.cvtColor(
        segmented["binary"],
        cv2.COLOR_GRAY2BGR
    )

    # STEP 4: Extract each character.
    for index, item in enumerate(
        segmented["characters"],
        start=1
    ):

        x, y, w, h = item["bbox"]

        pad_x = max(2, int(w * 0.12))
        pad_y = max(2, int(h * 0.10))

        xa = max(0, x-pad_x)
        ya = max(0, y-pad_y)

        xb = min(
            prep["enhanced"].shape[1],
            x+w+pad_x
        )

        yb = min(
            prep["enhanced"].shape[0],
            y+h+pad_y
        )

        character_crop = prep["enhanced"][
            ya:yb, xa:xb
        ]

        character_path = (
            output / "characters"
            / f"{prefix}_char_{index:02d}.png"
        )

        cv2.imwrite(
            str(character_path),
            character_crop
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Approximate source-image coordinates.
        # Exact mapping requires inverse geometric transforms.
        sx = box_w / max(1, plate_crop.shape[1])
        sy = box_h / max(1, plate_crop.shape[0])

        characters.append({

            "index": index,

            "crop_path": str(character_path),

            "bbox_rectified_resized": [
                x, y, w, h
            ],

            "bbox_in_original_image_approx": [
                int(x0 + xa / prep["scale_x"] * sx),
                int(y0 + ya / prep["scale_y"] * sy),
                int((xb-xa) / prep["scale_x"] * sx),
                int((yb-ya) / prep["scale_y"] * sy)
            ]
        })

    preview_path = (
        output / "binary"
        / f"{prefix}_segmented.png"
    )

    cv2.imwrite(
        str(preview_path),
        annotated
    )

    # STEP 5: Save metadata for Person 4.
    result = {

        "source_image": str(source_image),

        "plate_index": detection_index,

        "plate_bbox_xywh_original": [
            x0, y0, box_w, box_h
        ],

        "detection_confidence": detection_confidence,

        "rectification": rectification_info,

        "threshold_method": segmented["threshold_name"],

        "segmentation_success": segmented["success"],

        "segmentation_reason": segmented["reason"],

        "character_count": len(characters),

        "characters": characters,

        "outputs": {

            "rectified_plate": str(
                output / "rectified" / f"{prefix}.jpg"
            ),

            "enhanced_grayscale": str(
                output / "grayscale" / f"{prefix}.png"
            ),

            "binary_plate": str(
                output / "binary" / f"{prefix}.png"
            ),

            "segmentation_preview": str(
                preview_path
            )
        }
    }

    metadata_path = (
        output / f"{prefix}.json"
    )

    metadata_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    result["metadata_path"] = str(
        metadata_path
    )

    return result