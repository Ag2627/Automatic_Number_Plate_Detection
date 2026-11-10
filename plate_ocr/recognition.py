"""Turn ordered character crops in segmentation metadata into plate text."""

import json
from pathlib import Path


ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _crop_file(crop_path, metadata_file):
    """Support absolute paths and both common relative-path conventions."""
    path = Path(crop_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [metadata_file.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Character crop not found: {crop_path}")


def _reader():
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError("Install EasyOCR with `python -m pip install -r requirements.txt`.") from exc
    # Character boxes already exist, so the text detector is unnecessary.
    return easyocr.Reader(["en"], gpu=False, detector=False, verbose=False)


def recognize_plate(metadata_path, reader=None, confidence_threshold=0.5):
    """Recognize each character crop in metadata order and return plate text.

    ``reader`` may be an existing EasyOCR Reader, useful for multiple plates.
    A missing or ambiguous character is flagged for review; its first returned
    alphanumeric symbol is used when available.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    metadata_file = Path(metadata_path).expanduser().resolve()
    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    entries = metadata.get("characters")
    if not isinstance(entries, list):
        raise ValueError("Metadata must contain a characters list")

    # Segmentation writes characters in reading order. Preserve that order.
    results = []
    needs_review = not entries
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not entry.get("crop_path"):
            raise ValueError(f"Character {position} has no crop_path")
        crop_file = _crop_file(entry["crop_path"], metadata_file)
        if reader is None:
            reader = _reader()
        # recognize() processes the entire crop without trying to detect text
        # inside a narrow, already segmented character image.
        predictions = reader.recognize(str(crop_file), allowlist=ALLOWLIST, detail=1)
        if predictions:
            # A crop should contain one symbol. Keep the first if OCR produces
            # more, but mark that result for review.
            raw_text = str(predictions[0][1]).upper()
            symbols = "".join(char for char in raw_text if char in ALLOWLIST)
            prediction = symbols[:1]
            confidence = float(predictions[0][2])
            if len(predictions) != 1 or len(symbols) != 1:
                needs_review = True
        else:
            prediction = ""
            confidence = 0.0
        if not prediction or confidence < confidence_threshold:
            needs_review = True
        results.append({
            "index": entry.get("index", position),
            "prediction": prediction,
            "confidence": confidence,
        })

    return {
        "metadata_path": str(metadata_file),
        "predicted_plate": "".join(item["prediction"] for item in results),
        "characters": results,
        "needs_review": needs_review,
    }
