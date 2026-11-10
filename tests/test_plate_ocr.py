"""Check OCR assembly without downloading EasyOCR models."""

import json
from pathlib import Path

import pytest

from plate_ocr import recognize_plate


class FakeReader:
    def __init__(self, outputs):
        self.outputs = outputs
        self.paths = []

    def recognize(self, path, **kwargs):
        self.paths.append(Path(path).name)
        assert kwargs == {"allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "detail": 1}
        return self.outputs[Path(path).name]


def metadata(tmp_path, names):
    crop_dir = tmp_path / "characters"
    crop_dir.mkdir()
    entries = []
    for index, name in enumerate(names, start=1):
        (crop_dir / name).write_bytes(b"image")
        entries.append({"index": index, "crop_path": f"characters/{name}"})
    path = tmp_path / "plate.json"
    path.write_text(json.dumps({"characters": entries}), encoding="utf-8")
    return path


def test_recognizes_in_metadata_order(tmp_path):
    path = metadata(tmp_path, ["u.png", "p.png", "3.png"])
    reader = FakeReader({
        "u.png": [(None, "u", 0.91)],
        "p.png": [(None, "P", 0.82)],
        "3.png": [(None, "3", 0.79)],
    })

    result = recognize_plate(path, reader=reader)

    assert result["predicted_plate"] == "UP3"
    assert reader.paths == ["u.png", "p.png", "3.png"]
    assert result["characters"] == [
        {"index": 1, "prediction": "U", "confidence": 0.91},
        {"index": 2, "prediction": "P", "confidence": 0.82},
        {"index": 3, "prediction": "3", "confidence": 0.79},
    ]
    assert result["needs_review"] is False


def test_marks_missing_ambiguous_and_low_confidence_for_review(tmp_path):
    path = metadata(tmp_path, ["empty.png", "multiple.png", "low.png"])
    reader = FakeReader({
        "empty.png": [],
        "multiple.png": [(None, "AB", 0.95)],
        "low.png": [(None, "7", 0.2)],
    })

    result = recognize_plate(path, reader=reader)

    assert result["predicted_plate"] == "A7"
    assert result["characters"][0]["prediction"] == ""
    assert result["needs_review"] is True


def test_bad_crop_path_is_reported(tmp_path):
    path = tmp_path / "plate.json"
    path.write_text(json.dumps({"characters": [{"crop_path": "missing.png"}]}))

    with pytest.raises(FileNotFoundError, match="missing.png"):
        recognize_plate(path, reader=FakeReader({}))


def test_resolves_paths_written_by_segmentation(tmp_path, monkeypatch):
    """Segmentation stores paths relative to the working directory."""
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "detection_results" / "segmentation"
    crop = output / "characters" / "char_01.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"image")
    path = output / "plate.json"
    path.write_text(json.dumps({"characters": [{
        "index": 1,
        "crop_path": "detection_results/segmentation/characters/char_01.png",
    }]}))

    result = recognize_plate(path, reader=FakeReader({"char_01.png": [(None, "B", 0.9)]}))

    assert result["predicted_plate"] == "B"
