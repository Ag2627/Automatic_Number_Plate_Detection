"""Regression tests for OpenCV's alternate HoughLinesP output shapes."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("line_shape", [(2, 4), (2, 1, 4)])
def test_rectify_plate_accepts_both_hough_line_shapes(monkeypatch, line_shape):
    segments = np.array([[5, 5, 85, 12], [8, 15, 88, 22]], dtype=np.int32)
    cv2 = SimpleNamespace(
        GaussianBlur=lambda image, *_: image,
        Canny=lambda image, *_: image,
        dilate=lambda image, *_args, **_kwargs: image,
        findContours=lambda *_args: ([], None),
        contourArea=lambda *_args: 0,
        HoughLinesP=lambda *_args, **_kwargs: segments.reshape(line_shape),
        getRotationMatrix2D=lambda *_args: np.array([[1, 0, 0], [0, 1, 0]]),
        warpAffine=lambda image, *_args, **_kwargs: image.copy(),
        INTER_CUBIC=2,
        BORDER_REPLICATE=1,
        RETR_LIST=1,
        CHAIN_APPROX_SIMPLE=2,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    source = Path(__file__).resolve().parents[1] / "plate_segmentation" / "rectification.py"
    spec = importlib.util.spec_from_file_location("rectification_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    crop = np.zeros((40, 100), dtype=np.uint8)
    result, info = module.rectify_plate(crop)

    assert result.shape == crop.shape
    assert info["method"] == "rotation"
