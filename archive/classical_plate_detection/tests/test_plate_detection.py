import cv2
import numpy as np
import pytest

from plate_detection import (
    DetectorConfig, edge_map, morphology_map, detect_candidates, detect_plate,
    compare_edge_methods, detect_plate_pyramid, hough_line_segments,
    compare_canny_thresholds, harris_response, evaluate_boxes,
    edge_map_statistics,
    hog_descriptor, train_hog_svm, detect_hog_candidates,
)


@pytest.fixture
def plate_scene():
    image = np.full((360, 640, 3), 155, dtype=np.uint8)
    cv2.rectangle(image, (175, 115), (465, 260), (65, 75, 85), -1)
    cv2.rectangle(image, (265, 185), (385, 232), (245, 245, 245), -1)
    cv2.rectangle(image, (265, 185), (385, 232), (15, 15, 15), 2)
    cv2.putText(image, "UP32AB", (272, 218), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (10, 10, 10), 2, cv2.LINE_AA)
    return image


@pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
def test_edge_maps_are_binary_sized_images(plate_scene, method):
    result = edge_map(plate_scene, method)
    assert result.shape == plate_scene.shape[:2]
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0
    assert 0 <= edge_map_statistics(result)["edge_density"] <= 1


@pytest.mark.parametrize("operation", ["erode", "dilate", "open", "close", "gradient"])
def test_morphology_operations(plate_scene, operation):
    edges = edge_map(plate_scene)
    result = morphology_map(edges, plate_scene.shape, operation)
    assert result.shape == edges.shape
    assert result.dtype == np.uint8


def test_candidate_api_and_input_immutability(plate_scene):
    original = plate_scene.copy()
    candidates, diagnostics = detect_candidates(plate_scene, DetectorConfig())
    crop, box, score, details = detect_plate(plate_scene, return_diagnostics=True)
    assert np.array_equal(plate_scene, original)
    assert set(("gray", "edges", "morphology", "candidates")).issubset(diagnostics)
    assert details["edge_method"] == "canny"
    if candidates:
        x, y, w, h = box
        assert crop.shape[:2] == (h, w)
        assert 0 <= x < plate_scene.shape[1] and 0 <= y < plate_scene.shape[0]
        assert x + w <= plate_scene.shape[1] and y + h <= plate_scene.shape[0]
        assert np.isfinite(score) and 0 <= score <= 1
    else:
        assert crop is None and box is None and score == 0.0


def test_blank_image_returns_clear_no_detection():
    crop, box, score = detect_plate(np.zeros((200, 400, 3), dtype=np.uint8))
    assert crop is None and box is None and score == 0.0


def test_optional_method_comparisons(plate_scene):
    comparisons = compare_edge_methods(plate_scene)
    assert set(comparisons) == {"laplacian", "canny", "sobel"}
    for result in comparisons.values():
        assert set(result["morphology_operations"]) == {
            "erode", "dilate", "open", "close", "gradient"
        }
    lines_image, lines = hough_line_segments(plate_scene)
    assert lines_image.shape == plate_scene.shape[:2]
    assert isinstance(lines, list)
    response = harris_response(plate_scene)
    assert response.shape == plate_scene.shape[:2]
    assert np.isfinite(response).all()
    pyramid = detect_plate_pyramid(plate_scene, scales=(1.0, 0.5))
    assert all("scale" in candidate for candidate in pyramid)
    threshold_results = compare_canny_thresholds(plate_scene)
    assert len(threshold_results) == 3


def test_iou_metrics_and_one_to_one_matching():
    result = evaluate_boxes(
        [{"bbox": (10, 10, 100, 40), "score": 0.9},
         {"bbox": (12, 12, 100, 40), "score": 0.7}],
        [(10, 10, 100, 40)], iou_threshold=0.5,
    )
    assert result["tp"] == 1 and result["fp"] == 1 and result["fn"] == 0
    assert result["precision"] == 0.5 and result["recall"] == 1.0
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["mean_matched_iou"] == pytest.approx(1.0)


def test_hog_descriptor_and_optional_svm(plate_scene):
    positive = plate_scene[185:232, 265:385]
    negative = np.random.default_rng(9).integers(0, 255, (48, 120), dtype=np.uint8)
    assert hog_descriptor(positive).ndim == 1
    model = train_hog_svm([positive, positive.copy(), negative, negative.copy()],
                          [1, 1, -1, -1])
    candidates, _ = detect_hog_candidates(plate_scene, model)
    assert isinstance(candidates, list)


def test_invalid_input_and_method(plate_scene):
    with pytest.raises(ValueError):
        edge_map(plate_scene.astype(np.float32))
    with pytest.raises(ValueError):
        edge_map(plate_scene, "unknown")
    with pytest.raises(ValueError):
        detect_plate_pyramid(plate_scene, scales=(1.0, 0.0))
    with pytest.raises(ValueError):
        DetectorConfig(max_candidates=0)
    with pytest.raises(ValueError):
        edge_map(plate_scene, "canny", 200, 100)

