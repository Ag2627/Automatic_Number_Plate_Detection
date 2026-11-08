"""Classical Indian number-plate localization tools."""
from .detector import (
    DetectorConfig, edge_map, morphology_map, detect_candidates,
    detect_plate, save_diagnostics,
)
from .advanced import (
    compare_edge_methods, compare_canny_thresholds,
    edge_map_statistics, hough_line_segments, harris_response,
    detect_plate_pyramid, haar_candidates, hog_descriptor,
    train_hog_svm, detect_hog_candidates, build_hog_training_samples,
    train_hog_from_annotations,
)
from .evaluation import (
    evaluate_boxes, evaluate_dataset, evaluate_haar_dataset,
    evaluate_preprocessing_comparison,
    evaluate_hog_dataset,
)

__all__ = [name for name in globals() if not name.startswith("_")]

