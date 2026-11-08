"""Optional method comparisons and baselines named in the project plan."""
from pathlib import Path
import random
from dataclasses import replace
import cv2
import numpy as np
from .detector import DetectorConfig, detect_candidates


def compare_edge_methods(image, config=None):
    """Return edge/morphology maps and ranked candidates for Laplacian, Canny, and Sobel."""
    from .detector import morphology_map
    out = {}
    for method in ("laplacian", "canny", "sobel"):
        candidates, diag = detect_candidates(image, config, edge_method=method)
        out[method] = {"edges": diag["edges"], "morphology": diag["morphology"],
                       "morphology_operations": {
                           op: morphology_map(diag["edges"], image.shape, op, config)
                           for op in ("erode", "dilate", "open", "close", "gradient")
                       }, "candidates": candidates,
                       "edge_statistics": edge_map_statistics(diag["edges"])}
    return out


def edge_map_statistics(edges):
    """Simple comparable statistics; visual review is still needed for edge shape."""
    if edges.ndim != 2:
        raise ValueError("edges must be a 2-D edge map")
    count, _, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int32)
    return {"edge_density": float(np.count_nonzero(edges) / edges.size),
            "component_count": int(max(0, count - 1)),
            "mean_component_pixels": float(np.mean(areas)) if areas.size else 0.0,
            "median_component_pixels": float(np.median(areas)) if areas.size else 0.0}


def compare_canny_thresholds(image, threshold_pairs=((40, 120), (70, 150), (100, 200)), config=None):
    """Compare fixed Canny threshold pairs with morphology and candidate scores."""
    from .detector import edge_map, morphology_map
    cfg = config or DetectorConfig()
    results = {}
    for low, high in threshold_pairs:
        if low < 0 or high <= low:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high")
        threshold_cfg = replace(cfg, canny_low=low, canny_high=high)
        candidates, diag = detect_candidates(image, threshold_cfg, edge_method="canny")
        edges, morph = diag["edges"], diag["morphology"]
        results[f"{low}_{high}"] = {"thresholds": (low, high), "edges": edges,
                                      "morphology": morph, "candidates": candidates,
                                      "edge_density": float(np.count_nonzero(edges)/edges.size),
                                      "edge_statistics": edge_map_statistics(edges)}
    return results


def hough_line_segments(image, threshold=40, min_line_length=None, max_line_gap=8):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    min_line_length = min_line_length or max(20, gray.shape[1] // 12)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    # OpenCV versions differ: HoughLinesP may return (N, 1, 4) or (N, 4).
    # Normalize the shape before iterating so both layouts work.
    segments = [] if lines is None else [tuple(map(int, line)) for line in np.asarray(lines).reshape(-1, 4)]
    return edges, segments


def harris_response(image, block_size=2, aperture=3, k=0.04):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = cv2.cornerHarris(np.float32(gray), block_size, aperture, k)
    return cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)


def detect_plate_pyramid(image, scales=(1.0, 0.75, 0.5), config=None):
    """Run candidate search at multiple scales and map boxes to original pixels."""
    cfg = config or DetectorConfig()
    h, w = image.shape[:2]
    all_candidates = []
    for scale in scales:
        if scale <= 0:
            raise ValueError("scales must be positive")
        resized = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
        candidates, _ = detect_candidates(resized, cfg)
        for c in candidates:
            x, y, bw, bh = c["bbox"]
            box = (round(x/scale), round(y/scale), round(bw/scale), round(bh/scale))
            all_candidates.append({**c, "bbox": box, "scale": float(scale)})
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    return all_candidates


def haar_candidates(image, cascade_path, scale_factor=1.1, min_neighbors=3):
    """Optional Haar baseline; accepts a path or ``opencv-russian``."""
    if str(cascade_path).lower() in ("opencv-russian", "default"):
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if not data_dir:
            raise FileNotFoundError("This OpenCV install does not expose Haar cascade data")
        cascade_path = Path(data_dir) / "haarcascade_russian_plate_number.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Could not load Haar cascade: {cascade_path}")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                     minNeighbors=min_neighbors, minSize=(24, 10))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


def hog_descriptor(image, size=(96, 32)):
    """Compute a fixed-length HOG descriptor for a candidate plate crop."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    hog = cv2.HOGDescriptor(size, (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def train_hog_svm(samples, labels, model_path=None):
    """Train optional OpenCV linear SVM from labeled plate/non-plate crops.

    labels must be 1 for plate and -1 for non-plate. A useful model needs a
    representative, vehicle-disjoint labeled training set.
    """
    if len(samples) != len(labels) or len(samples) < 4:
        raise ValueError("provide at least four image crops and matching labels")
    features = np.vstack([hog_descriptor(im) for im in samples]).astype(np.float32)
    y = np.asarray(labels, np.int32).reshape(-1, 1)
    if set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError("labels must include both -1 (negative) and 1 (plate)")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6))
    svm.train(features, cv2.ml.ROW_SAMPLE, y)
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        svm.save(str(model_path))
    return svm


def build_hog_training_samples(annotation_csv, negatives_per_image=5, seed=7):
    """Build positive plate and random non-plate crops from box annotations.

    Keep train and test annotation files separate. Negative crops are sampled
    from the same vehicle images and rejected when they overlap a labeled plate.
    """
    from .evaluation import read_annotations, _iou
    annotations = read_annotations(annotation_csv)
    rng = random.Random(seed)
    samples, labels = [], []
    for filename, boxes in sorted(annotations.items()):
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        height, width = image.shape[:2]
        for x, y, w, h in boxes:
            crop = image[max(0,y):min(height,y+h), max(0,x):min(width,x+w)]
            if crop.size:
                samples.append(crop.copy())
                labels.append(1)
        for _ in range(max(0, negatives_per_image)):
            made = False
            for _attempt in range(100):
                _, _, bw, bh = rng.choice(boxes)
                bw = min(max(8, bw), width)
                bh = min(max(8, bh), height)
                nx = rng.randint(0, max(0, width - bw))
                ny = rng.randint(0, max(0, height - bh))
                negative_box = (nx, ny, bw, bh)
                if all(_iou(negative_box, gt) < 0.05 for gt in boxes):
                    samples.append(image[ny:ny+bh, nx:nx+bw].copy())
                    labels.append(-1)
                    made = True
                    break
            if not made:
                continue
    if not samples or 1 not in labels or -1 not in labels:
        raise ValueError("annotations did not yield positive and negative HOG crops")
    return samples, labels


def train_hog_from_annotations(annotation_csv, model_path, negatives_per_image=5, seed=7):
    """Train and save the optional HOG-SVM from the training annotation CSV."""
    samples, labels = build_hog_training_samples(
        annotation_csv, negatives_per_image=negatives_per_image, seed=seed)
    model = train_hog_svm(samples, labels, model_path)
    return model, {"samples": len(labels), "positive": labels.count(1),
                   "negative": labels.count(-1)}


def detect_hog_candidates(image, svm, config=None):
    candidates, diagnostics = detect_candidates(image, config)
    ranked = []
    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        crop = image[y:y+h, x:x+w]
        feature = hog_descriptor(crop).reshape(1, -1)
        _, prediction = svm.predict(feature)
        _, raw = svm.predict(feature, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        if int(prediction[0, 0]) == 1:
            ranked.append({**candidate, "svm_margin": float(raw[0, 0])})
    ranked.sort(key=lambda c: (c["svm_margin"], c["score"]), reverse=True)
    return ranked, diagnostics

