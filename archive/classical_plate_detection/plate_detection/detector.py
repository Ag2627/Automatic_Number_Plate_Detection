"""Explainable edge/morphology/contour number-plate detector.

This is a classical computer-vision baseline. Its score is a ranking score,
not a calibrated probability. Tune thresholds on validation data only.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    edge_method: str = "canny"
    min_area_fraction: float = 0.00035
    # Close-up plate crops can occupy a large portion of the input frame.
    max_area_fraction: float = 0.42
    min_aspect_ratio: float = 1.25
    max_aspect_ratio: float = 7.0
    canny_low: int = 55
    canny_high: int = 165
    close_width_fraction: float = 0.035
    close_height_fraction: float = 0.012
    max_candidates: int = 100
    nms_iou: float = 0.65

    def __post_init__(self):
        if not 0 <= self.min_area_fraction < self.max_area_fraction <= 1:
            raise ValueError("area fractions must satisfy 0 <= min < max <= 1")
        if not 0 < self.min_aspect_ratio < self.max_aspect_ratio:
            raise ValueError("aspect ratio bounds must be positive and ordered")
        if self.canny_low < 0 or self.canny_high <= self.canny_low:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high")
        if self.close_width_fraction <= 0 or self.close_height_fraction <= 0:
            raise ValueError("morphology kernel fractions must be positive")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou must be in [0, 1]")


def _validate_image(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a non-empty NumPy array")
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("image must be uint8 grayscale or uint8 BGR")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color input must have three BGR channels")
    if not image.size:
        raise ValueError("image cannot be empty")


def _gray(image):
    _validate_image(image)
    return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def edge_map(image, method="canny", canny_low=55, canny_high=165):
    """Return uint8 edges using Canny, Sobel, or Laplacian."""
    gray = _gray(image)
    if canny_low < 0 or canny_high <= canny_low:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    key = method.lower()
    if key == "canny":
        return cv2.Canny(smooth, canny_low, canny_high)
    if key == "sobel":
        kx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        ky = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(kx, ky)
        mag8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if key == "laplacian":
        response = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
        abs8 = cv2.convertScaleAbs(response)
        _, binary = cv2.threshold(abs8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    raise ValueError("method must be canny, sobel, or laplacian")
def morphology_map(edges, image_shape, operation="close", config=None):
    """Apply a named morphology operation with image-relative kernels."""
    config = config or DetectorConfig()
    if not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.dtype != np.uint8:
        raise ValueError("edges must be a 2-D uint8 image")
    h, w = image_shape[:2]
    kw = max(3, int(round(w * config.close_width_fraction))) | 1
    kh = max(3, int(round(h * config.close_height_fraction))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    ops = {
        "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
    }
    if operation not in ops:
        raise ValueError(f"operation must be one of {tuple(ops)}")
    return cv2.morphologyEx(edges, ops[operation], kernel)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0.0


def _rank_candidate(gray, edges, contour, rect, image_area, config):
    x, y, w, h = rect
    area = w * h
    ratio = w / max(h, 1)
    rect_contour = abs(cv2.contourArea(contour)) / max(area, 1)
    edge_density = np.count_nonzero(edges[y:y+h, x:x+w]) / max(area, 1)
    crop = gray[y:y+h, x:x+w]
    if crop.size:
        ink = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
        stroke_density = np.count_nonzero(ink) / ink.size
    else:
        stroke_density = 0.0
    # Broad geometry prior: Indian plates include both long single-row and
    # shorter/two-row layouts. Scores are ranking cues, not probabilities.
    ratio_score = float(np.exp(-abs(np.log(max(ratio, 1e-6) / 4.0))))
    area_fraction = area / image_area
    area_score = min(1.0, area_fraction / 0.025)
    edge_score = min(1.0, edge_density / 0.35)
    stroke_score = float(np.exp(-abs(stroke_density - 0.28) / 0.32))
    score = (0.25 * ratio_score + 0.18 * area_score +
             0.22 * min(1.0, rect_contour) + 0.20 * edge_score +
             0.15 * stroke_score)
    return {
        "bbox": (int(x), int(y), int(w), int(h)), "score": float(score),
        "area_fraction": float(area_fraction), "aspect_ratio": float(ratio),
        "rectangularity": float(rect_contour), "edge_density": float(edge_density),
        "stroke_density": float(stroke_density),
    }


def detect_candidates(image, config=None, edge_method=None):
    """Find and rank possible plate boxes from contours and connected components."""
    _validate_image(image)
    cfg = config or DetectorConfig()
    method = edge_method or cfg.edge_method
    gray = _gray(image)
    edges = edge_map(gray, method, cfg.canny_low, cfg.canny_high)
    morph = morphology_map(edges, gray.shape, "close", cfg)
    h, w = gray.shape
    image_area = h * w
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Connected components provide a second candidate-generation path.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(morph, 8)
    rects = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        rects.append(((x, y, rw, rh), contour))
    for i in range(1, count):
        x, y, rw, rh, _ = stats[i]
        if rw > 0 and rh > 0:
            rects.append(((int(x), int(y), int(rw), int(rh)), None))
    candidates = []
    for rect, contour in rects:
        x, y, rw, rh = rect
        frac = rw * rh / image_area
        ratio = rw / max(rh, 1)
        if not (cfg.min_area_fraction <= frac <= cfg.max_area_fraction):
            continue
        if not (cfg.min_aspect_ratio <= ratio <= cfg.max_aspect_ratio):
            continue
        if contour is None:
            contour = np.array([[[x,y]], [[x+rw,y]], [[x+rw,y+rh]], [[x,y+rh]]], np.int32)
        candidates.append(_rank_candidate(gray, edges, contour, rect, image_area, cfg))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for candidate in candidates:
        if all(_iou(candidate["bbox"], old["bbox"]) < cfg.nms_iou for old in kept):
            kept.append(candidate)
        if len(kept) >= cfg.max_candidates:
            break
    diagnostics = {"gray": gray, "edges": edges, "morphology": morph,
                   "candidates": kept, "edge_method": method}
    return kept, diagnostics


def detect_plate(image, config=None, edge_method=None, return_diagnostics=False):
    """Return ``(plate_crop, (x,y,w,h), score)``; no detection is ``(None,None,0)``."""
    candidates, diagnostics = detect_candidates(image, config, edge_method)
    if candidates:
        x, y, w, h = candidates[0]["bbox"]
        crop = image[y:y+h, x:x+w].copy()
        result = (crop, (x, y, w, h), candidates[0]["score"])
    else:
        result = (None, None, 0.0)
    return (*result, diagnostics) if return_diagnostics else result


def save_diagnostics(image, diagnostics, output_dir, prefix="image"):
    """Save edge, morphology, candidate-box, and selected-box views."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{prefix}_edges.png"), diagnostics["edges"])
    cv2.imwrite(str(out / f"{prefix}_morphology.png"), diagnostics["morphology"])
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, candidate in enumerate(diagnostics["candidates"]):
        x, y, w, h = candidate["bbox"]
        color = (0, 0, 255) if idx == 0 else (0, 200, 255)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        cv2.putText(canvas, f"{idx+1}: {candidate['score']:.2f}", (x, max(15, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / f"{prefix}_candidates.png"), canvas)
    if diagnostics["candidates"]:
        x, y, w, h = diagnostics["candidates"][0]["bbox"]
        cv2.imwrite(str(out / f"{prefix}_selected_crop.png"), image[y:y+h, x:x+w])

