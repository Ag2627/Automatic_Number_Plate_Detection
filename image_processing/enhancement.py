"""Histogram operations and image enhancement."""
import cv2
import numpy as np
from .fundamentals import to_grayscale


def histogram(image, channels=None, bins=256, value_range=(0, 256)):
    """Return histogram arrays. For color BGR, channels defaults to all channels."""
    if image.ndim == 2:
        channels = [0]
    elif channels is None:
        channels = [0, 1, 2]
    return [cv2.calcHist([image], [ch], None, [bins], list(value_range)).ravel()
            for ch in channels]


def histogram_equalization(image):
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def histogram_matching(source, reference):
    """Match source intensity histogram to reference; preserves source color using luminance."""
    src_gray = to_grayscale(source)
    ref_gray = to_grayscale(reference)
    src_values, src_indices, src_counts = np.unique(
        src_gray.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(ref_gray.ravel(), return_counts=True)
    src_quantiles = np.cumsum(src_counts).astype(np.float64) / src_gray.size
    ref_quantiles = np.cumsum(ref_counts).astype(np.float64) / ref_gray.size
    mapped = np.interp(src_quantiles, ref_quantiles, ref_values)
    matched_gray = mapped[src_indices].reshape(src_gray.shape).clip(0, 255).astype(np.uint8)
    if source.ndim == 2:
        return matched_gray
    ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = matched_gray
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def contrast_stretch(image, low_percentile=2, high_percentile=98):
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Invalid percentiles.")
    arr = image.astype(np.float32)
    if image.ndim == 2:
        lo, hi = np.percentile(arr, [low_percentile, high_percentile])
        if hi <= lo:
            return image.copy()
        return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        lo, hi = np.percentile(arr[:, :, c], [low_percentile, high_percentile])
        if hi <= lo:
            out[:, :, c] = image[:, :, c]
        else:
            out[:, :, c] = np.clip((arr[:, :, c]-lo)*255.0/(hi-lo), 0, 255)
    return out.astype(np.uint8)


def gamma_correction(image, gamma=1.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_image(image, method="clahe", **kwargs):
    methods = {
        "equalize": histogram_equalization,
        "contrast": contrast_stretch,
        "gamma": gamma_correction,
        "clahe": clahe_enhancement,
    }
    if method not in methods:
        raise ValueError(f"Unknown enhancement method '{method}'. Choose {list(methods)}.")
    return methods[method](image, **kwargs)
