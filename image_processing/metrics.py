"""Image quality metrics and downstream detector comparison helpers."""
import numpy as np
try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def _gray_float(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float64)


def mse(reference, test):
    a, b = _gray_float(reference), _gray_float(test)
    if a.shape != b.shape:
        raise ValueError("Images must have the same dimensions.")
    return float(np.mean((a - b) ** 2))


def psnr(reference, test, data_range=255.0):
    error = mse(reference, test)
    if error == 0:
        return float("inf")
    return float(10 * np.log10((data_range ** 2) / error))


def ssim(reference, test):
    if structural_similarity is None:
        raise ImportError("Install scikit-image to calculate SSIM.")
    a, b = _gray_float(reference), _gray_float(test)
    if a.shape != b.shape:
        raise ValueError("Images must have the same dimensions.")
    return float(structural_similarity(a, b, data_range=255.0))


def quality_report(reference, test):
    return {"mse": mse(reference, test), "psnr": psnr(reference, test), "ssim": ssim(reference, test)}


def detection_rate(successes, total):
    """Return success rate as a fraction; use same fixed image set for comparisons."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total > 0.")
    return successes / total
