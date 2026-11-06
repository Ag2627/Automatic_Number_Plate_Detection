"""Image I/O, color models, sampling, quantization and pixel relationships."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    """Read an image as BGR uint8; raise a useful error if unreadable."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save image: {path}")


def to_grayscale(image):
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_hsi(image):
    """Return H, S, I float arrays (H in radians, S and I in [0,1])."""
    rgb = bgr_to_rgb(image).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-8), -1.0, 1.0))
    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    minimum = np.minimum(np.minimum(r, g), b)
    intensity = (r + g + b) / 3.0
    saturation = np.where(intensity > 1e-8, 1.0 - minimum / (intensity + 1e-8), 0.0)
    return hue.astype(np.float32), saturation.astype(np.float32), intensity.astype(np.float32)


def resize_image(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if width is None and height is None:
        raise ValueError("Provide width or height.")
    if width is None:
        width = max(1, round(w * height / h))
    if height is None:
        height = max(1, round(h * width / w))
    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def crop_image(image, x, y, width, height):
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, x1 + int(width)), min(h, y1 + int(height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Crop lies outside the image or has zero area.")
    return image[y1:y2, x1:x2].copy()


def sample_image(image, row_step=2, col_step=2):
    """Spatial sampling by retaining every nth row and column."""
    if row_step < 1 or col_step < 1:
        raise ValueError("Sampling steps must be >= 1.")
    return image[::row_step, ::col_step].copy()


def quantize_image(image, levels=8):
    """Uniformly quantize uint8 image intensities to `levels` gray/color levels."""
    if not 2 <= levels <= 256:
        raise ValueError("levels must be between 2 and 256.")
    step = 256.0 / levels
    quantized = np.floor(image.astype(np.float32) / step) * step + step / 2
    return np.clip(quantized, 0, 255).astype(np.uint8)


def neighbors_4(row, col, height, width):
    candidates = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    return [(r, c) for r, c in candidates if 0 <= r < height and 0 <= c < width]


def neighbors_8(row, col, height, width):
    return [(r, c) for r in range(row-1, row+2)
            for c in range(col-1, col+2)
            if (r, c) != (row, col) and 0 <= r < height and 0 <= c < width]


def pixel_distance(p1, p2, metric="euclidean"):
    """Distance between (row, col) points: euclidean, cityblock or chessboard."""
    dr, dc = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
    if metric == "euclidean":
        return float(np.hypot(dr, dc))
    if metric == "cityblock":
        return float(dr + dc)
    if metric == "chessboard":
        return float(max(dr, dc))
    raise ValueError("metric must be euclidean, cityblock or chessboard")
