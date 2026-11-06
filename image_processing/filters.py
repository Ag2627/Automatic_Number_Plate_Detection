"""Spatial filters: mean variants, median, Gaussian, directional and bilateral."""
import cv2
import numpy as np


def _apply_per_channel(image, fn):
    if image.ndim == 2:
        return fn(image)
    return np.stack([fn(image[:, :, c]) for c in range(image.shape[2])], axis=2)


def arithmetic_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))


def geometric_mean_filter(image, kernel_size=3, epsilon=1e-6):
    def one(channel):
        x = np.maximum(channel.astype(np.float32), epsilon)
        log_mean = cv2.blur(np.log(x), (kernel_size, kernel_size))
        return np.exp(log_mean).clip(0, 255).astype(np.uint8)
    return _apply_per_channel(image, one)


def harmonic_mean_filter(image, kernel_size=3, epsilon=1e-6):
    def one(channel):
        x = channel.astype(np.float32)
        reciprocal_mean = cv2.blur(1.0 / np.maximum(x, epsilon),
                                   (kernel_size, kernel_size))
        return (1.0 / np.maximum(reciprocal_mean, epsilon)).clip(0, 255).astype(np.uint8)
    return _apply_per_channel(image, one)


def contra_harmonic_mean_filter(image, kernel_size=3, q=1.5, epsilon=1e-6):
    """Q>0 tends to remove pepper noise; Q<0 tends to remove salt noise."""
    def one(channel):
        x = channel.astype(np.float32)
        numerator = cv2.blur(np.power(np.maximum(x, epsilon), q+1),
                             (kernel_size, kernel_size))
        denominator = cv2.blur(np.power(np.maximum(x, epsilon), q),
                               (kernel_size, kernel_size))
        return (numerator / np.maximum(denominator, epsilon)).clip(0, 255).astype(np.uint8)
    return _apply_per_channel(image, one)


def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)


def gaussian_filter(image, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def directional_filter(image, direction="horizontal"):
    """Directional edge-preserving smoothing using a 1xN or Nx1 averaging kernel."""
    if direction == "horizontal":
        return cv2.blur(image, (5, 1))
    if direction == "vertical":
        return cv2.blur(image, (1, 5))
    raise ValueError("direction must be horizontal or vertical")


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def apply_filter(image, method="median", **kwargs):
    methods = {
        "arithmetic": arithmetic_mean_filter,
        "geometric": geometric_mean_filter,
        "harmonic": harmonic_mean_filter,
        "contra_harmonic": contra_harmonic_mean_filter,
        "median": median_filter,
        "gaussian": gaussian_filter,
        "directional": directional_filter,
        "bilateral": bilateral_filter,
    }
    if method not in methods:
        raise ValueError(f"Unknown filter '{method}'. Choose {list(methods)}.")
    return methods[method](image, **kwargs)
