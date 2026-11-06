"""Synthetic degradation/noise models for controlled experiments."""
import cv2
import numpy as np


def add_gaussian_noise(image, mean=0.0, sigma=15.0, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5, seed=None):
    if not 0 <= amount <= 1 or not 0 <= salt_vs_pepper <= 1:
        raise ValueError("amount and salt_vs_pepper must be in [0,1].")
    rng = np.random.default_rng(seed)
    out = image.copy()
    total = int(amount * image.shape[0] * image.shape[1])
    coords = (rng.integers(0, image.shape[0], total),
              rng.integers(0, image.shape[1], total))
    split = int(total * salt_vs_pepper)
    out[coords[0][:split], coords[1][:split]] = 255
    out[coords[0][split:], coords[1][split:]] = 0
    return out


def motion_blur(image, kernel_size=15, angle=0):
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and >= 3.")
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-8)
    return cv2.filter2D(image, -1, kernel)


def defocus_blur(image, kernel_size=9):
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and >= 3.")
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    cv2.circle(kernel, (kernel_size//2, kernel_size//2), kernel_size//2, 1, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)
