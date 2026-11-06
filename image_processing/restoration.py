"""Frequency-domain restoration demonstrations for known/assumed degradations."""
import cv2
import numpy as np


def motion_psf(size=15, angle=0):
    if size < 3:
        raise ValueError("size must be >= 3.")
    psf = np.zeros((size, size), np.float32)
    psf[size//2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1)
    psf = cv2.warpAffine(psf, matrix, (size, size))
    return psf / max(psf.sum(), 1e-8)


def _psf_to_otf(psf, shape):
    padded = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -kh//2, axis=0)
    padded = np.roll(padded, -kw//2, axis=1)
    return np.fft.fft2(padded)


def _frequency_restore(gray, psf, regularization=0.0, epsilon=1e-8):
    x = gray.astype(np.float32) / 255.0
    H = _psf_to_otf(psf.astype(np.float32), x.shape)
    G = np.fft.fft2(x)
    denominator = np.abs(H) ** 2 + regularization
    F = (np.conj(H) * G) / np.maximum(denominator, epsilon)
    restored = np.real(np.fft.ifft2(F))
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def inverse_filter(image, psf=None):
    """Unregularized inverse filtering; highly noise-sensitive."""
    if psf is None:
        psf = motion_psf()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return _frequency_restore(gray, psf, regularization=0.0)


def wiener_filter(image, psf=None, noise_to_signal=0.01):
    """Wiener-like regularized inverse filter; K is noise-to-signal power ratio."""
    if noise_to_signal < 0:
        raise ValueError("noise_to_signal must be >= 0.")
    if psf is None:
        psf = motion_psf()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return _frequency_restore(gray, psf, regularization=noise_to_signal)


def homomorphic_filter(image, gamma_low=0.7, gamma_high=1.5,
                       cutoff=30, sharpness=2.0):
    """Reduce slowly varying illumination and enhance reflectance details."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    x = np.log1p(gray.astype(np.float32))
    rows, cols = x.shape
    u = np.arange(rows) - rows / 2
    v = np.arange(cols) - cols / 2
    U, V = np.meshgrid(v, u)
    D2 = U**2 + V**2
    H = (gamma_high - gamma_low) * (1 - np.exp(-sharpness * D2 / (cutoff**2))) + gamma_low
    spectrum = np.fft.fftshift(np.fft.fft2(x))
    result = np.real(np.fft.ifft2(np.fft.ifftshift(H * spectrum)))
    result = np.expm1(result)
    lo, hi = np.percentile(result, [1, 99])
    if hi <= lo:
        return gray.copy()
    return np.clip((result-lo) * 255.0 / (hi-lo), 0, 255).astype(np.uint8)


def constrained_restoration(image, psf=None, regularization=0.02):
    """Regularized (constrained-style) restoration using a positive penalty."""
    if psf is None:
        psf = motion_psf()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return _frequency_restore(gray, psf, regularization=regularization)


def unconstrained_restoration(image, psf=None):
    """Unregularized inverse solution, shown for comparison; can amplify noise."""
    return inverse_filter(image, psf=psf)


def restore_image(image, method="wiener", **kwargs):
    methods = {
        "inverse": inverse_filter,
        "wiener": wiener_filter,
        "homomorphic": homomorphic_filter,
        "constrained": constrained_restoration,
        "unconstrained": unconstrained_restoration,
    }
    if method not in methods:
        raise ValueError(f"Unknown restoration method '{method}'. Choose {list(methods)}.")
    return methods[method](image, **kwargs)
