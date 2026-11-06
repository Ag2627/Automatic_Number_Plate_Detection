"""Image Processing Module preprocessing module public API."""
from .fundamentals import (
    read_image, save_image, to_grayscale, bgr_to_rgb, bgr_to_hsv, bgr_to_hsi,
    resize_image, crop_image, sample_image, quantize_image,
    neighbors_4, neighbors_8, pixel_distance,
)
from .transforms import translate, rotate, scale_image, affine_transform, perspective_transform
from .enhancement import (
    histogram, histogram_equalization, histogram_matching, contrast_stretch,
    gamma_correction, clahe_enhancement, enhance_image,
)
from .noise import add_gaussian_noise, add_salt_pepper_noise, motion_blur, defocus_blur
from .filters import (
    arithmetic_mean_filter, geometric_mean_filter, harmonic_mean_filter,
    contra_harmonic_mean_filter, median_filter, gaussian_filter,
    directional_filter, bilateral_filter, apply_filter,
)
from .restoration import (
    motion_psf, inverse_filter, wiener_filter, homomorphic_filter,
    constrained_restoration, unconstrained_restoration, restore_image,
)
from .metrics import mse, psnr, ssim, quality_report, detection_rate


def preprocess_image(image):
    """Conservative default preprocessing: grayscale denoise + CLAHE, return BGR."""
    import cv2
    if image is None:
        raise ValueError("image cannot be None")
    gray = to_grayscale(image)
    denoised = cv2.bilateralFilter(gray, 7, 45, 45)
    enhanced = clahe_enhancement(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


__all__ = [name for name in globals() if not name.startswith("_")]
