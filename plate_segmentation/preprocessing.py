import cv2
import numpy as np


def prepare_plate(image, target_width=640):

    if image is None or image.size == 0:
        raise ValueError("Empty plate image")

    h, w = image.shape[:2]

    # Resize while maintaining aspect ratio.
    scale = min(1.0, target_width / w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
        if scale < 1 else cv2.INTER_CUBIC
    )

    # Convert to grayscale.
    if resized.ndim == 3:

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

    else:
        gray = resized.copy()

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Reduce noise while preserving edges.
    denoised = cv2.bilateralFilter(
        enhanced,
        7,
        35,
        35
    )

    # Global thresholding.
    otsu_value, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Local thresholding.
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Both polarities are important.
    thresholds = {

        "otsu_dark_on_light": otsu,

        "otsu_light_on_dark":
            cv2.bitwise_not(otsu),

        "adaptive_dark_on_light":
            adaptive,

        "adaptive_light_on_dark":
            cv2.bitwise_not(adaptive)
    }

    return {
        "resized": resized,
        "gray": gray,
        "enhanced": denoised,
        "thresholds": thresholds,
        "scale_x": new_w / w,
        "scale_y": new_h / h,
        "otsu_threshold": float(otsu_value)
    }