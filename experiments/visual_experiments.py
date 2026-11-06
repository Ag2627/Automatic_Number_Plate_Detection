"""Generate visual experiment sheets for the image-processing module.

Usage:
    python experiments/visual_experiments.py --image path/to/vehicle.jpg
    python experiments/visual_experiments.py
All outputs are saved in visual_results/.
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from image_processing import (
    to_grayscale, bgr_to_hsv, bgr_to_hsi, resize_image, sample_image,
    quantize_image, translate, rotate, scale_image, enhance_image,
    histogram_equalization, histogram_matching, contrast_stretch,
    gamma_correction, clahe_enhancement, add_gaussian_noise,
    add_salt_pepper_noise, motion_blur, defocus_blur,
    arithmetic_mean_filter, geometric_mean_filter, harmonic_mean_filter,
    contra_harmonic_mean_filter, median_filter, gaussian_filter,
    directional_filter, bilateral_filter, inverse_filter, wiener_filter,
    homomorphic_filter, constrained_restoration, motion_psf, mse, psnr, ssim
)


def synthetic_image():
    """Create a synthetic vehicle-like image when no real image is supplied."""
    image = np.full((360, 640, 3), 190, dtype=np.uint8)
    cv2.rectangle(image, (70, 60), (570, 300), (55, 65, 75), -1)
    cv2.rectangle(image, (145, 90), (495, 190), (125, 150, 170), -1)
    cv2.rectangle(image, (245, 220), (395, 275), (245, 245, 245), -1)
    cv2.putText(image, "UP32AB1204", (250, 255), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.circle(image, (160, 292), 35, (20, 20, 20), -1)
    cv2.circle(image, (480, 292), 35, (20, 20, 20), -1)
    return image


def display_image(image):
    if image.ndim == 2:
        return image, "gray"
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None


def save_grid(items, title, path, cols=3, figsize_per_cell=(4.2, 3.2)):
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_cell[0]*cols, figsize_per_cell[1]*rows))
    axes = np.array(axes).reshape(-1)
    for ax, (label, image) in zip(axes, items):
        shown, cmap = display_image(image)
        ax.imshow(shown, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histograms(image, path):
    gray = to_grayscale(image)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale image")
    axes[0].axis("off")
    axes[1].hist(gray.ravel(), bins=256, range=(0, 256))
    axes[1].set_title("Grayscale histogram")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Real vehicle image path")
    parser.add_argument("--output-dir", default="visual_results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = cv2.imread(args.image) if args.image else synthetic_image()
    if original is None:
        raise FileNotFoundError(f"Could not read: {args.image}")

    gray = to_grayscale(original)

    # 1. Fundamentals: color spaces, sampling, quantization, resizing.
    hsv = bgr_to_hsv(original)
    hue, saturation, intensity = bgr_to_hsi(original)
    hsi_preview = np.clip(np.stack([
        hue / (2*np.pi) * 255, saturation * 255, intensity * 255
    ], axis=-1), 0, 255).astype(np.uint8)
    hsi_preview = cv2.cvtColor(hsi_preview, cv2.COLOR_RGB2BGR)
    save_grid([
        ("Original BGR", original), ("Grayscale", gray),
        ("HSV visualization", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)),
        ("HSI-style visualization", hsi_preview),
        ("Sampled: every 2 pixels", cv2.resize(sample_image(original, 2, 2),
                                                (original.shape[1], original.shape[0]))),
        ("Quantized: 8 levels", quantize_image(original, 8)),
        ("Half-size then resized", cv2.resize(resize_image(original, width=max(1, original.shape[1]//2)),
                                              (original.shape[1], original.shape[0]))),
    ], "01 – Image fundamentals", out / "01_fundamentals.png", cols=3)

    # 2. Geometric transformations.
    save_grid([
        ("Original", original),
        ("Translated (+35,+20)", translate(original, 35, 20)),
        ("Rotated +15 degrees", rotate(original, 15)),
        ("Rotated -15 degrees", rotate(original, -15)),
        ("Scaled 0.7 then resized", cv2.resize(scale_image(original, 0.7),
                                                (original.shape[1], original.shape[0]))),
    ], "02 – Geometric transformations", out / "02_transformations.png", cols=3)

    # 3. Histograms and enhancement.
    enhanced = [
        ("Original", original),
        ("Histogram equalization", histogram_equalization(original)),
        ("Contrast stretching", contrast_stretch(original)),
        ("Gamma 0.5 (darker)", gamma_correction(original, 0.5)),
        ("Gamma 2.0 (brighter)", gamma_correction(original, 2.0)),
        ("CLAHE", clahe_enhancement(original)),
    ]
    save_grid(enhanced, "03 – Enhancement comparisons", out / "03_enhancement.png", cols=3)
    save_histograms(original, out / "04_histogram.png")

    # 4. Noise and blur.
    gaussian_noisy = add_gaussian_noise(original, sigma=22, seed=42)
    sp_noisy = add_salt_pepper_noise(original, amount=0.04, seed=42)
    motion = motion_blur(original, kernel_size=15, angle=0)
    defocus = defocus_blur(original, kernel_size=9)
    save_grid([
        ("Original", original),
        ("Gaussian noise (sigma=22)", gaussian_noisy),
        ("Salt & pepper (4%)", sp_noisy),
        ("Motion blur", motion),
        ("Defocus blur", defocus),
    ], "05 – Noise and blur models", out / "05_noise_blur.png", cols=3)

    # 5. Spatial filters: test Gaussian noise and salt-pepper noise separately.
    save_grid([
        ("Gaussian noisy input", gaussian_noisy),
        ("Arithmetic mean", arithmetic_mean_filter(gaussian_noisy, 3)),
        ("Geometric mean", geometric_mean_filter(gaussian_noisy, 3)),
        ("Harmonic mean", harmonic_mean_filter(gaussian_noisy, 3)),
        ("Gaussian filter", gaussian_filter(gaussian_noisy, 5)),
        ("Bilateral filter", bilateral_filter(gaussian_noisy)),
    ], "06 – Filters on Gaussian noise", out / "06_filters_gaussian.png", cols=3)

    save_grid([
        ("Salt-pepper noisy input", sp_noisy),
        ("Median filter", median_filter(sp_noisy, 3)),
        ("Contra-harmonic Q=+1.5", contra_harmonic_mean_filter(sp_noisy, 3, q=1.5)),
        ("Contra-harmonic Q=-1.5", contra_harmonic_mean_filter(sp_noisy, 3, q=-1.5)),
        ("Directional horizontal", directional_filter(gaussian_noisy, "horizontal")),
        ("Directional vertical", directional_filter(gaussian_noisy, "vertical")),
    ], "07 – Filters on impulse noise and directional smoothing",
       out / "07_filters_impulse.png", cols=3)

    # 6. Restoration. The motion PSF is assumed for this controlled comparison.
    psf = motion_psf(15, 0)
    save_grid([
        ("Clean reference", original),
        ("Motion-blurred input", motion),
        ("Inverse filtering", inverse_filter(motion, psf)),
        ("Wiener filtering K=0.02", wiener_filter(motion, psf, 0.02)),
        ("Constrained-style K=0.05", constrained_restoration(motion, psf, 0.05)),
        ("Homomorphic filtering", homomorphic_filter(original)),
    ], "08 – Restoration comparisons", out / "08_restoration.png", cols=3)

    # 7. Metric comparison for controlled degraded/restored samples.
    rows = []
    for name, result in [
        ("motion_blur", motion),
        ("inverse_filter", inverse_filter(motion, psf)),
        ("wiener_filter", wiener_filter(motion, psf, 0.02)),
        ("constrained_filter", constrained_restoration(motion, psf, 0.05)),
    ]:
        rows.append((name, mse(original, result), psnr(original, result), ssim(original, result)))
    report = out / "09_metric_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("Metrics are calculated against the clean reference image.\n")
        f.write("These values are meaningful for this controlled synthetic blur experiment only.\n\n")
        f.write(f"{'Experiment':24s} {'MSE':>12s} {'PSNR':>12s} {'SSIM':>12s}\n")
        for name, m, p, s in rows:
            f.write(f"{name:24s} {m:12.4f} {p:12.4f} {s:12.4f}\n")

    print(f"Visual experiments completed. Open the images in: {out.resolve()}")
    print("Use a real vehicle image for project evidence; the synthetic image is only a smoke test.")
    print(f"Metric report: {report.resolve()}")


if __name__ == "__main__":
    main()
