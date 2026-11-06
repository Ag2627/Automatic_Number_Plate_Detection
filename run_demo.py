"""Run a complete Image Processing Module demonstration and save outputs."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processing import (
    read_image, save_image, enhance_image, add_gaussian_noise,
    add_salt_pepper_noise, motion_blur, defocus_blur, apply_filter,
    restore_image, histogram, quality_report
)


def synthetic_vehicle_like_image():
    """Simple synthetic test pattern; replace with real vehicle images for experiments."""
    img = np.full((360, 640, 3), 185, dtype=np.uint8)
    cv2.rectangle(img, (75, 65), (565, 295), (55, 65, 75), -1)
    cv2.rectangle(img, (145, 95), (495, 190), (115, 145, 165), -1)
    cv2.rectangle(img, (250, 220), (390, 270), (245, 245, 245), -1)
    cv2.putText(img, "UP32AB1204", (255, 253), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.circle(img, (160, 285), 34, (20, 20, 20), -1)
    cv2.circle(img, (480, 285), 34, (20, 20, 20), -1)
    return img


def save_comparison(images, titles, output_path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        ax.imshow(display, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to vehicle image (BGR loaded by OpenCV)")
    parser.add_argument("--output-dir", default="sample_outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = read_image(args.image) if args.image else synthetic_vehicle_like_image()
    save_image(out / "00_original.png", image)

    enhanced = {
        "equalized": enhance_image(image, "equalize"),
        "contrast": enhance_image(image, "contrast"),
        "gamma": enhance_image(image, "gamma", gamma=1.3),
        "clahe": enhance_image(image, "clahe"),
    }
    for name, result in enhanced.items():
        save_image(out / f"enhancement_{name}.png", result)
    save_comparison([image, *enhanced.values()],
                    ["Original", *enhanced.keys()], out / "enhancement_comparison.png")

    noisy_gaussian = add_gaussian_noise(image, sigma=18, seed=42)
    noisy_sp = add_salt_pepper_noise(image, amount=0.03, seed=42)
    motion = motion_blur(image, kernel_size=15, angle=0)
    defocus = defocus_blur(image, kernel_size=9)
    for name, result in [("gaussian_noise", noisy_gaussian), ("salt_pepper_noise", noisy_sp),
                         ("motion_blur", motion), ("defocus_blur", defocus)]:
        save_image(out / f"{name}.png", result)

    filtered = {
        "mean": apply_filter(noisy_gaussian, "arithmetic", kernel_size=3),
        "median": apply_filter(noisy_sp, "median", kernel_size=3),
        "gaussian": apply_filter(noisy_gaussian, "gaussian", kernel_size=5),
        "bilateral": apply_filter(noisy_gaussian, "bilateral"),
        "geometric": apply_filter(noisy_gaussian, "geometric", kernel_size=3),
        "harmonic": apply_filter(noisy_gaussian, "harmonic", kernel_size=3),
        "contra_harmonic": apply_filter(noisy_sp, "contra_harmonic", kernel_size=3, q=1.5),
    }
    for name, result in filtered.items():
        save_image(out / f"filter_{name}.png", result)

    restored = {
        "wiener": restore_image(motion, "wiener", noise_to_signal=0.02),
        "inverse": restore_image(motion, "inverse"),
        "homomorphic": restore_image(image, "homomorphic"),
        "constrained": restore_image(motion, "constrained", regularization=0.03),
    }
    for name, result in restored.items():
        save_image(out / f"restoration_{name}.png", result)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = histogram(gray)[0]
    plt.figure(figsize=(8, 4))
    plt.plot(hist)
    plt.title("Grayscale histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(out / "histogram.png", dpi=150)
    plt.close()

    print(f"Demo complete. Outputs saved to: {out.resolve()}")
    print("For meaningful restoration metrics, compare a degraded/restored image against its clean reference.")
    print("Example metrics:", quality_report(image, enhanced["clahe"]))


if __name__ == "__main__":
    main()
