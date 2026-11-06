"""Small experiment runner for Image Processing Module. Uses a supplied image or synthetic pattern."""
import argparse
from pathlib import Path
import sys
import cv2
import pandas as pd

from run_image_processing import synthetic_vehicle_like_image
from image_processing import (
    add_gaussian_noise, add_salt_pepper_noise, apply_filter,
    enhance_image, mse, psnr, ssim
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="experiments/image_processing_metrics.csv")
    args = parser.parse_args()

    image = cv2.imread(args.image) if args.image else synthetic_vehicle_like_image()
    if image is None:
        raise FileNotFoundError(args.image)
    rows = []

    variants = {
        "clahe": enhance_image(image, "clahe"),
        "equalization": enhance_image(image, "equalize"),
        "contrast_stretch": enhance_image(image, "contrast"),
    }
    gaussian = add_gaussian_noise(image, sigma=18, seed=7)
    sp = add_salt_pepper_noise(image, amount=0.03, seed=7)
    variants.update({
        "gaussian_noise_median": apply_filter(gaussian, "median", kernel_size=3),
        "gaussian_noise_bilateral": apply_filter(gaussian, "bilateral"),
        "salt_pepper_median": apply_filter(sp, "median", kernel_size=3),
        "salt_pepper_contra_harmonic": apply_filter(sp, "contra_harmonic", kernel_size=3, q=1.5),
    })

    for name, result in variants.items():
        rows.append({"experiment": name, "mse_vs_original": mse(image, result),
                     "psnr_vs_original": psnr(image, result), "ssim_vs_original": ssim(image, result)})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
