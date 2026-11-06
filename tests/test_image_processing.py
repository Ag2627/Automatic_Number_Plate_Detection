import cv2
import numpy as np
import pytest

from image_processing import (
    read_image, save_image, to_grayscale, bgr_to_rgb, bgr_to_hsv, bgr_to_hsi,
    resize_image, crop_image, sample_image, quantize_image, neighbors_4,
    neighbors_8, pixel_distance, translate, rotate, scale_image,
    affine_transform, perspective_transform, histogram, histogram_equalization,
    histogram_matching, contrast_stretch, gamma_correction, clahe_enhancement,
    enhance_image, add_gaussian_noise, add_salt_pepper_noise, motion_blur,
    defocus_blur, arithmetic_mean_filter, geometric_mean_filter,
    harmonic_mean_filter, contra_harmonic_mean_filter, median_filter,
    gaussian_filter, directional_filter, bilateral_filter, apply_filter,
    motion_psf, inverse_filter, wiener_filter, homomorphic_filter,
    constrained_restoration, unconstrained_restoration, mse, psnr, ssim,
    quality_report, detection_rate, preprocess_image
)


@pytest.fixture
def image():
    rng = np.random.default_rng(123)
    img = rng.integers(20, 230, (64, 96, 3), dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (80, 45), (255, 255, 255), 2)
    return img


def test_io_and_color_models(tmp_path, image):
    path = tmp_path / "input.png"
    save_image(path, image)
    loaded = read_image(path)
    assert loaded.shape == image.shape and loaded.dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    assert to_grayscale(image).shape == image.shape[:2]
    assert bgr_to_rgb(image).shape == image.shape
    assert bgr_to_hsv(image).shape == image.shape
    h, s, i = bgr_to_hsi(image)
    assert h.shape == s.shape == i.shape == image.shape[:2]
    assert np.all((s >= 0) & (s <= 1))
    assert np.all((i >= 0) & (i <= 1))


def test_resize_crop_sampling_quantization(image):
    assert resize_image(image, width=48).shape[:2] == (32, 48)
    assert resize_image(image, height=32).shape[:2] == (32, 48)
    assert crop_image(image, 2, 3, 10, 11).shape[:2] == (11, 10)
    assert sample_image(image, 2, 3).shape[:2] == (32, 32)
    assert len(np.unique(quantize_image(image, 4))) <= 4
    with pytest.raises(ValueError):
        quantize_image(image, 1)
    with pytest.raises(ValueError):
        crop_image(image, 1000, 1000, 5, 5)


def test_neighborhoods_and_distances():
    assert len(neighbors_4(2, 2, 5, 5)) == 4
    assert len(neighbors_4(0, 0, 5, 5)) == 2
    assert len(neighbors_8(2, 2, 5, 5)) == 8
    assert len(neighbors_8(0, 0, 5, 5)) == 3
    assert pixel_distance((0, 0), (3, 4), "euclidean") == 5
    assert pixel_distance((0, 0), (3, 4), "cityblock") == 7
    assert pixel_distance((0, 0), (3, 4), "chessboard") == 4
    with pytest.raises(ValueError):
        pixel_distance((0, 0), (1, 1), "bad")


def test_geometric_transforms(image):
    assert translate(image, 5, 4).shape == image.shape
    assert rotate(image, 15).shape == image.shape
    assert scale_image(image, 2).shape[:2] == (128, 192)
    with pytest.raises(ValueError):
        scale_image(image, 0)
    src = [(0, 0), (95, 0), (0, 63)]
    dst = [(5, 5), (90, 5), (5, 58)]
    assert affine_transform(image, src, dst).shape == image.shape
    src4 = [(0, 0), (95, 0), (95, 63), (0, 63)]
    dst4 = [(5, 5), (90, 5), (90, 58), (5, 58)]
    assert perspective_transform(image, src4, dst4, (96, 64)).shape == image.shape
    with pytest.raises(ValueError):
        perspective_transform(image, src4[:3], dst4[:3], (96, 64))


def test_enhancement(image):
    assert len(histogram(to_grayscale(image))[0]) == 256
    assert len(histogram(image)) == 3
    for method in ("equalize", "contrast", "gamma", "clahe"):
        out = enhance_image(image, method)
        assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(gamma_correction(image, 1), image)
    assert histogram_matching(image, image).shape == image.shape
    with pytest.raises(ValueError):
        enhance_image(image, "not-a-method")


def test_noise_models(image):
    g1 = add_gaussian_noise(image, sigma=10, seed=1)
    g2 = add_gaussian_noise(image, sigma=10, seed=1)
    assert np.array_equal(g1, g2)
    sp1 = add_salt_pepper_noise(image, amount=0.1, seed=1)
    sp2 = add_salt_pepper_noise(image, amount=0.1, seed=1)
    assert np.array_equal(sp1, sp2)
    assert motion_blur(image, 9, 20).shape == image.shape
    assert defocus_blur(image, 9).shape == image.shape
    with pytest.raises(ValueError):
        add_salt_pepper_noise(image, amount=1.5)
    with pytest.raises(ValueError):
        motion_blur(image, 8)


def test_filters(image):
    cases = [
        arithmetic_mean_filter(image), geometric_mean_filter(image),
        harmonic_mean_filter(image), contra_harmonic_mean_filter(image),
        median_filter(image), gaussian_filter(image),
        directional_filter(image, "horizontal"),
        directional_filter(image, "vertical"), bilateral_filter(image),
    ]
    for out in cases:
        assert out.shape == image.shape
        assert out.dtype == np.uint8
    with pytest.raises(ValueError):
        directional_filter(image, "diagonal")
    with pytest.raises(ValueError):
        apply_filter(image, "unknown")


def test_restoration(image):
    psf = motion_psf(9, 0)
    assert np.isclose(psf.sum(), 1.0, atol=1e-5)
    gray = to_grayscale(image)
    for out in (
        inverse_filter(gray, psf),
        wiener_filter(gray, psf, 0.02),
        homomorphic_filter(gray),
        constrained_restoration(gray, psf, 0.02),
        unconstrained_restoration(gray, psf),
    ):
        assert out.shape == gray.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()
    with pytest.raises(ValueError):
        wiener_filter(gray, psf, -1)


def test_metrics_and_public_preprocess(image):
    assert mse(image, image) == 0
    assert np.isinf(psnr(image, image))
    assert ssim(image, image) == pytest.approx(1.0)
    report = quality_report(image, image)
    assert report["mse"] == 0 and report["ssim"] == pytest.approx(1.0)
    assert detection_rate(8, 10) == 0.8
    with pytest.raises(ValueError):
        detection_rate(11, 10)
    out = preprocess_image(image)
    assert out.shape == image.shape and out.dtype == np.uint8
