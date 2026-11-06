# ANPR Image Processing Module

This is a standalone image-processing module for an Indian ANPR system. It is independent of the plate detector and
can be developed/tested on any team member's laptop.

## Features

- Image I/O, BGR/RGB/grayscale/HSV/HSI-style conversions
- Sampling, quantization, resizing, cropping, neighborhood and distance helpers
- Translation, rotation, scaling, affine and perspective transformations
- Histograms, histogram equalization, histogram matching, contrast stretching,
  gamma correction and CLAHE
- Gaussian and salt-and-pepper noise; motion and defocus blur
- Arithmetic, geometric, harmonic, contra-harmonic, median, Gaussian,
  directional and bilateral filters
- Inverse, Wiener and homomorphic filtering
- MSE, PSNR, SSIM and downstream detection-rate comparison helper
- Command-line demo that creates a synthetic sample if no image is supplied

## 1. Setup

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Run the demo

With your own vehicle image:
```bash
python run_demo.py --image path/to/vehicle.jpg --output-dir sample_outputs
```

Without an image (creates a synthetic test image):
```bash
python run_demo.py --output-dir sample_outputs
```

The demo saves enhancement, noise, filtering, restoration and histogram
comparison images under `sample_outputs/`.

## 3. Import the module from plate detection module

Run from the repository root:

```python
import cv2
from image_processing import preprocess_image, enhance_image

image = cv2.imread("vehicle.jpg")  # OpenCV uses BGR
processed = preprocess_image(image)
clahe_image = enhance_image(image, method="clahe")

# plate detection module can choose original or processed input and compare detector results.
```

### Stable handoff contract

- Input: OpenCV NumPy image, normally `uint8`, BGR, shape `(H, W, 3)`.
- `preprocess_image(image)` returns a BGR `uint8` image.
- `enhance_image(image, method)` returns an image in the same channel format
  as its input (for color images, enhancement is applied to luminance).
- `restore_image(image, method, **kwargs)` returns a restored image.
- Functions do not modify the caller's input array.

plate detection module should evaluate detection on both the original and selected processed
images. Enhancement is optional: some operations can amplify noise or remove
useful character edges.

## 4. Project structure

```text
ANPR_ImageProcessing_Preprocessing/
├── image_processing/
│   ├── __init__.py
│   ├── fundamentals.py
│   ├── transforms.py
│   ├── enhancement.py
│   ├── noise.py
│   ├── filters.py
│   ├── restoration.py
│   └── metrics.py
├── experiments/
│   └── run_experiments.py
├── tests/
│   └── test_image_processing.py
├── sample_outputs/
├── run_demo.py
├── requirements.txt
└── README.md
```

## 5. Important notes / limitations

- HSI is not a native OpenCV color space. `bgr_to_hsi()` provides an
  HSI-style conversion for demonstration and returns H, S, I floating-point
  arrays.
- Inverse and Wiener filtering require an assumed/known point-spread function
  (PSF). Results are meaningful mainly for controlled degradation examples.
- Homomorphic filtering is an illumination-correction technique, not a
  universal restoration method.
- Constrained and unconstrained restoration are demonstrated through
  regularized and unregularized frequency-domain inverse solutions. They
  depend strongly on the assumed blur kernel and regularization parameter.
- SSIM requires `scikit-image`.
- MSE/PSNR/SSIM require a clean reference image. For real images without a
  reference, compare downstream plate-detection performance instead.
- This module does not detect number plates. It prepares images and provides
  experiments for plate detection module.

## 6. Run tests

```bash
pytest -q
```

## 7. Suggested experiment protocol

1. Keep a fixed set of vehicle images.
2. Run detection on original images to establish a baseline.
3. Apply one enhancement/filter at a time.
4. Run the same detector with the same parameters on each variant.
5. Record successful plate detections / total images, precision, recall and F1.
6. Do not conclude that an image looks better means ANPR performance improved.
7. Keep train/validation/test images fixed and avoid tuning on the test set.


See `TESTING_GUIDE.md` for the full manual test matrix, expected outcomes, integration tests and acceptance criteria.


## Visual experiments (recommended for project verification)

Run with a real vehicle image:

```bash
python experiments/visual_experiments.py --image data/vehicle.jpg
```

Or run a smoke test without a real image:

```bash
python experiments/visual_experiments.py
```

The script creates a `visual_results/` folder containing comparison sheets for:
1. Color models, grayscale, sampling, quantization and resizing
2. Geometric transformations
3. Histogram equalization, contrast stretching, gamma correction and CLAHE
4. Noise and blur models
5. Spatial filters on Gaussian noise
6. Spatial filters on salt-and-pepper noise
7. Restoration comparisons
8. A metric report for controlled restoration experiments

For project sign-off, repeat these experiments on a representative set of real
vehicle images and inspect the output sheets. A single synthetic image is not
sufficient evidence of ANPR performance.
