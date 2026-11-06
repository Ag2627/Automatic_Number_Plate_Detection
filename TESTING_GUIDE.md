# Testing Guide – ANPR Image Processing Module

This guide contains automated and manual tests for every implemented function.
Run the automated tests first, then complete the visual and downstream tests.

## A. Automated test execution

From the project root:

```bash
pip install -r requirements.txt
python -m pytest -v
```

For coverage-like confidence without a coverage plugin, make sure the test report
shows all test functions passing. If a test fails, do not ignore it: record the
error, Python version, OS and package versions.

## B. Image fundamentals

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| F01 | Read valid image | Read a known JPG/PNG | Non-empty BGR array, `uint8`, 3 channels |
| F02 | Invalid path | Read a nonexistent path | Clear `FileNotFoundError` |
| F03 | Save image | Save then read saved image | File exists and dimensions are preserved |
| F04 | Grayscale | Convert a color image | 2-D array, same height/width |
| F05 | RGB conversion | Convert BGR to RGB | Red/blue channels are swapped |
| F06 | HSV conversion | Convert BGR to HSV | 3-channel HSV array |
| F07 | HSI conversion | Convert BGR to HSI-style values | H, S, I arrays match image dimensions; S and I in [0,1] |
| F08 | Resize by width | Resize using width only | Requested width; aspect ratio preserved |
| F09 | Resize by height | Resize using height only | Requested height; aspect ratio preserved |
| F10 | Crop | Crop a known rectangle | Exact requested region and dimensions |
| F11 | Out-of-bounds crop | Crop beyond image edge | Clipped crop or clear error for empty crop |
| F12 | Sampling | Sample every 2nd row/3rd column | Output dimensions are ceil(H/2), ceil(W/3) |
| F13 | Quantization | Quantize to 2, 4, 8 and 16 levels | Unique intensity levels do not exceed requested levels |
| F14 | Invalid quantization | Use 1 or 257 levels | `ValueError` |
| F15 | 4-neighbours | Test center and corner pixels | Center has 4; top-left corner has 2 |
| F16 | 8-neighbours | Test center and corner pixels | Center has 8; top-left corner has 3 |
| F17 | Distances | Points (0,0), (3,4) | Euclidean=5, city-block=7, chessboard=4 |

## C. Geometric transformations

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| T01 | Translation | Translate by (20,10) | Object shifts right 20 and down 10 pixels |
| T02 | Rotation | Rotate a non-symmetric test pattern by 90° | Orientation changes; output dimensions remain original |
| T03 | Scaling up | Scale by 2 | Width and height double |
| T04 | Scaling down | Scale by 0.5 | Width and height halve (rounded) |
| T05 | Invalid scale | Use zero or negative scale | `ValueError` |
| T06 | Affine transform | Use 3 non-collinear point pairs | Output has requested dimensions and transformed geometry |
| T07 | Perspective transform | Warp a four-corner quadrilateral to a rectangle | Plate-like quadrilateral becomes approximately front-facing |
| T08 | Invalid homography input | Supply other than 4 point pairs | `ValueError` |

## D. Enhancement and histograms

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| E01 | Histogram grayscale | Plot histogram of grayscale image | 256-bin intensity distribution |
| E02 | Histogram color | Request BGR channel histograms | Three histograms returned |
| E03 | Histogram equalization | Apply to low-contrast image | Contrast generally increases; dimensions preserved |
| E04 | Histogram matching | Match source to a reference image | Source intensity distribution moves toward reference |
| E05 | Contrast stretching | Apply to low-dynamic-range image | Values expand toward [0,255] |
| E06 | Gamma < 1 | Apply gamma=0.5 | Image generally becomes darker under this implementation's convention |
| E07 | Gamma > 1 | Apply gamma=2.0 | Image generally becomes brighter |
| E08 | Gamma=1 | Apply gamma=1 | Output equals input (apart from rounding) |
| E09 | CLAHE grayscale | Apply CLAHE to grayscale | Local contrast changes; dimensions preserved |
| E10 | CLAHE color | Apply CLAHE to BGR image | Color output remains BGR with 3 channels |
| E11 | Unknown method | Pass invalid enhancement method | `ValueError` |
| E12 | Flat image | Enhance constant image | No crash or NaN values |

## E. Noise and blur models

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| N01 | Gaussian noise | Add noise with fixed seed | Output differs from input |
| N02 | Reproducibility | Repeat with same seed | Identical noisy outputs |
| N03 | Gaussian strength | Compare sigma=5 and sigma=30 | Larger sigma has larger average deviation |
| N04 | Salt-pepper noise | Add amount=0.05 | Some pixels become 0 or 255 |
| N05 | Reproducibility | Repeat salt-pepper with same seed | Identical output |
| N06 | Invalid noise amount | Use amount outside [0,1] | `ValueError` |
| N07 | Motion blur | Apply horizontal and angled blur | Edges become directionally blurred |
| N08 | Defocus blur | Apply circular blur | Fine details soften approximately in all directions |
| N09 | Invalid blur kernel | Use even or too-small kernel | `ValueError` |

## F. Spatial filters

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| S01 | Arithmetic mean | Apply to impulse/noisy image | Local averaging smooths noise and edges |
| S02 | Geometric mean | Apply to positive-valued image | Valid uint8 output, no NaNs |
| S03 | Harmonic mean | Apply to positive-valued image | Valid output, no divide-by-zero |
| S04 | Contra-harmonic Q>0 | Test salt-pepper image | Tends to suppress pepper noise |
| S05 | Contra-harmonic Q<0 | Test salt-pepper image | Tends to suppress salt noise |
| S06 | Median | Apply to salt-pepper noise | Impulse noise is reduced |
| S07 | Gaussian | Apply to noisy image | Smooth output |
| S08 | Directional horizontal | Apply horizontal filter | Horizontal smoothing |
| S09 | Directional vertical | Apply vertical filter | Vertical smoothing |
| S10 | Invalid direction | Pass unsupported direction | `ValueError` |
| S11 | Bilateral | Apply to edge-containing image | Smoothing with better edge preservation than a plain mean filter |
| S12 | Shape/dtype | Run every filter | Same height/width and uint8 output |
| S13 | Unknown filter | Pass invalid filter name | `ValueError` |

## G. Restoration

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| R01 | PSF generation | Generate motion PSF | Values sum approximately to 1 |
| R02 | Inverse filtering | Blur clean grayscale image using same PSF, then restore | Executes and returns image-sized output; may amplify noise |
| R03 | Wiener filtering | Restore same degraded image with known PSF | Executes; compare PSNR/SSIM against inverse result |
| R04 | Regularization sweep | Try K=0.001, 0.01, 0.05, 0.1 | Different outputs; select using validation data |
| R05 | Homomorphic filtering | Use image with uneven illumination | Illumination variation may be reduced |
| R06 | Constrained-style restoration | Run with positive regularization | Finite output, no NaN/Inf |
| R07 | Unconstrained restoration | Run without regularization | Finite output; can be unstable/noisy |
| R08 | Invalid regularization | Use negative Wiener K | `ValueError` |
| R09 | Color input | Pass BGR input to restoration | Documented grayscale output for restoration functions |

Important: inverse/Wiener tests must use a known degradation kernel. If the assumed PSF does not match the blur that created the test image, restoration quality may be poor. This is not necessarily a software defect.

## H. Metrics

| ID | Test | Procedure | Expected result |
|---|---|---|---|
| M01 | MSE identity | Compare image with itself | 0 |
| M02 | PSNR identity | Compare image with itself | Infinity |
| M03 | SSIM identity | Compare image with itself | Approximately 1 |
| M04 | Known error | Add a small fixed perturbation | MSE > 0 and PSNR finite |
| M05 | Size mismatch | Compare differently sized images | `ValueError` |
| M06 | Detection rate | successes=8, total=10 | 0.8 |
| M07 | Invalid detection count | successes > total or total=0 | `ValueError` |

## I. End-to-end integration tests

These tests involve the plate detection module and are not fully testable inside this package alone.

1. Choose and freeze a small set of representative vehicle images.
2. Run the plate detector on original images and record TP, FP and FN.
3. Run it on grayscale, CLAHE, contrast-stretched and denoised variants.
4. Keep detector parameters identical for all variants.
5. Compare precision, recall, F1 and plate IoU.
6. Report results per lighting condition and image quality.
7. Keep the same train/validation/test split for every comparison.
8. Do not claim preprocessing improved ANPR based only on visual appearance or MSE/PSNR/SSIM.

## J. Acceptance criteria

The module is ready for integration when:

- All automated tests pass.
- Every public function has been exercised at least once.
- Invalid inputs fail clearly rather than silently.
- Output shape, dtype and channel conventions are documented.
- Visual outputs have been inspected for all enhancement/filter/restoration groups.
- Restoration experiments use known degradation kernels.
- A results table records parameters, image IDs and metrics.
- The downstream detector comparison has been run on a fixed dataset subset.
- Known limitations and failure cases are documented.

## K. Results log template

Record each manual experiment with these fields:

| Date | Image ID | Operation | Parameters | MSE | PSNR | SSIM | Detector result | Notes |
|---|---|---|---|---:|---:|---:|---|---|

Do not compare MSE/PSNR/SSIM between unrelated images. Use the same clean reference for each controlled degradation experiment.
