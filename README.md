# Indian ANPR Detection and Image Processing

This repository combines a pretrained YOLO number-plate detector with an image
processing module. The plate detector localizes plates and returns bounding
boxes and confidence scores. The `plate_ocr` command adds character
segmentation and EasyOCR recognition. Model training is not included.

## Setup

Create and activate a virtual environment, then install the dependencies from
the single requirements file:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

On Python installations where PyYAML cannot build its optional native
extension, install with its pure-Python implementation:

~~~powershell
$env:PYYAML_FORCE_LIBYAML = "0"
python -m pip install -r requirements.txt
Remove-Item Env:PYYAML_FORCE_LIBYAML
~~~

## YOLO plate detection

From the repository root, run:

~~~powershell
python -m plate_detection --image "data\images\image1.jpg"
~~~

The command prints each detected plate's pixel coordinates and confidence,
saves a box-only image under detection_results/yolo/, and updates
data/annotations/plate_boxes.csv. Running it again for the same image replaces
that image's earlier CSV entries. The first run downloads the pretrained
checkpoint into the Hugging Face cache.

The CSV contains model predictions. Manually verify boxes before using them as
ground-truth annotations for accuracy evaluation.

## Character OCR

The `plate_ocr` package runs plate detection, character segmentation, then
EasyOCR recognition on each character crop. Run it from the repository root:

~~~bash
python3 -m plate_ocr --image data/images/image1.jpg
~~~

To recognize crops already produced by `plate_segmentation`, use:

~~~bash
python3 -m plate_ocr --metadata detection_results/segmentation/image1_plate_01.json
~~~

Both commands print JSON. Image mode returns an `image` and a `plates` list,
because an image may contain several plates. Metadata mode returns one plate
result directly. Each result includes `predicted_plate`, ordered per-character
predictions and OCR confidences, and `needs_review`. Missing, ambiguous, or
low-confidence characters trigger review. `--ocr-confidence` changes the
review threshold (default `0.5`). For Python callers:

~~~python
from plate_ocr import recognize_plate
result = recognize_plate("detection_results/segmentation/image1_plate_01.json")
~~~

EasyOCR may download its English recognition model on first use. OCR confidence
is separate from the plate and character detector confidence scores. The
predicted plate is not automatically validated against a registration format.

## Archived classical comparison

A non-ML detector is preserved under archive/classical_plate_detection/. The
standalone comparison runs Laplacian, Canny, and Sobel edge methods:

~~~powershell
python archive/classical_plate_detection/experiments/plate_detection_demo.py --image "data\images\image1.jpg"
~~~

It prints each method's top candidate coordinates and ranking score, then saves
box-only images under detection_results/classical/. It does not update the
YOLO annotation CSV. The ranking score is a heuristic, not a confidence
probability.

## Image processing

The image_processing/ module contains image conversion, geometric transforms,
enhancement, noise, filtering, restoration, and image quality metrics. It can
be used independently of plate detection.

Run the image-processing demo on an image:

~~~powershell
python run_demo.py --image "path\to\vehicle.jpg" --output-dir sample_outputs
~~~

Run the visual experiment suite:

~~~powershell
python experiments/visual_experiments.py --image "path\to\vehicle.jpg"
~~~

## Project layout

- plate_detection/: active YOLO detector and command-line runner
- plate_segmentation/: plate straightening and character crop extraction
- plate_ocr/: EasyOCR recognition of those character crops
- archive/classical_plate_detection/: preserved classical detector, demo, and tests
- image_processing/: preprocessing and image analysis functions
- data/images/: input images
- data/annotations/plate_boxes.csv: predicted plate coordinates
- detection_results/yolo/ and detection_results/classical/: generated boxed images
- experiments/: image-processing experiment scripts

Generated detection results and virtual-environment files are ignored by Git.
Input images and annotations under data/ are not ignored.
