# YOLO plate detection

The active plate-detection path uses the pretrained YOLO plate detector. The
previous edge/morphology implementation is preserved under
`archive/classical_plate_detection/` and is no longer imported by the active
package.

## Setup

```powershell
$env:PYYAML_FORCE_LIBYAML = "0"  # needed for PyYAML on Python 3.15 without MSVC
python -m pip install -r requirements.txt
Remove-Item Env:PYYAML_FORCE_LIBYAML
```

The environment variable lets PyYAML use its pure-Python implementation when
installing on Python 3.15 without Microsoft C++ Build Tools.

## Detect plates, print coordinates, and update the CSV

```powershell
python -m plate_detection --image "data\images\image1.jpg"
```

The command runs the YOLO detector in this folder, prints each plate's `x`,
`y`, `width`, and `height` in image pixels plus confidence, and saves an image
with only green boxes under `detection_results/yolo/`. It also records the detected
boxes in `data/annotations/plate_boxes.csv`. If you run it again for the same
image, its existing CSV rows are replaced with the latest detections.

Options include `--confidence 0.4`, `--weights path\to\best.pt`, and
`--output path\to\boxed.jpg`. The first run downloads the public `best.pt`
weights into the Hugging Face cache.

The CSV entries produced by this command are YOLO predictions. For an
independent accuracy evaluation, manually verify the boxes before treating them
as ground-truth annotations.

The checkpoint is from `Babblu2821/alpr-plate-detector`. Review that model card's
license, training data, and limitations, and validate predictions on images
representative of your use case. Detection gives plate boxes; OCR is not part of
this active path.


