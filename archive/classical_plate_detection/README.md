# Archived classical plate detector

This folder preserves the earlier classical-computer-vision plate detector.
The active project detector is YOLO under the project-root plate_detection/
package. The archived detector is separate and does not write to the YOLO
annotations CSV.

## Three-filter demo

The demo compares Laplacian, Canny, and Sobel edge maps. For each method, it
prints the top plate candidate's pixel coordinates and heuristic ranking score,
then saves an image with only the candidate box drawn. The score ranks
candidates; it is not a confidence probability.

Run from the project root:

~~~powershell
python archive/classical_plate_detection/experiments/plate_detection_demo.py --image "data\images\image1.jpg"
~~~

The three boxed images are saved under detection_results/classical/. To use a
different output folder, add --output-dir "path\to\folder". The demo does not
create or update any CSV.

## Contents

- plate_detection/: archived classical detector and supporting code
- experiments/plate_detection_demo.py: the three-filter visual comparison
- tests/: tests retained for the archived code
- README.md: this overview
