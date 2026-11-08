# Archived classical detector package

This package contains the project's earlier non-ML plate-localization approach.
It is archived for reference and is not used by the active YOLO command.

The candidate detector creates edge maps with **Laplacian, Canny, or Sobel**,
closes gaps with morphology, finds contours and connected components, then
ranks plate-like regions using shape and image statistics. Its score is a
heuristic ranking value, not a calibrated confidence.

## Run the comparison

From the project root:

~~~powershell
python archive/classical_plate_detection/experiments/plate_detection_demo.py --image "data\images\image1.jpg"
~~~

The runner prints each method's top candidate coordinates and ranking score. It
saves three box-only images under detection_results/classical/. It does not
write or update CSV annotations.

The implementation is in detector.py and advanced.py. Additional archived
evaluation and feature-experiment code remains available in this package, but
the visual comparison runner uses only Laplacian, Canny, and Sobel.
