# Detection evaluation data

The six sample vehicle images are stored in `images/`. Keep only images that
the team is permitted to use and share, and record each source dataset's
license and attribution details with the project.

## Annotation CSV

Copy `annotations/plate_boxes_template.csv` and add one row per ground-truth
plate. Coordinates are integer pixels in the original image: top-left `x`,
top-left `y`, `width`, and `height`. Image paths are relative to the CSV file.

```csv
image,x,y,width,height
../images/vehicle_001.jpg,210,142,138,39
```

Use distinct vehicle/image groups for train, validation, and test. Do not tune
detector thresholds on the held-out test set. For one plate per image, keep one
row per image; the evaluator also accepts multiple ground-truth rows for the
same image.

## Required report notes

Record the dataset name/version, license, number of images, annotation method,
split rule, IoU threshold, detector configuration, precision, recall, F1,
mean matched IoU, false positives, missed plates, and representative failure
cases. Compare original and preprocessed inputs with identical detector
settings.
