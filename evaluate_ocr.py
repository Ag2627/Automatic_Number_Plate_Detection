
import csv
from pathlib import Path


# ---------------- CONFIGURATION ----------------

GROUND_TRUTH_FILE = Path("ground_truth.csv")

# Change this path if your CSV is stored elsewhere.
PREDICTION_FILE = Path("detection_results/batch_ocr/ocr_results.csv")
OUTPUT_DIR = Path("evaluation_results")

# ------------------------------------------------


def edit_distance_details(reference, prediction):
    """
    Calculate Levenshtein distance and identify
    substitutions, deletions and insertions.
    """

    m = len(reference)
    n = len(prediction)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):

            if reference[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    # Backtrack to count individual error types.
    i, j = m, n

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        # Matching characters or substitution.
        if (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1]
            + (reference[i - 1] != prediction[j - 1])
        ):

            if reference[i - 1] != prediction[j - 1]:
                substitutions += 1

            i -= 1
            j -= 1

        # Deletion from prediction.
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:

            deletions += 1
            i -= 1

        # Insertion into prediction.
        else:

            insertions += 1
            j -= 1

    return (
        dp[m][n],
        substitutions,
        deletions,
        insertions
    )


def load_ground_truth():

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {GROUND_TRUTH_FILE.resolve()}"
        )

    ground_truth = {}

    with open(
        GROUND_TRUTH_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        # Remove extra spaces from column headers.
        if reader.fieldnames is None:
            raise ValueError("Ground truth CSV is empty.")

        reader.fieldnames = [
            name.strip().lower() for name in reader.fieldnames
        ]

        if "image" not in reader.fieldnames:
            raise ValueError(
                f"Missing 'image' column. Found columns: {reader.fieldnames}"
            )

        if "ground_truth" not in reader.fieldnames:
            raise ValueError(
                f"Missing 'ground_truth' column. Found columns: {reader.fieldnames}"
            )

        for row in reader:

            image = row["image"].strip().lower()
            plate_number = row["ground_truth"].strip().upper()

            ground_truth[image] = plate_number

    return ground_truth
    
def load_predictions():

    predictions = {}

    with open(
        PREDICTION_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            image = row["image"].strip().lower()
            plate = row["plate_number"].strip().upper()

            predictions[image] = plate

    return predictions


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(GROUND_TRUTH_FILE)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(PREDICTION_FILE)

    ground_truth = load_ground_truth()
    predictions = load_predictions()

    results = []

    total_correct = 0
    total_errors = 0
    total_characters = 0

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    for image, actual in ground_truth.items():

        predicted = predictions.get(image, "")

        distance, subs, dels, ins = (
            edit_distance_details(actual, predicted)
        )

        correct = actual == predicted

        if correct:
            total_correct += 1

        total_errors += distance
        total_characters += len(actual)

        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins

        cer = distance / len(actual) if actual else 0

        results.append({
            "image": image,
            "ground_truth": actual,
            "predicted": predicted,
            "exact_match": "Yes" if correct else "No",
            "edit_distance": distance,
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "CER": round(cer, 4)
        })

        print("\nImage:", image)
        print("Actual:", actual)
        print("Predicted:", predicted)
        print("Exact match:", correct)
        print("Edit distance:", distance)
        print("CER:", f"{cer:.2%}")

    # Overall evaluation.
    total_images = len(ground_truth)

    accuracy = (
        total_correct / total_images * 100
        if total_images else 0
    )

    overall_cer = (
        total_errors / total_characters
        if total_characters else 0
    )

    # Save per-image results.
    results_file = OUTPUT_DIR / "evaluation_results.csv"

    fieldnames = [
        "image",
        "ground_truth",
        "predicted",
        "exact_match",
        "edit_distance",
        "substitutions",
        "deletions",
        "insertions",
        "CER"
    ]

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # Save overall summary.
    summary_file = OUTPUT_DIR / "summary.txt"

    summary = f"""
ANPR OCR EVALUATION REPORT
==========================

Total images evaluated: {total_images}

Correct predictions: {total_correct}
Incorrect predictions: {total_images - total_correct}

Exact-match accuracy: {accuracy:.2f}%

Total character errors: {total_errors}
Total ground-truth characters: {total_characters}

Character Error Rate (CER): {overall_cer:.4f}
Character Error Rate (%): {overall_cer * 100:.2f}%

Substitutions: {total_substitutions}
Deletions: {total_deletions}
Insertions: {total_insertions}

Per-image results:
{results_file}
"""

    summary_file.write_text(summary, encoding="utf-8")

    print("\n" + "=" * 50)
    print(summary)
    print("Evaluation completed!")
    print("Results folder:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
