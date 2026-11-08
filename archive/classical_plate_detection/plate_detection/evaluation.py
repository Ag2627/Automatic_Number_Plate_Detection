"""Bounding-box evaluation for manually annotated plate images."""
import csv
from pathlib import Path
import cv2
import numpy as np
from .detector import detect_plate, _iou


def evaluate_boxes(predictions, ground_truth, iou_threshold=0.5):
    """Greedy one-to-one matching; return precision, recall, F1, TP/FP/FN."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    preds = sorted(predictions, key=lambda x: x.get("score", 0), reverse=True)
    truths = list(ground_truth)
    matched = set()
    matched_ious = []
    tp = fp = 0
    for pred in preds:
        options = [(_iou(pred["bbox"], gt), idx) for idx, gt in enumerate(truths)
                   if idx not in matched]
        best_iou, best_idx = max(options, default=(0.0, -1))
        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched.add(best_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1
    fn = len(truths) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "iou_threshold": iou_threshold,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0}


def read_annotations(csv_path):
    """Read CSV columns image,x,y,width,height; image paths are CSV-relative."""
    path = Path(csv_path)
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = str((path.parent / row["image"]).resolve())
            box = tuple(int(float(row[k])) for k in ("x", "y", "width", "height"))
            rows.setdefault(key, []).append(box)
    return rows


def evaluate_dataset(annotation_csv, output_csv=None, iou_threshold=0.5,
                     edge_method="canny", preprocess=False):
    """Evaluate one-image/one-plate predictions against a CSV annotation set."""
    from image_processing import preprocess_image
    annotations = read_annotations(annotation_csv)
    records = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    iou_sum = 0.0
    for filename, gt_boxes in sorted(annotations.items()):
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        detector_input = preprocess_image(image) if preprocess else image
        crop, box, score = detect_plate(detector_input, edge_method=edge_method)
        pred = [] if box is None else [{"bbox": box, "score": score}]
        metrics = evaluate_boxes(pred, gt_boxes, iou_threshold)
        for key in totals:
            totals[key] += metrics[key]
        iou_sum += metrics["mean_matched_iou"] * metrics["tp"]
        records.append({"image": filename, "prediction": box, "score": score,
                        "ground_truth_count": len(gt_boxes), **metrics})
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    summary = {"precision": p, "recall": r,
               "f1": 2*p*r/(p+r) if p+r else 0.0,
               "tp": tp, "fp": fp, "fn": fn, "images": len(records),
               "iou_threshold": iou_threshold,
               "mean_matched_iou": iou_sum / tp if tp else 0.0}
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else ["image"])
            writer.writeheader()
            writer.writerows(records)
    return summary, records


def evaluate_haar_dataset(annotation_csv, cascade_path, output_csv=None,
                          iou_threshold=0.5):
    """Evaluate all Haar detections against the same CSV ground truth."""
    from .advanced import haar_candidates
    annotations = read_annotations(annotation_csv)
    records = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    iou_sum = 0.0
    for filename, gt_boxes in sorted(annotations.items()):
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        boxes = haar_candidates(image, cascade_path)
        preds = [{"bbox": box, "score": 1.0} for box in boxes]
        metrics = evaluate_boxes(preds, gt_boxes, iou_threshold)
        for key in totals:
            totals[key] += metrics[key]
        iou_sum += metrics["mean_matched_iou"] * metrics["tp"]
        records.append({"image": filename, "predictions": boxes,
                        "ground_truth_count": len(gt_boxes), **metrics})
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    summary = {"precision": p, "recall": r,
               "f1": 2*p*r/(p+r) if p+r else 0.0,
               "tp": tp, "fp": fp, "fn": fn, "images": len(records),
               "iou_threshold": iou_threshold,
               "mean_matched_iou": iou_sum / tp if tp else 0.0}
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else ["image"])
            writer.writeheader()
            writer.writerows(records)
    return summary, records


def evaluate_preprocessing_comparison(annotation_csv, output_csv=None,
                                      iou_threshold=0.5, edge_method="canny"):
    """Run identical settings on original and Person 1-processed images."""
    from image_processing import preprocess_image
    annotations = read_annotations(annotation_csv)
    records = []
    totals = {"original": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0},
              "processed": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}}
    for filename, gt_boxes in sorted(annotations.items()):
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        processed = preprocess_image(image)
        row = {"image": filename, "ground_truth_count": len(gt_boxes)}
        for label, source in (("original", image), ("processed", processed)):
            _, box, score = detect_plate(source, edge_method=edge_method)
            pred = [] if box is None else [{"bbox": box, "score": score}]
            metric = evaluate_boxes(pred, gt_boxes, iou_threshold)
            for key in ("tp", "fp", "fn"):
                totals[label][key] += metric[key]
            totals[label]["iou_sum"] += metric["mean_matched_iou"] * metric["tp"]
            row[f"{label}_bbox"] = box
            row[f"{label}_score"] = score
            row[f"{label}_detected"] = bool(metric["tp"])
            row[f"{label}_iou"] = metric["mean_matched_iou"]
        records.append(row)
    summary = {}
    for label, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        summary[label] = {"precision": p, "recall": r,
                          "f1": 2*p*r/(p+r) if p+r else 0.0,
                          "tp": tp, "fp": fp, "fn": fn,
                          "mean_matched_iou": counts["iou_sum"] / tp if tp else 0.0}
    summary["images"] = len(records)
    summary["iou_threshold"] = iou_threshold
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else ["image"])
            writer.writeheader()
            writer.writerows(records)
    return summary, records


def evaluate_hog_dataset(annotation_csv, model_path, output_csv=None,
                         iou_threshold=0.5):
    """Evaluate HOG-SVM candidate classification on a separate labeled set."""
    from .advanced import detect_hog_candidates
    svm = cv2.ml.SVM_load(str(model_path))
    annotations = read_annotations(annotation_csv)
    records = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    iou_sum = 0.0
    for filename, gt_boxes in sorted(annotations.items()):
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        candidates, _ = detect_hog_candidates(image, svm)
        predictions = [{"bbox": c["bbox"], "score": c["svm_margin"]}
                       for c in candidates]
        metrics = evaluate_boxes(predictions, gt_boxes, iou_threshold)
        for key in totals:
            totals[key] += metrics[key]
        iou_sum += metrics["mean_matched_iou"] * metrics["tp"]
        records.append({"image": filename, "predictions": [c["bbox"] for c in candidates],
                        "ground_truth_count": len(gt_boxes), **metrics})
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    summary = {"precision": p, "recall": r,
               "f1": 2*p*r/(p+r) if p+r else 0.0,
               "tp": tp, "fp": fp, "fn": fn, "images": len(records),
               "iou_threshold": iou_threshold,
               "mean_matched_iou": iou_sum / tp if tp else 0.0}
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else ["image"])
            writer.writeheader()
            writer.writerows(records)
    return summary, records
