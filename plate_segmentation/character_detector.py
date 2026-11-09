from pathlib import Path
import numpy as np
from ultralytics import YOLO

def load_character_model(weights_path):
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Character detector weights not found: {weights_path}")
    return YOLO(str(weights_path))

def detect_characters(image, model, confidence=0.25, image_size=640):
    if image is None or image.size == 0:
        raise ValueError("Empty plate image")
    if model is None:
        raise ValueError("A trained character detection model is required")
    results = model.predict(source=image, conf=confidence, imgsz=image_size, verbose=False)
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    height, width = image.shape[:2]
    detections = []
    for coords, score in zip(xyxy, scores):
        x1, y1, x2, y2 = np.rint(coords).astype(int)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "bbox": (x1, y1, x2-x1, y2-y1),
            "confidence": float(score),
            "centroid": ((x1+x2)/2.0, (y1+y2)/2.0)
        })
    if not detections:
        return []
    detections.sort(key=lambda item: item["centroid"][1])
    rows = []
    for item in detections:
        cy = item["centroid"][1]
        candidates = []
        for row in rows:
            center = float(np.median([obj["centroid"][1] for obj in row]))
            typical_h = float(np.median([obj["bbox"][3] for obj in row]))
            distance = abs(cy-center)
            if distance <= max(0.5*typical_h, 5):
                candidates.append((distance, row))
        if candidates:
            min(candidates, key=lambda pair: pair[0])[1].append(item)
        else:
            rows.append([item])
    rows.sort(key=lambda row: np.median([obj["centroid"][1] for obj in row]))
    ordered = []
    for row in rows:
        row.sort(key=lambda item: item["bbox"][0])
        ordered.extend(row)
    return ordered
