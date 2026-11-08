"""Optional pretrained YOLO license-plate detector.

This path is deliberately separate from the classical detector. It downloads
only the model weights into the Hugging Face cache unless local weights are
provided, and returns plate boxes in image pixel coordinates.
"""
from pathlib import Path

MODEL_REPO = "Babblu2821/alpr-plate-detector"
MODEL_FILENAME = "best.pt"


def load_plate_model(weights=None):
    """Load a local checkpoint or download the public checkpoint once to cache."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "The YOLO option needs optional dependencies. Install them with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if weights:
        model_path = Path(weights).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Install optional dependencies with `python -m pip install -r requirements.txt`."
            ) from exc
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    return YOLO(str(model_path))


def detect_plate_boxes(image, weights=None, confidence=0.25, image_size=640, model=None):
    """Return all plate detections with pixel ``bbox=(x,y,width,height)``.

    The confidence is the detector's score, not a guarantee. Image input must
    be an OpenCV BGR/gray uint8 array. Supply ``model`` to reuse a loaded model
    across images instead of reloading it per call.
    """
    import numpy as np

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 NumPy image")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    detector = model if model is not None else load_plate_model(weights)
    results = detector.predict(source=image, conf=confidence, imgsz=image_size, verbose=False)
    detections = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    height, width = image.shape[:2]
    for coords, score in zip(xyxy, scores):
        x1, y1, x2, y2 = map(int, np.rint(coords))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            detections.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": float(score),
                "crop": image[y1:y2, x1:x2].copy(),
            })
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections

