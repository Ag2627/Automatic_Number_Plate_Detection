"""YOLO-based plate detector API."""
from .yolo_detector import detect_plate_boxes, load_plate_model

__all__ = ["detect_plate_boxes", "load_plate_model"]
