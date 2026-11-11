
import re
from paddleocr import PaddleOCR


class PlateOCR:

    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    @staticmethod
    def normalize_text(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def arrange_reading_order(entries):
        """
        Group OCR detections into text rows.
        Read rows from top to bottom and text within
        each row from left to right.
        """

        if not entries:
            return []

        entries.sort(key=lambda item: item["cy"])

        rows = []

        for entry in entries:

            matched_row = None

            for row in rows:

                average_y = sum(
                    item["cy"] for item in row
                ) / len(row)

                average_height = sum(
                    item["h"] for item in row
                ) / len(row)

                tolerance = max(
                    average_height * 0.6,
                    5
                )

                if abs(entry["cy"] - average_y) <= tolerance:
                    matched_row = row
                    break

            if matched_row is None:
                rows.append([entry])
            else:
                matched_row.append(entry)

        # Sort rows top to bottom.
        rows.sort(
            key=lambda row: sum(
                item["cy"] for item in row
            ) / len(row)
        )

        ordered = []

        # Sort text within each row left to right.
        for row in rows:
            row.sort(key=lambda item: item["x"])
            ordered.extend(row)

        return ordered

    def recognize(self, plate_image):

        if plate_image is None or plate_image.size == 0:
            raise ValueError("Empty plate image.")

        results = self.ocr.predict(plate_image)

        if not results:
            return {
                "text": "",
                "confidence": 0.0,
                "raw_text": ""
            }

        result = results[0]

        texts = result.get("rec_texts", []) or []
        scores = result.get("rec_scores", []) or []
        boxes = result.get("rec_boxes", [])

        entries = []

        for i, text in enumerate(texts):

            if not text:
                continue

            score = (
                float(scores[i])
                if i < len(scores)
                else 0.0
            )

            if i < len(boxes):

                x1, y1, x2, y2 = map(
                    float, boxes[i]
                )

                entries.append({
                    "text": str(text),
                    "score": score,
                    "x": x1,
                    "cy": (y1 + y2) / 2,
                    "h": max(y2 - y1, 1)
                })

            else:
                entries.append({
                    "text": str(text),
                    "score": score,
                    "x": i,
                    "cy": 0,
                    "h": 1
                })

        # Correct reading order for single-line and
        # multi-line number plates.
        entries = self.arrange_reading_order(entries)

        raw_text = " ".join(
            item["text"] for item in entries
        )

        # Exclude a separately detected IND marking.
        plate_parts = [
            item["text"]
            for item in entries
            if self.normalize_text(item["text"]) != "IND"
        ]

        combined = self.normalize_text(
            "".join(plate_parts)
        )

        confidence = (
            sum(item["score"] for item in entries)
            / len(entries)
            if entries else 0.0
        )

        return {
            "text": combined,
            "confidence": confidence,
            "raw_text": raw_text
        }
