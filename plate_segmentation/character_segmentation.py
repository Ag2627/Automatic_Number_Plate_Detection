import cv2
import numpy as np


def find_components(binary):

    h, w = binary.shape[:2]

    # Reconnect tiny breaks in character strokes.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 3)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            8
        )
    )

    components = []

    for i in range(1, count):

        x, y, bw, bh, area = map(
            int, stats[i]
        )

        height_ratio = bh / max(h, 1)
        width_ratio = bw / max(w, 1)
        area_ratio = area / max(h * w, 1)
        aspect_ratio = bw / max(bh, 1)

        # Filter out small noise and plate borders.
        valid = (
            0.22 <= height_ratio <= 0.98
            and 0.008 <= width_ratio <= 0.22
            and 0.0015 <= area_ratio <= 0.20
            and 0.045 <= aspect_ratio <= 1.15
            and x > 0
            and y > 0
            and x + bw < w
            and y + bh < h
        )

        if valid:

            components.append({
                "bbox": (x, y, bw, bh),
                "area": area,
                "centroid": tuple(
                    map(float, centroids[i])
                )
            })

    return components


def projection_fallback(binary):

    """
    Use vertical projection to identify character-like
    regions when connected components are insufficient.
    """

    h, w = binary.shape[:2]

    row_sum = np.count_nonzero(
        binary, axis=1
    )

    active_rows = np.flatnonzero(
        row_sum >= max(3, int(0.025 * w))
    )

    if len(active_rows) == 0:
        return []

    # Find horizontal bands.
    bands = []

    start = previous = int(active_rows[0])

    for value in active_rows[1:]:

        value = int(value)

        if value - previous > 3:

            bands.append(
                (start, previous + 1)
            )

            start = value

        previous = value

    bands.append((start, previous + 1))

    bands = [
        (a, b) for a, b in bands
        if b - a >= 0.30 * h
    ]

    if not bands:
        return []

    # Select the strongest character band.
    y1, y2 = max(
        bands,
        key=lambda band:
            (band[1] - band[0])
            * np.max(row_sum[band[0]:band[1]])
    )

    region = binary[y1:y2, :]

    col_sum = np.count_nonzero(
        region, axis=0
    )

    active_cols = np.flatnonzero(
        col_sum >= max(1, int(0.025 * (y2-y1)))
    )

    if len(active_cols) == 0:
        return []

    runs = []

    start = previous = int(active_cols[0])

    for value in active_cols[1:]:

        value = int(value)

        if value - previous > 3:

            runs.append(
                (start, previous + 1)
            )

            start = value

        previous = value

    runs.append((start, previous + 1))

    components = []

    for x1, x2 in runs:

        bw = x2 - x1

        if bw < max(2, int(0.008 * w)):
            continue

        if bw > 0.22 * w:
            continue

        roi = region[:, x1:x2]

        active_y = np.flatnonzero(
            np.count_nonzero(roi, axis=1) > 0
        )

        if len(active_y) == 0:
            continue

        yy1 = int(active_y[0])
        yy2 = int(active_y[-1]) + 1

        bh = yy2 - yy1

        if bh < 0.30 * h:
            continue

        if bw / max(bh, 1) > 1.15:
            continue

        components.append({
            "bbox": (
                x1,
                y1 + yy1,
                bw,
                bh
            ),
            "area": int(
                np.count_nonzero(
                    roi[yy1:yy2]
                )
            ),
            "centroid": (
                (x1+x2)/2,
                (y1+yy1+y2)/2
            )
        })

    return components


def group_and_sort_characters(components):

    """
    Supports one-line and two-line plate layouts.
    Sort characters left to right within each row.
    """

    if not components:
        return []

    rows = []

    for item in sorted(
        components,
        key=lambda c: c["centroid"][1]
    ):

        cy = item["centroid"][1]

        placed = False

        for row in rows:

            center = np.mean([
                c["centroid"][1]
                for c in row
            ])

            typical_height = np.median([
                c["bbox"][3]
                for c in row
            ])

            if abs(cy-center) <= max(
                typical_height * 0.48,
                5
            ):

                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    rows.sort(
        key=lambda row: np.mean([
            c["centroid"][1]
            for c in row
        ])
    )

    for row in rows:

        row.sort(
            key=lambda c: c["bbox"][0]
        )

    return rows


def segment_characters(thresholds):

    candidates = []

    for name, binary in thresholds.items():

        components = find_components(binary)

        method = "connected_components"

        # Use projection fallback when too few
        # components are found.
        if len(components) < 5:

            projected = projection_fallback(
                binary
            )

            if len(projected) > len(components):

                components = projected

                method = "vertical_projection"

        # Avoid claiming success for implausible results.
        if 3 <= len(components) <= 14:

            heights = [
                c["bbox"][3]
                for c in components
            ]

            consistency = (
                np.std(heights)
                / max(np.mean(heights), 1)
            )

            count_score = (
                1.0 - abs(len(components)-8)*0.12
            )

            score = (
                count_score
                - 0.45 * consistency
                - 0.025 * max(
                    0, len(components)-10
                )
            )

            candidates.append((
                score,
                name,
                binary,
                components,
                method
            ))

    if not candidates:

        return {
            "success": False,
            "reason": "No plausible character regions found",
            "threshold_name": None,
            "binary": next(iter(thresholds.values())),
            "characters": [],
            "rows": []
        }

    score, name, binary, components, method = max(
        candidates,
        key=lambda candidate: candidate[0]
    )

    rows = group_and_sort_characters(
        components
    )

    ordered = [
        character
        for row in rows
        for character in row
    ]

    return {
        "success": True,
        "reason": None,
        "threshold_name": name,
        "binary": binary,
        "characters": ordered,
        "rows": rows,
        "candidate_score": float(score),
        "segmentation_method": method
    }