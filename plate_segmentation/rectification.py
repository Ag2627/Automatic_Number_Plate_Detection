import cv2
import numpy as np


def order_points(points):
    """
    Arrange four corner points in:
    top-left, top-right, bottom-right, bottom-left order.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(d)],
        points[np.argmax(s)],
        points[np.argmax(d)]
    ], dtype=np.float32)


def four_point_warp(image, points):
    """
    Apply perspective transformation using four plate corners.
    """

    rect = order_points(points)

    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))

    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    if width < 40 or height < 12:
        return None

    if width / height < 1.2:
        return None

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect, destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped, matrix


def rectify_plate(crop):

    if crop is None or crop.size == 0:
        raise ValueError("Empty plate crop")

    h, w = crop.shape[:2]

    identity = np.eye(3).tolist()

    # Small crops should not be aggressively transformed.
    if h < 20 or w < 40:
        return crop.copy(), {
            "method": "identity_small_crop",
            "transform": identity
        }

    if crop.ndim == 3:
        gray = cv2.cvtColor(
            crop, cv2.COLOR_BGR2GRAY
        )
    else:
        gray = crop.copy()

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    edges = cv2.Canny(
        blurred, 50, 150
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:30]:

        area = cv2.contourArea(contour)

        if area < 0.30 * w * h:
            continue

        perimeter = cv2.arcLength(
            contour, True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True
        )

        if len(approx) == 4 and cv2.isContourConvex(approx):

            points = approx.reshape(4, 2)

            rect = cv2.minAreaRect(points)

            rw, rh = rect[1]

            ratio = max(rw, rh) / max(
                1.0, min(rw, rh)
            )

            if 1.5 <= ratio <= 6.5:
                candidates.append(
                    (area, points)
                )

    if candidates:

        _, points = max(
            candidates,
            key=lambda item: item[0]
        )

        result = four_point_warp(
            crop, points
        )

        if result is not None:

            warped, matrix = result

            return warped, {
                "method": "perspective",
                "transform": matrix.tolist()
            }

    # If no reliable perspective is found,
    # try conservative rotation correction.

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, w // 5),
        minLineLength=max(25, w // 4),
        maxLineGap=12
    )

    angles = []

    if lines is not None:

        for line in lines[:, 0, :]:

            x1, y1, x2, y2 = map(
                int, line
            )

            length = np.hypot(
                x2 - x1, y2 - y1
            )

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            if length > 0.25 * w and abs(angle) <= 15:
                angles.append((angle, length))

    if angles:

        angle = np.average(
            [a for a, _ in angles],
            weights=[length for _, length in angles]
        )

        if 1 <= abs(angle) <= 12:

            matrix = cv2.getRotationMatrix2D(
                (w / 2, h / 2),
                angle,
                1.0
            )

            rotated = cv2.warpAffine(
                crop,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            return rotated, {
                "method": "rotation",
                "transform": np.vstack([
                    matrix, [0, 0, 1]
                ]).tolist()
            }

    # Do not distort the plate if correction is uncertain.
    return crop.copy(), {
        "method": "unchanged",
        "transform": identity
    }