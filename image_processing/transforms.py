"""Geometric transformations."""
import cv2
import numpy as np


def translate(image, tx, ty):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h))


def rotate(image, angle_degrees, scale=1.0):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    return cv2.warpAffine(image, matrix, (w, h))


def scale_image(image, scale_x=1.0, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Scale factors must be positive.")
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, int(w*scale_x)), max(1, int(h*scale_y))))


def affine_transform(image, source_points, destination_points, output_size=None):
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)
    src = np.float32(source_points[:3])
    dst = np.float32(destination_points[:3])
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, output_size)


def perspective_transform(image, source_points, destination_points, output_size):
    """Apply homography from four source and destination points."""
    src = np.float32(source_points)
    dst = np.float32(destination_points)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Perspective transform needs exactly four 2D points.")
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, tuple(output_size))
