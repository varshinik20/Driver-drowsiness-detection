"""
utils/geometry.py

Pure geometric helper functions shared across detector modules.

Contains no OpenCV/MediaPipe-specific logic beyond simple vector math,
keeping it easily unit-testable in isolation.
"""

from typing import Sequence, Tuple

import numpy as np


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """
    Compute the Euclidean distance between two 2D (or 3D) points.

    Args:
        point_a: First point as an (x, y[, z]) sequence.
        point_b: Second point as an (x, y[, z]) sequence.

    Returns:
        The straight-line distance between the two points.
    """
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def midpoint(point_a: Sequence[float], point_b: Sequence[float]) -> Tuple[float, float]:
    """
    Compute the midpoint between two 2D points.

    Args:
        point_a: First point as an (x, y) sequence.
        point_b: Second point as an (x, y) sequence.

    Returns:
        The (x, y) midpoint coordinate.
    """
    return (
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Clamp a value between a minimum and maximum bound.

    Args:
        value: The value to clamp.
        minimum: Lower bound.
        maximum: Upper bound.

    Returns:
        The clamped value.
    """
    return max(minimum, min(value, maximum))


def normalize_landmark_to_pixel(
    landmark_x: float,
    landmark_y: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """
    Convert a MediaPipe normalized landmark coordinate (0-1 range) into
    pixel coordinates for a given frame size.

    Args:
        landmark_x: Normalized x coordinate (0.0 - 1.0).
        landmark_y: Normalized y coordinate (0.0 - 1.0).
        frame_width: Width of the target frame in pixels.
        frame_height: Height of the target frame in pixels.

    Returns:
        (x, y) pixel coordinates as integers.
    """
    return int(landmark_x * frame_width), int(landmark_y * frame_height)
