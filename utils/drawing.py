"""
utils/drawing.py

All on-screen drawing/overlay logic for the AI-DMS application.

Keeping drawing code isolated here means detector modules stay free of
any cv2.putText / cv2.rectangle calls, enforcing a clean separation
between business logic (detection) and presentation (UI).
"""

from typing import Iterable, Tuple

import cv2
import numpy as np

# Color palette (BGR, since OpenCV uses BGR not RGB).
COLOR_SAFE = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_CRITICAL = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_LANDMARK = (0, 255, 255)

_STATUS_COLORS = {
    "SAFE": COLOR_SAFE,
    "WARNING": COLOR_WARNING,
    "CRITICAL": COLOR_CRITICAL,
}


def draw_landmarks(frame: np.ndarray, points: Iterable[Tuple[int, int]]) -> None:
    """
    Draw small circles at each given landmark pixel coordinate.

    Args:
        frame: The BGR image (modified in place).
        points: Iterable of (x, y) pixel coordinates to draw.
    """
    for x, y in points:
        cv2.circle(frame, (x, y), 1, COLOR_LANDMARK, -1)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = COLOR_TEXT,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """
    Draw a line of text on the frame.

    Args:
        frame: The BGR image (modified in place).
        text: Text string to render.
        position: (x, y) bottom-left corner of the text.
        color: BGR color tuple.
        scale: Font scale factor.
        thickness: Stroke thickness in pixels.
    """
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Draw the current FPS counter in the top-left corner.

    Args:
        frame: The BGR image (modified in place).
        fps: Current frames-per-second value.
    """
    draw_text(frame, f"FPS: {fps:.1f}", (10, 30), COLOR_TEXT)


def draw_metrics_panel(
    frame: np.ndarray,
    ear: float,
    mar: float,
    blink_count: int,
    yawn_count: int,
    head_direction: str,
    fatigue_score: float,
    fatigue_status: str,
) -> None:
    """
    Draw the full real-time metrics overlay panel on the frame.

    Args:
        frame: The BGR image (modified in place).
        ear: Current Eye Aspect Ratio value.
        mar: Current Mouth Aspect Ratio value.
        blink_count: Total blinks counted so far.
        yawn_count: Total yawns counted so far.
        head_direction: Current head pose direction label.
        fatigue_score: Current fatigue score (0-100).
        fatigue_status: Current fatigue status (SAFE/WARNING/CRITICAL).
    """
    status_color = _STATUS_COLORS.get(fatigue_status, COLOR_TEXT)

    lines = [
        f"EAR: {ear:.2f}",
        f"MAR: {mar:.2f}",
        f"Blinks: {blink_count}",
        f"Yawns: {yawn_count}",
        f"Head: {head_direction}",
        f"Fatigue: {fatigue_score:.0f}/100",
    ]

    y_offset = 60
    for line in lines:
        draw_text(frame, line, (10, y_offset))
        y_offset += 25

    draw_text(
        frame,
        f"STATUS: {fatigue_status}",
        (10, y_offset + 10),
        color=status_color,
        scale=0.8,
        thickness=2,
    )

    if fatigue_status == "CRITICAL":
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CRITICAL, 6)
