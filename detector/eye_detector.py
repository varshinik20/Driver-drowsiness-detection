"""
detector/eye_detector.py

Eye state detection: Eye Aspect Ratio (EAR), blink counting, and eye
closure duration tracking.

Consumes landmarks produced by detector/face_mesh.py and applies the
classic EAR algorithm (Soukupova & Cech, 2016) to determine whether the
driver's eyes are open or closed.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import config
from detector.face_mesh import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from utils.geometry import euclidean_distance
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EyeState:
    """
    Result of eye-state analysis for a single frame.

    Attributes:
        ear: Averaged Eye Aspect Ratio of both eyes.
        eyes_closed: Whether the eyes are currently classified as closed.
        blink_count: Cumulative number of completed blinks.
        eye_closure_duration: Seconds the eyes have been continuously closed
            (0.0 if currently open).
    """

    ear: float
    eyes_closed: bool
    blink_count: int
    eye_closure_duration: float


class EyeDetector:
    """
    Tracks eye openness over time using the Eye Aspect Ratio (EAR) metric.

    Maintains internal state across frames (consecutive closed-frame
    counter, blink count, and closure start time) to provide blink
    detection and closure-duration measurement.
    """

    def __init__(
        self,
        ear_threshold: float = config.EAR_THRESHOLD,
        consec_frames: int = config.EAR_CONSEC_FRAMES,
    ) -> None:
        """
        Initialize the eye detector.

        Args:
            ear_threshold: EAR value below which an eye is "closed".
            consec_frames: Consecutive closed frames required to confirm
                a blink (debounces single-frame noise).
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        self._closed_frame_counter: int = 0
        self._blink_count: int = 0
        self._eyes_currently_closed: bool = False
        self._closure_start_time: float = 0.0

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        """
        Compute the Eye Aspect Ratio for a single eye given 6 landmark
        points, ordered as: [p1 (left corner), p2, p3, p4 (right corner),
        p5, p6] following the standard EAR convention.

        Args:
            eye_points: Six (x, y) landmark points for one eye.

        Returns:
            The computed EAR value. Returns 0.0 if points are insufficient.
        """
        if len(eye_points) != 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def update(self, landmarks: List[Tuple[int, int]]) -> EyeState:
        """
        Process one frame's worth of face landmarks and update internal
        blink/closure tracking state.

        Args:
            landmarks: Full list of 468(+iris) (x, y) face landmarks, as
                produced by FaceMeshDetector.process().

        Returns:
            An EyeState describing the current EAR, closure status, blink
            count, and closure duration.
        """
        left_eye_points = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye_points = [landmarks[i] for i in RIGHT_EYE_INDICES]

        left_ear = self._eye_aspect_ratio(left_eye_points)
        right_ear = self._eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        is_closed_this_frame = avg_ear < self.ear_threshold

        if is_closed_this_frame:
            self._closed_frame_counter += 1

            if (
                self._closed_frame_counter >= self.consec_frames
                and not self._eyes_currently_closed
            ):
                self._eyes_currently_closed = True
                self._closure_start_time = time.time()
        else:
            if self._eyes_currently_closed:
                # Eyes just reopened after being closed -> count as a blink.
                self._blink_count += 1
                logger.debug("Blink detected. Total blinks: %s", self._blink_count)

            self._closed_frame_counter = 0
            self._eyes_currently_closed = False

        closure_duration = 0.0
        if self._eyes_currently_closed:
            closure_duration = time.time() - self._closure_start_time

        return EyeState(
            ear=avg_ear,
            eyes_closed=self._eyes_currently_closed,
            blink_count=self._blink_count,
            eye_closure_duration=closure_duration,
        )

    def reset(self) -> None:
        """Reset all internal tracking state (e.g. on a new session)."""
        self._closed_frame_counter = 0
        self._blink_count = 0
        self._eyes_currently_closed = False
        self._closure_start_time = 0.0
