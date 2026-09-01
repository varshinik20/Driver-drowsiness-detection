"""
detector/yawn_detector.py

Yawn detection via Mouth Aspect Ratio (MAR).

Consumes landmarks produced by detector/face_mesh.py and applies a MAR
calculation analogous to EAR, but for the mouth, to detect sustained
mouth-opening events characteristic of yawning.
"""

from dataclasses import dataclass
from typing import List, Tuple

import config
from detector.face_mesh import MOUTH_INDICES
from utils.geometry import euclidean_distance
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MouthState:
    """
    Result of mouth-state analysis for a single frame.

    Attributes:
        mar: Current Mouth Aspect Ratio.
        is_yawning: Whether a yawn is currently in progress (confirmed
            after `consec_frames` consecutive open-mouth frames).
        yawn_count: Cumulative number of completed yawns.
    """

    mar: float
    is_yawning: bool
    yawn_count: int


class YawnDetector:
    """
    Tracks mouth openness over time using the Mouth Aspect Ratio (MAR)
    metric to detect and count yawning events.
    """

    def __init__(
        self,
        mar_threshold: float = config.MAR_THRESHOLD,
        consec_frames: int = config.MAR_CONSEC_FRAMES,
    ) -> None:
        """
        Initialize the yawn detector.

        Args:
            mar_threshold: MAR value above which the mouth is "open".
            consec_frames: Consecutive open-mouth frames required to
                confirm a yawn (debounces talking/single-frame noise).
        """
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames

        self._open_frame_counter: int = 0
        self._yawn_count: int = 0
        self._yawn_in_progress: bool = False
        self._yawn_already_counted: bool = False

    @staticmethod
    def _mouth_aspect_ratio(mouth_points: List[Tuple[int, int]]) -> float:
        """
        Compute the Mouth Aspect Ratio from 8 mouth landmark points,
        ordered as: [left corner, top-outer-left, top-center, top-outer-right,
        right corner, bottom-outer-right, bottom-center, bottom-outer-left].

        Args:
            mouth_points: Eight (x, y) landmark points for the mouth.

        Returns:
            The computed MAR value. Returns 0.0 if points are insufficient.
        """
        if len(mouth_points) != 8:
            return 0.0

        left, top1, top2, top3, right, bottom1, bottom2, bottom3 = mouth_points

        vertical_1 = euclidean_distance(top1, bottom3)
        vertical_2 = euclidean_distance(top2, bottom2)
        vertical_3 = euclidean_distance(top3, bottom1)
        horizontal = euclidean_distance(left, right)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)

    def update(self, landmarks: List[Tuple[int, int]]) -> MouthState:
        """
        Process one frame's worth of face landmarks and update internal
        yawn tracking state.

        Args:
            landmarks: Full list of (x, y) face landmarks from
                FaceMeshDetector.process().

        Returns:
            A MouthState describing the current MAR, yawn-in-progress
            status, and cumulative yawn count.
        """
        mouth_points = [landmarks[i] for i in MOUTH_INDICES]
        mar = self._mouth_aspect_ratio(mouth_points)

        is_open_this_frame = mar > self.mar_threshold

        if is_open_this_frame:
            self._open_frame_counter += 1

            if self._open_frame_counter >= self.consec_frames:
                self._yawn_in_progress = True

                if not self._yawn_already_counted:
                    self._yawn_count += 1
                    self._yawn_already_counted = True
                    logger.debug("Yawn detected. Total yawns: %s", self._yawn_count)
        else:
            self._open_frame_counter = 0
            self._yawn_in_progress = False
            self._yawn_already_counted = False

        return MouthState(
            mar=mar,
            is_yawning=self._yawn_in_progress,
            yawn_count=self._yawn_count,
        )

    def reset(self) -> None:
        """Reset all internal tracking state (e.g. on a new session)."""
        self._open_frame_counter = 0
        self._yawn_count = 0
        self._yawn_in_progress = False
        self._yawn_already_counted = False
