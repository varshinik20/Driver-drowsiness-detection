"""
detector/face_mesh.py

Central face-tracking engine built on MediaPipe Face Mesh.

This module is the AI backbone of the entire system: every other detector
(eyes, mouth, head pose) consumes the 468 landmarks produced here. It
exposes landmark index groups (eyes, mouth, nose, iris) so downstream
modules never have to hardcode MediaPipe's raw index numbers themselves.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ==================================================
# MediaPipe Face Mesh landmark index groups
# ==================================================
# These indices are fixed by MediaPipe's 468-point face mesh topology.
# Centralizing them here means eye_detector.py, yawn_detector.py, and
# head_pose.py never need to know MediaPipe's internals directly.

LEFT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]

MOUTH_INDICES: List[int] = [78, 81, 13, 311, 308, 402, 14, 178]

NOSE_TIP_INDEX: int = 1
CHIN_INDEX: int = 152
LEFT_EYE_CORNER_INDEX: int = 33
RIGHT_EYE_CORNER_INDEX: int = 263
LEFT_MOUTH_CORNER_INDEX: int = 61
RIGHT_MOUTH_CORNER_INDEX: int = 291

LEFT_IRIS_INDICES: List[int] = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES: List[int] = [473, 474, 475, 476, 477]


@dataclass
class FaceMeshResult:
    """
    Container for a single frame's face-mesh detection result.

    Attributes:
        face_detected: Whether a face was found in the frame.
        landmarks: List of (x, y) pixel coordinates for all 468 landmarks
            (plus iris points if refine_landmarks is enabled). Empty if
            no face was detected.
        bounding_box: (x_min, y_min, x_max, y_max) pixel bounding box of
            the detected face, or None if no face was detected.
    """

    face_detected: bool
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    bounding_box: Optional[Tuple[int, int, int, int]] = None

    def get_points(self, indices: List[int]) -> List[Tuple[int, int]]:
        """
        Retrieve pixel coordinates for a specific subset of landmark indices.

        Args:
            indices: List of landmark indices to retrieve.

        Returns:
            List of (x, y) pixel coordinates corresponding to `indices`.
            Returns an empty list if no face was detected.
        """
        if not self.face_detected:
            return []
        return [self.landmarks[i] for i in indices if i < len(self.landmarks)]


class FaceMeshDetector:
    """
    Wraps MediaPipe's FaceMesh solution to provide per-frame facial
    landmark detection with a simplified, application-specific interface.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize the MediaPipe Face Mesh model.

        Args:
            max_num_faces: Maximum number of faces to track (1 for a
                single-driver monitoring scenario).
            refine_landmarks: Whether to enable iris landmark refinement.
            min_detection_confidence: Minimum confidence for initial
                face detection to be considered successful.
            min_tracking_confidence: Minimum confidence for landmark
                tracking to be considered successful between frames.
        """
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info("FaceMeshDetector initialized (refine_landmarks=%s)", refine_landmarks)

    def process(self, frame: np.ndarray) -> FaceMeshResult:
        """
        Run face mesh detection on a single BGR frame.

        Args:
            frame: Input frame in BGR color format (as read by OpenCV).

        Returns:
            A FaceMeshResult describing whether a face was found and,
            if so, its landmarks and bounding box.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return FaceMeshResult(face_detected=False)

        frame_height, frame_width = frame.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]

        points: List[Tuple[int, int]] = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in face_landmarks.landmark
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bounding_box = (min(xs), min(ys), max(xs), max(ys))

        return FaceMeshResult(
            face_detected=True, landmarks=points, bounding_box=bounding_box
        )

    def close(self) -> None:
        """Release MediaPipe FaceMesh resources."""
        self._face_mesh.close()
        logger.info("FaceMeshDetector closed.")
