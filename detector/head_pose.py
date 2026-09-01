"""
detector/head_pose.py

Head pose estimation using OpenCV's solvePnP against a generic 3D face
model, mapped to the corresponding MediaPipe Face Mesh landmark indices.

Classifies the driver's head orientation into a discrete direction label
(Forward, Left, Right, Up, Down) and tracks sustained distraction time.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

import config
from detector.face_mesh import (
    CHIN_INDEX,
    LEFT_EYE_CORNER_INDEX,
    LEFT_MOUTH_CORNER_INDEX,
    NOSE_TIP_INDEX,
    RIGHT_EYE_CORNER_INDEX,
    RIGHT_MOUTH_CORNER_INDEX,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Generic 3D model points (in an arbitrary unit, mm-like scale) representing
# an "average" human face, used as the reference for solvePnP. Order must
# match the 2D landmark order used in `_get_2d_points` below.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class HeadPoseResult:
    """
    Result of head pose estimation for a single frame.

    Attributes:
        yaw: Rotation around the vertical axis (left/right turn), degrees.
        pitch: Rotation around the horizontal axis (up/down tilt), degrees.
        roll: Rotation around the depth axis (head tilt sideways), degrees.
        direction: Discrete classification label, e.g. "FORWARD", "LEFT",
            "RIGHT", "UP", "DOWN".
        distraction_duration: Seconds the head has been continuously in a
            non-forward orientation (0.0 if currently forward-facing).
    """

    yaw: float
    pitch: float
    roll: float
    direction: str
    distraction_duration: float


class HeadPoseEstimator:
    """
    Estimates 3D head orientation (yaw/pitch/roll) from 2D facial
    landmarks using OpenCV's solvePnP, and classifies it into a
    discrete direction label for fatigue/distraction scoring.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        yaw_threshold: float = config.HEAD_YAW_THRESHOLD,
        pitch_threshold: float = config.HEAD_PITCH_THRESHOLD,
    ) -> None:
        """
        Initialize the head pose estimator.

        Args:
            frame_width: Width of the video frame in pixels (for the
                camera intrinsic matrix approximation).
            frame_height: Height of the video frame in pixels.
            yaw_threshold: Degrees beyond which yaw counts as left/right.
            pitch_threshold: Degrees beyond which pitch counts as up/down.
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        self._camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        # Assume no lens distortion (reasonable approximation for webcams).
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self._distraction_start_time: float = 0.0
        self._is_distracted: bool = False

    @staticmethod
    def _get_2d_points(landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extract the six 2D landmark points needed for solvePnP, in the
        same order as `_MODEL_POINTS_3D`.

        Args:
            landmarks: Full list of (x, y) face landmarks.

        Returns:
            A (6, 2) numpy array of 2D points.
        """
        indices = [
            NOSE_TIP_INDEX,
            CHIN_INDEX,
            LEFT_EYE_CORNER_INDEX,
            RIGHT_EYE_CORNER_INDEX,
            LEFT_MOUTH_CORNER_INDEX,
            RIGHT_MOUTH_CORNER_INDEX,
        ]
        return np.array([landmarks[i] for i in indices], dtype=np.float64)

    def _classify_direction(self, yaw: float, pitch: float) -> str:
        """
        Map continuous yaw/pitch angles to a discrete direction label.

        Args:
            yaw: Yaw angle in degrees.
            pitch: Pitch angle in degrees.

        Returns:
            One of "FORWARD", "LEFT", "RIGHT", "UP", "DOWN".
        """
        if yaw > self.yaw_threshold:
            return "RIGHT"
        if yaw < -self.yaw_threshold:
            return "LEFT"
        if pitch > self.pitch_threshold:
            return "DOWN"
        if pitch < -self.pitch_threshold:
            return "UP"
        return "FORWARD"

    def update(self, landmarks: List[Tuple[int, int]]) -> HeadPoseResult:
        """
        Estimate head pose for the current frame and update distraction
        duration tracking.

        Args:
            landmarks: Full list of (x, y) face landmarks from
                FaceMeshDetector.process().

        Returns:
            A HeadPoseResult with yaw, pitch, roll, direction label, and
            current distraction duration.
        """
        image_points = self._get_2d_points(landmarks)

        success, rotation_vector, _translation_vector = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.warning("solvePnP failed to converge for this frame.")
            return HeadPoseResult(0.0, 0.0, 0.0, "FORWARD", 0.0)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)

        direction = self._classify_direction(yaw, pitch)

        if direction != "FORWARD":
            if not self._is_distracted:
                self._is_distracted = True
                self._distraction_start_time = time.time()
        else:
            self._is_distracted = False

        distraction_duration = (
            time.time() - self._distraction_start_time if self._is_distracted else 0.0
        )

        return HeadPoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            direction=direction,
            distraction_duration=distraction_duration,
        )

    @staticmethod
    def _rotation_matrix_to_euler_angles(
        rotation_matrix: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Convert a 3x3 rotation matrix into (pitch, yaw, roll) Euler angles
        in degrees.

        Args:
            rotation_matrix: 3x3 rotation matrix from cv2.Rodrigues.

        Returns:
            Tuple of (pitch, yaw, roll) angles in degrees.
        """
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )
        singular = sy < 1e-6

        if not singular:
            x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
            z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
            z_angle = 0.0

        pitch = np.degrees(x_angle)
        yaw = np.degrees(y_angle)
        roll = np.degrees(z_angle)

        # Normalize pitch into a more intuitive range for "looking down/up".
        pitch = (pitch - 180) if pitch > 90 else (pitch + 180 if pitch < -90 else pitch)

        return float(pitch), float(yaw), float(roll)
