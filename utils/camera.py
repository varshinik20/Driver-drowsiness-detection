"""
utils/camera.py

Webcam capture wrapper for the AI-DMS application.

Encapsulates all OpenCV VideoCapture logic behind a clean class interface
so the rest of the application never touches cv2.VideoCapture directly.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class Camera:
    """
    Wraps a single webcam device and exposes a simple frame-reading API.

    Attributes:
        index: Camera device index (from config.CAMERA_INDEX by default).
        width: Requested capture frame width.
        height: Requested capture frame height.
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        """
        Initialize the camera wrapper (does not open the device yet).

        Args:
            index: Index of the webcam device to open.
            width: Desired capture frame width in pixels.
            height: Desired capture frame height in pixels.
        """
        self.index = index
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """
        Open the webcam device and configure resolution.

        Raises:
            RuntimeError: If the camera device cannot be opened.
        """
        self._capture = cv2.VideoCapture(self.index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._capture.isOpened():
            logger.error("Failed to open camera at index %s", self.index)
            raise RuntimeError(f"Could not open camera at index {self.index}")

        logger.info(
            "Camera opened (index=%s, requested=%sx%s)",
            self.index,
            self.width,
            self.height,
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns:
            A tuple (success, frame). `success` is False if the frame
            could not be read (e.g. camera disconnected).
        """
        if self._capture is None:
            raise RuntimeError("Camera.open() must be called before read_frame().")

        success, frame = self._capture.read()
        if not success:
            logger.warning("Failed to read frame from camera.")
            return False, None

        # Mirror the frame horizontally so it behaves like a selfie camera,
        # which feels natural for a driver-facing monitor.
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        """Release the underlying camera device and free resources."""
        if self._capture is not None:
            self._capture.release()
            logger.info("Camera released.")
            self._capture = None

    def __enter__(self) -> "Camera":
        """Support `with Camera() as cam:` usage."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the camera is released when exiting a `with` block."""
        self.release()
