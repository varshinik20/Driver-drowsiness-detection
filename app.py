"""
app.py

Main application entry point for the AI Driver Monitoring System.

Wires together the camera, face mesh engine, eye/yawn/head-pose
detectors, fatigue engine, alarm system, database, and drawing overlays
into a single real-time monitoring loop.

Run with:
    python app.py
"""

import time
from datetime import datetime

import cv2

import config
from database.database import EventDatabase, EventRecord
from detector.eye_detector import EyeDetector
from detector.face_mesh import FaceMeshDetector
from detector.fatigue_engine import FatigueEngine
from detector.head_pose import HeadPoseEstimator
from detector.yawn_detector import YawnDetector
from utils.alarm import AlarmSystem
from utils.camera import Camera
from utils.drawing import draw_fps, draw_landmarks, draw_metrics_panel
from utils.logger import get_logger

logger = get_logger(__name__)


class DriverMonitoringApp:
    """
    Top-level application orchestrator.

    Owns all subsystems (camera, detectors, fatigue engine, alarm,
    database) and runs the main real-time processing loop. This class
    contains no detection math itself -- it only coordinates calls to
    the specialized modules, keeping business logic and orchestration
    cleanly separated.
    """

    def __init__(self) -> None:
        """Initialize all subsystems required by the application."""
        self.camera = Camera()
        self.face_mesh = FaceMeshDetector()
        self.eye_detector = EyeDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = HeadPoseEstimator(
            frame_width=config.FRAME_WIDTH, frame_height=config.FRAME_HEIGHT
        )
        self.fatigue_engine = FatigueEngine()
        self.alarm_system = AlarmSystem()
        self.database = EventDatabase()

        self._last_db_log_time: float = 0.0
        self._fps_frame_count: int = 0
        self._fps_start_time: float = time.time()
        self._current_fps: float = 0.0

    def _compute_fps(self) -> float:
        """
        Update and return a smoothed FPS estimate based on a 1-second
        rolling window of processed frames.

        Returns:
            Current frames-per-second value.
        """
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start_time

        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_start_time = time.time()

        return self._current_fps

    def _save_screenshot(self, frame) -> str:
        """
        Save a screenshot of the current frame to disk, named with a
        timestamp for uniqueness.

        Args:
            frame: The current BGR video frame to save.

        Returns:
            The filesystem path where the screenshot was saved.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"incident_{timestamp_str}.jpg"
        path = f"{config.SCREENSHOTS_DIR}/{filename}"
        cv2.imwrite(path, frame)
        logger.info("Screenshot saved: %s", path)
        return path

    def _handle_fatigue_status(self, frame, fatigue_score, fatigue_status, ear, mar, blink_count) -> None:
        """
        React to the current fatigue status: trigger/stop the alarm and
        log critical/warning events to the database (rate-limited).

        Args:
            frame: Current BGR video frame (used for screenshots).
            fatigue_score: Current fatigue score (0-100).
            fatigue_status: Current status label.
            ear: Current Eye Aspect Ratio.
            mar: Current Mouth Aspect Ratio.
            blink_count: Cumulative blink count.
        """
        if fatigue_status == config.ALARM_TRIGGER_STATUS:
            self.alarm_system.trigger()
        else:
            self.alarm_system.stop()

        now = time.time()
        should_log = (
            fatigue_status in ("WARNING", "CRITICAL")
            and now - self._last_db_log_time >= config.DB_LOG_INTERVAL_SECONDS
        )

        if should_log:
            screenshot_path = (
                self._save_screenshot(frame) if fatigue_status == "CRITICAL" else None
            )
            event = EventRecord(
                timestamp=EventDatabase.now(),
                ear=ear,
                mar=mar,
                blink_count=blink_count,
                fatigue_score=fatigue_score,
                alert_type=fatigue_status,
                screenshot_path=screenshot_path,
            )
            self.database.insert_event(event)
            self._last_db_log_time = now

    def run(self) -> None:
        """
        Run the main real-time monitoring loop until the user exits
        (by pressing the configured exit key or closing the window).
        """
        self.database.initialize()

        try:
            self.camera.open()
        except RuntimeError as exc:
            logger.error("Cannot start application: %s", exc)
            return

        logger.info("Driver Monitoring System started. Press '%s' to quit.", config.EXIT_KEY)

        try:
            while True:
                success, frame = self.camera.read_frame()
                if not success:
                    continue

                mesh_result = self.face_mesh.process(frame)

                if mesh_result.face_detected:
                    eye_state = self.eye_detector.update(mesh_result.landmarks)
                    mouth_state = self.yawn_detector.update(mesh_result.landmarks)
                    head_pose = self.head_pose_estimator.update(mesh_result.landmarks)

                    fatigue_result = self.fatigue_engine.evaluate(
                        ear=eye_state.ear,
                        mar=mouth_state.mar,
                        blink_count=eye_state.blink_count,
                        eye_closure_duration=eye_state.eye_closure_duration,
                        head_yaw=head_pose.yaw,
                        head_pitch=head_pose.pitch,
                        head_distraction_duration=head_pose.distraction_duration,
                    )

                    if config.SHOW_LANDMARKS:
                        draw_landmarks(frame, mesh_result.landmarks)

                    draw_metrics_panel(
                        frame,
                        ear=eye_state.ear,
                        mar=mouth_state.mar,
                        blink_count=eye_state.blink_count,
                        yawn_count=mouth_state.yawn_count,
                        head_direction=head_pose.direction,
                        fatigue_score=fatigue_result.score,
                        fatigue_status=fatigue_result.status,
                    )

                    self._handle_fatigue_status(
                        frame,
                        fatigue_result.score,
                        fatigue_result.status,
                        eye_state.ear,
                        mouth_state.mar,
                        eye_state.blink_count,
                    )
                else:
                    self.alarm_system.stop()
                    cv2.putText(
                        frame,
                        "NO FACE DETECTED",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                if config.SHOW_FPS:
                    draw_fps(frame, self._compute_fps())

                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(config.EXIT_KEY):
                    logger.info("Exit key pressed. Shutting down.")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Shutting down.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Release all resources cleanly on application exit."""
        self.alarm_system.stop()
        self.camera.release()
        self.face_mesh.close()
        self.database.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def main() -> None:
    """Application entry point."""
    app = DriverMonitoringApp()
    app.run()


if __name__ == "__main__":
    main()
