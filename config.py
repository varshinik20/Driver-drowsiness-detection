"""
config.py

Central configuration module for the AI Driver Monitoring System (AI-DMS).

This file is the SINGLE SOURCE OF TRUTH for every tunable constant used
across the application: camera settings, file paths, detection thresholds,
fatigue scoring weights, alarm behavior, and database configuration.

No other module in this project should hardcode a threshold, path, or
magic number. Every value that might need tuning (e.g. EAR threshold,
fatigue score boundaries) lives here so the whole system can be re-tuned
from one place without touching business logic.

Future ML models (Phase 9) will also read their feature/label configuration
from this file, keeping the architecture consistent.
"""

import os

# ==================================================
# BASE PATHS
# ==================================================

#: Absolute path to the project root directory (where this file lives).
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

#: Directory where the SQLite database file is stored.
DATABASE_DIR: str = os.path.join(BASE_DIR, "database")

#: Full path to the SQLite database file.
DATABASE_PATH: str = os.path.join(DATABASE_DIR, "dms_events.db")

#: Directory where runtime log files are written.
LOGS_DIR: str = os.path.join(BASE_DIR, "logs")

#: Directory where incident screenshots are saved.
SCREENSHOTS_DIR: str = os.path.join(BASE_DIR, "assets", "screenshots")

#: Directory where generated reports (e.g. CSV/PDF summaries) are saved.
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")

#: Directory where trained ML models (Phase 9) are stored.
MODELS_DIR: str = os.path.join(BASE_DIR, "models")

#: Directory containing static assets (sounds, icons, etc.).
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

#: Full path to the alarm sound file.
ALARM_SOUND_PATH: str = os.path.join(ASSETS_DIR, "alert.wav")

# Ensure all required runtime directories exist at import time. This keeps
# every module that depends on these paths safe to use immediately,
# without each module needing its own directory-creation logic.
for _directory in (
    DATABASE_DIR,
    LOGS_DIR,
    SCREENSHOTS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    ASSETS_DIR,
):
    os.makedirs(_directory, exist_ok=True)


# ==================================================
# CAMERA CONFIGURATION
# ==================================================

#: Index of the webcam device to use (0 = default system camera).
CAMERA_INDEX: int = 0

#: Desired capture frame width in pixels.
FRAME_WIDTH: int = 640

#: Desired capture frame height in pixels.
FRAME_HEIGHT: int = 480

#: Target frames-per-second the application aims to process.
TARGET_FPS: int = 30


# ==================================================
# WINDOW / UI CONFIGURATION
# ==================================================

#: Title shown on the application display window.
WINDOW_NAME: str = "AI Driver Monitoring System"

#: Whether to display the FPS counter overlay.
SHOW_FPS: bool = True

#: Whether to display raw facial landmarks on screen (debug aid).
SHOW_LANDMARKS: bool = True

#: Key (as returned by cv2.waitKey) that gracefully exits the application.
EXIT_KEY: str = "q"


# ==================================================
# LOGGING CONFIGURATION
# ==================================================

#: Full path to the rotating application log file.
LOG_FILE_PATH: str = os.path.join(LOGS_DIR, "dms.log")

#: Logging verbosity level. One of: "DEBUG", "INFO", "WARNING", "ERROR".
LOG_LEVEL: str = "INFO"

#: Maximum size (bytes) of a single log file before rotation occurs.
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB

#: Number of rotated backup log files to keep.
LOG_BACKUP_COUNT: int = 3


# ==================================================
# EYE ASPECT RATIO (EAR) CONFIGURATION  -- Phase 3
# ==================================================

#: EAR value below which an eye is considered "closed".
EAR_THRESHOLD: float = 0.21

#: Number of consecutive frames the EAR must stay below threshold
#: before a blink/closure is confirmed (reduces false positives from noise).
EAR_CONSEC_FRAMES: int = 2

#: Eye closure duration (seconds) beyond which the driver is considered
#: to be experiencing prolonged/microsleep-level eye closure.
EYE_CLOSURE_DURATION_THRESHOLD: float = 1.5


# ==================================================
# MOUTH ASPECT RATIO (MAR) CONFIGURATION  -- Phase 4
# ==================================================

#: MAR value above which the mouth is considered "open" (yawning).
MAR_THRESHOLD: float = 0.6

#: Number of consecutive frames MAR must stay above threshold
#: before a yawn is confirmed.
MAR_CONSEC_FRAMES: int = 15


# ==================================================
# HEAD POSE CONFIGURATION  -- Phase 5
# ==================================================

#: Yaw angle (degrees) beyond which the head is considered turned
#: left/right (away from forward-facing).
HEAD_YAW_THRESHOLD: float = 20.0

#: Pitch angle (degrees) beyond which the head is considered tilted
#: up/down (away from forward-facing).
HEAD_PITCH_THRESHOLD: float = 18.0

#: Duration (seconds) the head must remain in a distracted orientation
#: before it counts as a "distraction" event.
HEAD_POSE_DISTRACTION_DURATION: float = 2.0


# ==================================================
# FATIGUE ENGINE CONFIGURATION  -- Phase 6
# ==================================================

FATIGUE_WEIGHT_EYE: float = 1.0
FATIGUE_WEIGHT_YAWN: float = 0.55
FATIGUE_WEIGHT_HEAD_POSE: float = 0.35
FATIGUE_WEIGHT_BLINK_RATE: float = 0.20

#: Fatigue score (0-100) at or above which status is "WARNING".
FATIGUE_SCORE_WARNING: float = 40.0

#: Fatigue score (0-100) at or above which status is "CRITICAL".
FATIGUE_SCORE_CRITICAL: float = 70.0

#: Normal blink rate range (blinks per minute) used as a baseline for
#: scoring abnormal blink frequency.
NORMAL_BLINK_RATE_MIN: int = 10
NORMAL_BLINK_RATE_MAX: int = 25


# ==================================================
# ALARM CONFIGURATION  -- Phase 7
# ==================================================

#: Fatigue status string that triggers the audible alarm.
ALARM_TRIGGER_STATUS: str = "CRITICAL"

#: Minimum seconds between two alarm "start" events, to avoid rapid
#: re-triggering/flickering of the alarm sound.
ALARM_COOLDOWN_SECONDS: float = 3.0


# ==================================================
# DATABASE CONFIGURATION  -- Phase 8
# ==================================================

#: Name of the primary events table in the SQLite database.
DB_TABLE_EVENTS: str = "events"

#: Minimum seconds between two consecutive database log writes,
#: to avoid flooding the database during continuous critical states.
DB_LOG_INTERVAL_SECONDS: float = 1.0


# ==================================================
# MACHINE LEARNING CONFIGURATION  -- Phase 9
# ==================================================

#: Feature columns used to train/predict with the ML fatigue classifier.
ML_FEATURE_COLUMNS = [
    "ear",
    "mar",
    "blink_rate",
    "eye_closure_duration",
    "head_yaw",
    "head_pitch",
]

#: Class labels predicted by the ML fatigue classifier.
ML_CLASS_LABELS = ["NORMAL", "WARNING", "CRITICAL"]

#: Filename of the trained model artifact (joblib-serialized).
ML_MODEL_FILENAME: str = "fatigue_classifier.joblib"

#: Full path to the trained model artifact.
ML_MODEL_PATH: str = os.path.join(MODELS_DIR, ML_MODEL_FILENAME)

#: If True, the FatigueEngine will use the trained ML model (when present)
#: instead of the rule-based scoring logic.
USE_ML_MODEL: bool = False
