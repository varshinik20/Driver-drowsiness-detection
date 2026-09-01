"""
detector/fatigue_engine.py

Fatigue scoring engine: combines EAR, MAR, blink rate, eye closure
duration, and head pose into a single 0-100 fatigue score and a discrete
status (SAFE / WARNING / CRITICAL).

Designed so that Phase 9 can swap the rule-based `_compute_rule_based_score`
method for a trained ML model (see detector/ml_classifier usage via
config.USE_ML_MODEL) WITHOUT changing this class's public interface or
any calling code in app.py.
"""

import time
from dataclasses import dataclass
from typing import Optional

import joblib

import config
from utils.geometry import clamp
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FatigueResult:
    """
    Output of one fatigue-engine evaluation.

    Attributes:
        score: Fatigue score from 0 (fully alert) to 100 (fully fatigued).
        status: One of "SAFE", "WARNING", "CRITICAL".
    """

    score: float
    status: str


class FatigueEngine:
    """
    Aggregates per-frame detection signals into a unified fatigue score.

    Two scoring backends are supported:
      1. Rule-based weighted scoring (default, Phase 6).
      2. A trained ML classifier (Phase 9), loaded from
         config.ML_MODEL_PATH when config.USE_ML_MODEL is True.

    The public `evaluate()` method signature never changes between the two
    backends, so app.py and other callers require no modification when
    switching modes.
    """

    def __init__(self, use_ml_model: bool = config.USE_ML_MODEL) -> None:
        """
        Initialize the fatigue engine.

        Args:
            use_ml_model: If True, attempt to load a trained ML model and
                use it for scoring instead of the rule-based formula.
        """
        self._blink_timestamps: list = []
        self.use_ml_model = use_ml_model
        self._model: Optional[object] = None

        if self.use_ml_model:
            self._load_ml_model()

    def _load_ml_model(self) -> None:
        """Attempt to load the trained ML fatigue classifier from disk."""
        try:
            self._model = joblib.load(config.ML_MODEL_PATH)
            logger.info("ML fatigue model loaded from %s", config.ML_MODEL_PATH)
        except FileNotFoundError:
            logger.warning(
                "ML model not found at %s; falling back to rule-based scoring.",
                config.ML_MODEL_PATH,
            )
            self.use_ml_model = False
            self._model = None

    def _update_blink_rate(self, blink_count: int) -> float:
        """
        Maintain a rolling 60-second window of blink timestamps and
        compute the current blink rate (blinks per minute).

        Args:
            blink_count: Cumulative blink count reported by EyeDetector.

        Returns:
            Estimated blinks-per-minute over the trailing 60-second window.
        """
        now = time.time()

        # Detect a new blink by comparing against how many we've recorded.
        if blink_count > len(self._blink_timestamps):
            new_blinks = blink_count - len(self._blink_timestamps)
            self._blink_timestamps.extend([now] * new_blinks)

        # Drop timestamps older than 60 seconds.
        self._blink_timestamps = [
            t for t in self._blink_timestamps if now - t <= 60.0
        ]

        return float(len(self._blink_timestamps))

    @staticmethod
    def _status_from_score(score: float) -> str:
        """
        Map a numeric fatigue score to a discrete status label.

        Args:
            score: Fatigue score (0-100).

        Returns:
            "CRITICAL", "WARNING", or "SAFE".
        """
        if score >= config.FATIGUE_SCORE_CRITICAL:
            return "CRITICAL"
        if score >= config.FATIGUE_SCORE_WARNING:
            return "WARNING"
        return "SAFE"

    def _compute_rule_based_score(
        self,
        ear: float,
        mar: float,
        blink_rate: float,
        eye_closure_duration: float,
        head_distraction_duration: float,
    ) -> float:
        """
        Compute the fatigue score using a weighted rule-based formula.

        Each component is normalized to a 0-1 "severity" sub-score, then
        combined using the weights defined in config.py.

        Args:
            ear: Current Eye Aspect Ratio.
            mar: Current Mouth Aspect Ratio.
            blink_rate: Estimated blinks per minute.
            eye_closure_duration: Seconds eyes have been continuously closed.
            head_distraction_duration: Seconds head has been non-forward.

        Returns:
            Fatigue score from 0 to 100.
        """
        # Eye severity: scales with how far below threshold EAR is, and
        # how long the eyes have been closed.
        eye_closure_severity = clamp(
            eye_closure_duration / config.EYE_CLOSURE_DURATION_THRESHOLD, 0.0, 1.0
        )
        ear_severity = clamp(
            (config.EAR_THRESHOLD - ear) / config.EAR_THRESHOLD, 0.0, 1.0
        )
        eye_severity = max(eye_closure_severity, ear_severity)

        # Yawn severity: scales with how far MAR exceeds threshold.
        yawn_severity = clamp(
            (mar - config.MAR_THRESHOLD) / config.MAR_THRESHOLD, 0.0, 1.0
        )

        # Head pose severity: scales with sustained distraction duration.
        head_severity = clamp(
            head_distraction_duration / config.HEAD_POSE_DISTRACTION_DURATION,
            0.0,
            1.0,
        )

        # Blink rate severity: abnormally low or high blink rate both
        # indicate fatigue (too few = staring/microsleep, too many = strain).
        if blink_rate < config.NORMAL_BLINK_RATE_MIN:
            blink_severity = clamp(
                (config.NORMAL_BLINK_RATE_MIN - blink_rate)
                / config.NORMAL_BLINK_RATE_MIN,
                0.0,
                1.0,
            )
        elif blink_rate > config.NORMAL_BLINK_RATE_MAX:
            blink_severity = clamp(
                (blink_rate - config.NORMAL_BLINK_RATE_MAX)
                / config.NORMAL_BLINK_RATE_MAX,
                0.0,
                1.0,
            )
        else:
            blink_severity = 0.0

        weighted_score = (
            eye_severity * config.FATIGUE_WEIGHT_EYE
            + yawn_severity * config.FATIGUE_WEIGHT_YAWN
            + head_severity * config.FATIGUE_WEIGHT_HEAD_POSE
            + blink_severity * config.FATIGUE_WEIGHT_BLINK_RATE
        )

        return clamp(weighted_score * 100.0, 0.0, 100.0)

    def _compute_ml_score(
        self,
        ear: float,
        mar: float,
        blink_rate: float,
        eye_closure_duration: float,
        head_yaw: float,
        head_pitch: float,
    ) -> float:
        """
        Compute the fatigue score using the trained ML classifier.

        Args:
            ear: Current Eye Aspect Ratio.
            mar: Current Mouth Aspect Ratio.
            blink_rate: Estimated blinks per minute.
            eye_closure_duration: Seconds eyes have been continuously closed.
            head_yaw: Current head yaw angle in degrees.
            head_pitch: Current head pitch angle in degrees.

        Returns:
            Fatigue score from 0 to 100, derived from the model's predicted
            class probabilities.
        """
        features = [[ear, mar, blink_rate, eye_closure_duration, head_yaw, head_pitch]]

        try:
            probabilities = self._model.predict_proba(features)[0]
            # Map class probabilities to a continuous 0-100 score using the
            # class order defined in config.ML_CLASS_LABELS
            # (NORMAL=0, WARNING=50, CRITICAL=100 contribution weights).
            class_weights = {"NORMAL": 0.0, "WARNING": 50.0, "CRITICAL": 100.0}
            score = sum(
                probabilities[i] * class_weights[label]
                for i, label in enumerate(config.ML_CLASS_LABELS)
            )
            return clamp(score, 0.0, 100.0)
        except Exception as exc:
            logger.error("ML scoring failed, falling back to 0: %s", exc)
            return 0.0

    def evaluate(
        self,
        ear: float,
        mar: float,
        blink_count: int,
        eye_closure_duration: float,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        head_distraction_duration: float = 0.0,
    ) -> FatigueResult:
        """
        Evaluate the current driver state and produce a fatigue score
        and status. This is the single public entry point used by app.py.

        Args:
            ear: Current Eye Aspect Ratio.
            mar: Current Mouth Aspect Ratio.
            blink_count: Cumulative blink count.
            eye_closure_duration: Seconds eyes have been continuously closed.
            head_yaw: Current head yaw angle in degrees.
            head_pitch: Current head pitch angle in degrees.
            head_distraction_duration: Seconds head has been non-forward.

        Returns:
            A FatigueResult containing the numeric score and status label.
        """
        blink_rate = self._update_blink_rate(blink_count)

        if self.use_ml_model and self._model is not None:
            score = self._compute_ml_score(
                ear, mar, blink_rate, eye_closure_duration, head_yaw, head_pitch
            )
        else:
            score = self._compute_rule_based_score(
                ear, mar, blink_rate, eye_closure_duration, head_distraction_duration
            )

        status = self._status_from_score(score)
        return FatigueResult(score=score, status=status)
