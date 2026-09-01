"""
utils/alarm.py

Audible alarm system for the AI-DMS application.

Handles playing/stopping the alert sound using pygame's mixer, with
cooldown logic to prevent overlapping or rapidly-flickering alarms.
"""

import time
from typing import Optional

import pygame

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class AlarmSystem:
    """
    Manages playback of the audible drowsiness alarm.

    Ensures the alarm does not overlap itself, respects a cooldown period
    between triggers, and stops automatically once the driver recovers.
    """

    def __init__(self, sound_path: str = config.ALARM_SOUND_PATH) -> None:
        """
        Initialize the alarm system and the pygame mixer.

        Args:
            sound_path: Path to the .wav alarm sound file.
        """
        self.sound_path = sound_path
        self._is_playing: bool = False
        self._last_trigger_time: float = 0.0
        self._sound: Optional[pygame.mixer.Sound] = None

        try:
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(self.sound_path)
            logger.info("Alarm sound loaded from %s", self.sound_path)
        except Exception as exc:  # pygame raises generic errors on bad files
            logger.error("Could not load alarm sound: %s", exc)
            self._sound = None

    def trigger(self) -> None:
        """
        Start the alarm sound if it is not already playing and the
        cooldown period has elapsed.
        """
        if self._sound is None:
            return

        now = time.time()
        if self._is_playing:
            return  # Already sounding, do nothing.

        if now - self._last_trigger_time < config.ALARM_COOLDOWN_SECONDS:
            return  # Still within cooldown window.

        self._sound.play(loops=-1)  # Loop indefinitely until stopped.
        self._is_playing = True
        self._last_trigger_time = now
        logger.warning("Alarm triggered.")

    def stop(self) -> None:
        """Stop the alarm sound if it is currently playing."""
        if self._sound is None or not self._is_playing:
            return

        self._sound.stop()
        self._is_playing = False
        logger.info("Alarm stopped.")

    @property
    def is_playing(self) -> bool:
        """Whether the alarm is currently sounding."""
        return self._is_playing
