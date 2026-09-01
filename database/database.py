"""
database/database.py

SQLite persistence layer for the AI-DMS application.

Encapsulates all database access behind the `EventDatabase` class so that
no other module writes raw SQL. This keeps the schema and query logic in
one place and makes it easy to swap SQLite for another backend later.
"""

import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EventRecord:
    """
    Represents a single logged driver-monitoring event.

    Attributes:
        timestamp: Unix timestamp (seconds) when the event occurred.
        ear: Eye Aspect Ratio at the time of the event.
        mar: Mouth Aspect Ratio at the time of the event.
        blink_count: Cumulative blink count at the time of the event.
        fatigue_score: Computed fatigue score (0-100).
        alert_type: Status label, e.g. "SAFE", "WARNING", "CRITICAL".
        screenshot_path: Path to a saved screenshot, if any.
    """

    timestamp: float
    ear: float
    mar: float
    blink_count: int
    fatigue_score: float
    alert_type: str
    screenshot_path: Optional[str] = None


class EventDatabase:
    """
    Manages a SQLite database storing driver-monitoring events.

    Provides a minimal, explicit API (`initialize`, `insert_event`,
    `fetch_recent_events`) so the rest of the application never has to
    write SQL directly.
    """

    def __init__(self, db_path: str = config.DATABASE_PATH) -> None:
        """
        Initialize the database wrapper (does not open a connection yet).

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """
        Open the database connection and create the events table if it
        does not already exist.
        """
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.DB_TABLE_EVENTS} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                ear REAL NOT NULL,
                mar REAL NOT NULL,
                blink_count INTEGER NOT NULL,
                fatigue_score REAL NOT NULL,
                alert_type TEXT NOT NULL,
                screenshot_path TEXT
            )
            """
        )
        self._connection.commit()
        logger.info("Database initialized at %s", self.db_path)

    def insert_event(self, event: EventRecord) -> None:
        """
        Insert a new event record into the database.

        Args:
            event: The EventRecord to persist.

        Raises:
            RuntimeError: If the database has not been initialized.
        """
        if self._connection is None:
            raise RuntimeError("Database.initialize() must be called first.")

        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            INSERT INTO {config.DB_TABLE_EVENTS}
                (timestamp, ear, mar, blink_count, fatigue_score, alert_type, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp,
                event.ear,
                event.mar,
                event.blink_count,
                event.fatigue_score,
                event.alert_type,
                event.screenshot_path,
            ),
        )
        self._connection.commit()
        logger.debug("Event inserted: %s", event)

    def fetch_recent_events(self, limit: int = 50) -> List[tuple]:
        """
        Fetch the most recent N events, newest first.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            A list of raw row tuples from the events table.
        """
        if self._connection is None:
            raise RuntimeError("Database.initialize() must be called first.")

        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            SELECT * FROM {config.DB_TABLE_EVENTS}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    def close(self) -> None:
        """Close the database connection if it is open."""
        if self._connection is not None:
            self._connection.close()
            logger.info("Database connection closed.")
            self._connection = None

    @staticmethod
    def now() -> float:
        """Convenience helper returning the current Unix timestamp."""
        return time.time()
