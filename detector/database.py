"""SQLite-backed persistence for analyses and training samples."""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .utils import get_db_path

logger = logging.getLogger(__name__)


class Database:
    """Manages the SQLite database for the lie detector application.

    Stores:
      - ``analyses``: every prediction run through the system.
      - ``training_samples``: labeled recordings used to train/retrain the model.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        """Initialise and migrate the database.

        Args:
            db_path: Path to the SQLite file.  Defaults to the project data dir.
        """
        self.db_path = Path(db_path) if db_path else get_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename        TEXT    NOT NULL,
                    timestamp       TEXT    NOT NULL,
                    truth_prob      REAL    NOT NULL,
                    lie_prob        REAL    NOT NULL,
                    prediction      TEXT    NOT NULL,
                    confidence      REAL    NOT NULL,
                    features_json   TEXT,
                    transcript      TEXT,
                    duration_sec    REAL,
                    model_version   TEXT
                );

                CREATE TABLE IF NOT EXISTS training_samples (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename        TEXT    NOT NULL,
                    label           INTEGER NOT NULL,  -- 0 = truth, 1 = lie
                    label_text      TEXT    NOT NULL,
                    timestamp       TEXT    NOT NULL,
                    features_json   TEXT,
                    duration_sec    REAL
                );
            """)
        logger.debug("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Analysis records
    # ------------------------------------------------------------------

    def save_analysis(
        self,
        filename: str,
        truth_prob: float,
        lie_prob: float,
        prediction: str,
        confidence: float,
        features: Optional[dict[str, Any]] = None,
        transcript: Optional[str] = None,
        duration_sec: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Persist an analysis result and return its row ID.

        Args:
            filename: Original audio file name.
            truth_prob: Probability of truth (0–1).
            lie_prob: Probability of lie (0–1).
            prediction: "Truth", "Lie", or "Uncertain".
            confidence: Confidence score (0–1).
            features: Feature dictionary (serialised as JSON).
            transcript: Transcribed text, if available.
            duration_sec: Audio duration in seconds.
            model_version: Model timestamp/version string.

        Returns:
            The ``id`` of the newly inserted row.
        """
        ts = datetime.now().isoformat(timespec="seconds")
        features_json = json.dumps(features) if features else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses
                    (filename, timestamp, truth_prob, lie_prob, prediction,
                     confidence, features_json, transcript, duration_sec, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    ts,
                    truth_prob,
                    lie_prob,
                    prediction,
                    confidence,
                    features_json,
                    transcript,
                    duration_sec,
                    model_version,
                ),
            )
            return cursor.lastrowid

    def get_analyses(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent analyses.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts with analysis data.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, timestamp, truth_prob, lie_prob,
                       prediction, confidence, transcript, duration_sec, model_version
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_analyses(self) -> int:
        """Delete all analysis records.

        Returns:
            Number of rows deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM analyses")
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Training samples
    # ------------------------------------------------------------------

    def add_training_sample(
        self,
        filename: str,
        label: int,
        features: Optional[dict[str, Any]] = None,
        duration_sec: Optional[float] = None,
    ) -> int:
        """Add a labelled training sample.

        Args:
            filename: Audio file name.
            label: 0 = truth, 1 = lie.
            features: Extracted feature dictionary.
            duration_sec: Audio duration in seconds.

        Returns:
            The ``id`` of the new row.
        """
        label_text = "truth" if label == 0 else "lie"
        ts = datetime.now().isoformat(timespec="seconds")
        features_json = json.dumps(features) if features else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO training_samples
                    (filename, label, label_text, timestamp, features_json, duration_sec)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename, label, label_text, ts, features_json, duration_sec),
            )
            return cursor.lastrowid

    def get_training_samples(self) -> list[dict[str, Any]]:
        """Return all training samples with their features.

        Returns:
            List of dicts including parsed ``features`` key.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, label, label_text, timestamp,
                       features_json, duration_sec
                FROM training_samples
                ORDER BY id ASC
                """
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["features"] = json.loads(d.pop("features_json")) if d["features_json"] else {}
            result.append(d)
        return result

    def get_training_stats(self) -> dict[str, int]:
        """Return counts of truth/lie samples.

        Returns:
            Dict with keys ``total``, ``truth``, ``lie``.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                        AS total,
                    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS truth,
                    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS lie
                FROM training_samples
                """
            ).fetchone()
        return {
            "total": row["total"] or 0,
            "truth": row["truth"] or 0,
            "lie": row["lie"] or 0,
        }

    def clear_training_samples(self) -> int:
        """Delete all training samples.

        Returns:
            Number of rows deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM training_samples")
            return cursor.rowcount

    def remove_training_sample(self, sample_id: int) -> bool:
        """Remove a single training sample by ID.

        Args:
            sample_id: Row ID to delete.

        Returns:
            True if a row was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM training_samples WHERE id = ?", (sample_id,)
            )
            return cursor.rowcount > 0
