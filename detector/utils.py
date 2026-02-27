"""Utility functions for the lie detector system."""

import os
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def get_prediction_label(truth_probability: float) -> str:
    """Convert truth probability to a human-readable label.

    Args:
        truth_probability: Float between 0 and 1 (1 = truth).

    Returns:
        "Truth", "Lie", or "Uncertain"
    """
    if truth_probability >= 0.70:
        return "Truth"
    elif truth_probability <= 0.30:
        return "Lie"
    else:
        return "Uncertain"


def get_confidence_color(truth_probability: float) -> str:
    """Return a CSS color based on truth probability.

    Args:
        truth_probability: Float between 0 and 1.

    Returns:
        Hex color string: green (truth), yellow (uncertain), red (lie).
    """
    if truth_probability >= 0.70:
        return "#22c55e"   # Green
    elif truth_probability >= 0.50:
        return "#eab308"   # Yellow
    elif truth_probability >= 0.30:
        return "#f97316"   # Orange
    else:
        return "#ef4444"   # Red


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string, e.g. "1m 23s" or "45s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_percentage(probability: float) -> str:
    """Format a probability as a percentage string.

    Args:
        probability: Float between 0 and 1.

    Returns:
        Formatted percentage string, e.g. "73.4%".
    """
    return f"{probability * 100:.1f}%"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, returning default if b is zero.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value returned when b == 0.

    Returns:
        a / b or default.
    """
    if b == 0:
        return default
    return a / b


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        The Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Return the project root directory (parent of this file's package)."""
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """Return the models directory, creating it if necessary."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_data_dir() -> Path:
    """Return the data directory, creating it if necessary."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_db_path() -> Path:
    """Return the SQLite database path."""
    return get_project_root() / "data" / "lie_detector.db"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (default INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
