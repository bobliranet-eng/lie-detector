"""Lie Detector - Voice Deception Analysis System.

Core engine for audio feature extraction, ML-based prediction,
and SQLite-backed persistence.
"""

from .audio_features import AudioFeatureExtractor
from .model import LieDetectorModel
from .database import Database
from .utils import get_prediction_label, get_confidence_color, format_duration

__all__ = [
    "AudioFeatureExtractor",
    "LieDetectorModel",
    "Database",
    "get_prediction_label",
    "get_confidence_color",
    "format_duration",
]
