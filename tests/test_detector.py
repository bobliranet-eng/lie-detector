"""Basic tests for the lie detector components."""

import tempfile
from pathlib import Path

import pytest

from detector.audio_features import AudioFeatureExtractor
from detector.model import LieDetectorModel
from detector.database import Database
from detector.utils import (
    format_duration,
    format_percentage,
    get_prediction_label,
    safe_divide,
    get_confidence_color,
)


# ---------------------------------------------------------------------------
# AudioFeatureExtractor
# ---------------------------------------------------------------------------

class TestAudioFeatureExtractor:
    def test_instantiation(self):
        extractor = AudioFeatureExtractor()
        assert extractor is not None

    def test_instantiation_without_opensmile(self):
        extractor = AudioFeatureExtractor(use_opensmile=False)
        assert extractor is not None
        assert extractor._use_opensmile is False

    def test_core_feature_names_returns_list(self):
        names = AudioFeatureExtractor.core_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_core_feature_names_contains_expected(self):
        names = AudioFeatureExtractor.core_feature_names()
        for expected in ["f0_mean", "f0_std", "rms_mean", "zcr_mean", "hnr", "mfcc_1_mean"]:
            assert expected in names, f"{expected} missing from core_feature_names"

    def test_sample_rate(self):
        assert AudioFeatureExtractor.SAMPLE_RATE == 16_000


# ---------------------------------------------------------------------------
# LieDetectorModel
# ---------------------------------------------------------------------------

class TestLieDetectorModel:
    def test_instantiation(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        assert model is not None

    def test_is_trained_property_false_before_load(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        assert model.is_trained is False

    def test_get_stats_returns_dict(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        stats = model.get_stats()
        assert isinstance(stats, dict)
        assert "is_trained" in stats
        assert "synthetic" in stats

    def test_train_requires_minimum_samples(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        # Only 1 sample per class — should raise ValueError
        features = [{"f0_mean": 100.0, "rms_mean": 0.05}] * 2
        labels = [0, 1]
        with pytest.raises(ValueError, match="at least"):
            model.train(features, labels)

    def test_train_with_sufficient_samples(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        # 3 truth + 3 lie = minimum viable training set
        features = []
        labels = []
        for _ in range(3):
            features.append({"f0_mean": 140.0, "rms_mean": 0.04, "f0_std": 12.0})
            labels.append(0)
        for _ in range(3):
            features.append({"f0_mean": 170.0, "rms_mean": 0.08, "f0_std": 28.0})
            labels.append(1)
        result = model.train(features, labels)
        assert "accuracy" in result
        assert model.is_trained is True

    def test_predict_after_train(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        features = []
        labels = []
        for _ in range(5):
            features.append({"f0_mean": 140.0, "rms_mean": 0.04})
            labels.append(0)
        for _ in range(5):
            features.append({"f0_mean": 170.0, "rms_mean": 0.08})
            labels.append(1)
        model.train(features, labels)
        result = model.predict({"f0_mean": 145.0, "rms_mean": 0.05})
        assert "truth_probability" in result
        assert "lie_probability" in result
        assert "prediction" in result
        assert result["prediction"] in {"Truth", "Lie", "Uncertain"}

    def test_predict_raises_without_training(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        with pytest.raises(RuntimeError):
            model.predict({"f0_mean": 150.0})

    def test_save_and_load(self, tmp_path):
        model = LieDetectorModel(models_dir=tmp_path)
        features = []
        labels = []
        for _ in range(4):
            features.append({"f0_mean": 140.0})
            labels.append(0)
        for _ in range(4):
            features.append({"f0_mean": 170.0})
            labels.append(1)
        model.train(features, labels)
        saved_path = model.save()
        assert saved_path.exists()

        model2 = LieDetectorModel(models_dir=tmp_path)
        success = model2.load(model_path=saved_path)
        assert success is True
        assert model2.is_trained is True


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_create_with_temp_path(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path=db_path)
        assert db is not None
        assert db_path.exists()

    def test_save_and_get_analysis(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        row_id = db.save_analysis(
            filename="test.wav",
            truth_prob=0.8,
            lie_prob=0.2,
            prediction="Truth",
            confidence=0.8,
        )
        assert isinstance(row_id, int)
        analyses = db.get_analyses()
        assert len(analyses) == 1
        assert analyses[0]["filename"] == "test.wav"

    def test_add_and_get_training_sample(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.add_training_sample(
            filename="truth_sample.wav",
            label=0,
            features={"f0_mean": 140.0},
        )
        db.add_training_sample(
            filename="lie_sample.wav",
            label=1,
            features={"f0_mean": 170.0},
        )
        samples = db.get_training_samples()
        assert len(samples) == 2

    def test_get_training_stats(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.add_training_sample(filename="t1.wav", label=0)
        db.add_training_sample(filename="t2.wav", label=0)
        db.add_training_sample(filename="l1.wav", label=1)
        stats = db.get_training_stats()
        assert stats["truth"] == 2
        assert stats["lie"] == 1
        assert stats["total"] == 3

    def test_clear_analyses(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.save_analysis("a.wav", 0.6, 0.4, "Uncertain", 0.6)
        db.save_analysis("b.wav", 0.3, 0.7, "Lie", 0.7)
        deleted = db.clear_analyses()
        assert deleted == 2
        assert db.get_analyses() == []

    def test_clear_training_samples(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.add_training_sample("x.wav", label=0)
        db.add_training_sample("y.wav", label=1)
        deleted = db.clear_training_samples()
        assert deleted == 2
        assert db.get_training_samples() == []


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_format_duration_seconds(self):
        assert format_duration(45.0) == "45.0s"
        assert format_duration(0.5) == "0.5s"

    def test_format_duration_minutes(self):
        result = format_duration(90.0)
        assert "1m" in result

    def test_format_percentage(self):
        assert format_percentage(0.734) == "73.4%"
        assert format_percentage(1.0) == "100.0%"
        assert format_percentage(0.0) == "0.0%"

    def test_get_prediction_label(self):
        assert get_prediction_label(0.75) == "Truth"
        assert get_prediction_label(0.25) == "Lie"
        assert get_prediction_label(0.50) == "Uncertain"

    def test_safe_divide(self):
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(5.0, 0.0) == 0.0
        assert safe_divide(5.0, 0.0, default=99.0) == 99.0

    def test_get_confidence_color_returns_string(self):
        color = get_confidence_color(0.9)
        assert isinstance(color, str)
        assert color.startswith("#")
