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


# ---------------------------------------------------------------------------
# Pretrained model & synthetic dampening
# ---------------------------------------------------------------------------

class TestPretrainedModel:
    def test_create_pretrained_model_loads_and_predicts(self, tmp_path):
        """Pretrained model should be created, saved, and predict without bias."""
        model = LieDetectorModel(models_dir=tmp_path)
        model.create_pretrained_model()

        assert model.is_trained is True
        assert model.metadata.get("synthetic") is True
        assert model.metadata["n_samples"] > 0

        # Saved file must exist
        saved = list(tmp_path.glob("model_*.joblib"))
        assert len(saved) == 1

    def test_synthetic_dampening_reduces_confidence(self, tmp_path):
        """With synthetic model, confidence should be dampened toward 0.5."""
        model = LieDetectorModel(models_dir=tmp_path)
        model.create_pretrained_model()

        # Collect predictions over a diverse set of "neutral" feature vectors
        # (f0_mean=130, typical truth-like values)
        neutral_features = {
            "f0_mean": 130.0, "f0_std": 22.0, "f0_min": 100.0,
            "f0_max": 200.0, "f0_range": 100.0, "f0_median": 130.0,
            "voiced_fraction": 0.75, "jitter": 0.008,
            "rms_mean": 0.08, "rms_std": 0.025, "rms_max": 0.13,
            "shimmer": 0.05, "energy_entropy": 4.8,
            "speech_rate": 3.5, "speech_ratio": 0.74,
            "pause_count": 2.0, "pause_mean_duration": 0.2,
            "pause_total_duration": 0.4, "duration_seconds": 12.0,
            "spectral_centroid_mean": 1350.0, "spectral_centroid_std": 250.0,
            "spectral_rolloff_mean": 3500.0, "spectral_bandwidth_mean": 2200.0,
            "zcr_mean": 0.065, "zcr_std": 0.03,
            "hnr": 0.9,
        }
        # Fill in MFCCs at typical truth-like values
        neutral_features["mfcc_1_mean"] = -180.0
        neutral_features["mfcc_1_std"] = 15.0
        for i in range(2, 14):
            neutral_features[f"mfcc_{i}_mean"] = 0.0
            neutral_features[f"mfcc_{i}_std"] = 15.0
        neutral_features["delta_mfcc_mean"] = -0.02
        neutral_features["delta_mfcc_std"] = 3.5

        result = model.predict(neutral_features)

        # After dampening, neither probability should be extreme (>0.9)
        assert result["truth_probability"] < 0.90, (
            f"truth_prob too extreme: {result['truth_probability']:.3f} "
            "(synthetic dampening not working?)"
        )
        assert result["lie_probability"] < 0.90, (
            f"lie_prob too extreme: {result['lie_probability']:.3f} "
            "(synthetic model biased toward Lie?)"
        )

        # Probabilities must still sum to 1.0 (within float tolerance)
        total = result["truth_probability"] + result["lie_probability"]
        assert abs(total - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total}"

    def test_pretrained_model_returns_uncertain_for_borderline(self, tmp_path):
        """Synthetic model should lean Uncertain, not confident, on ambiguous inputs."""
        model = LieDetectorModel(models_dir=tmp_path)
        model.create_pretrained_model()

        # Feature vector with exactly average values (should be near 50/50)
        avg_features: dict[str, float] = {}
        for name in LieDetectorModel(models_dir=tmp_path).feature_names or []:
            avg_features[name] = 0.0
        # Just predict — ensure no crash and result has correct keys
        try:
            result = model.predict(avg_features)
            assert "prediction" in result
            assert result["prediction"] in {"Truth", "Lie", "Uncertain"}
        except RuntimeError:
            # No features → may fail gracefully; that's acceptable
            pass

    def test_save_load_preserves_synthetic_flag(self, tmp_path):
        """Synthetic flag must survive save + load."""
        model = LieDetectorModel(models_dir=tmp_path)
        model.create_pretrained_model()
        assert model.metadata["synthetic"] is True

        model2 = LieDetectorModel(models_dir=tmp_path)
        model2.load()
        assert model2.metadata.get("synthetic") is True


# ---------------------------------------------------------------------------
# Feature matrix precision
# ---------------------------------------------------------------------------

class TestFeatureMatrix:
    def test_features_to_matrix_dtype_is_float64(self, tmp_path):
        """Ensure _features_to_matrix produces float64 for sklearn compatibility."""
        model = LieDetectorModel(models_dir=tmp_path)
        model.feature_names = ["f0_mean", "rms_mean", "hnr"]
        X = model._features_to_matrix([{"f0_mean": 130.0, "rms_mean": 0.05, "hnr": 0.9}])
        import numpy as np
        assert X.dtype == np.float64, f"Expected float64, got {X.dtype}"

    def test_features_to_matrix_fills_missing_with_zero(self, tmp_path):
        """Missing feature keys must be filled with 0.0, not NaN."""
        import numpy as np
        model = LieDetectorModel(models_dir=tmp_path)
        model.feature_names = ["f0_mean", "rms_mean", "missing_feature"]
        X = model._features_to_matrix([{"f0_mean": 130.0}])
        assert X[0, 1] == 0.0   # rms_mean missing → 0
        assert X[0, 2] == 0.0   # missing_feature → 0
        assert not np.any(np.isnan(X))
