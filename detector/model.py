"""ML model for voice deception detection.

Ensemble of RandomForest + GradientBoosting + SVM (VotingClassifier).

Key design decisions:
  - Feature names recorded at training time; inference aligns to them.
  - StandardScaler applied inside the pipeline for all estimators.
  - Synthetic pre-trained model shipped so the app works out of the box.
  - Model versioning: each save includes a UTC timestamp in the filename.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .audio_features import AudioFeatureExtractor
from .utils import get_models_dir

logger = logging.getLogger(__name__)

_SYNTHETIC_N_PER_CLASS = 1000  # samples per class for the pre-trained model
_MIN_SAMPLES_PER_CLASS = 3     # minimum real samples before retraining


class LieDetectorModel:
    """Trainable lie/truth detection model.

    Usage::

        model = LieDetectorModel()
        model.load()          # loads latest saved model or creates pre-trained one
        result = model.predict(features_dict)
    """

    def __init__(self, models_dir: Optional[str | Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self.is_trained: bool = False
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: Optional[str | Path] = None) -> bool:
        """Load a saved model.  If *model_path* is None, loads the latest.

        Returns True on success, False if no model file is found.
        """
        if model_path:
            path = Path(model_path)
        else:
            path = self._latest_model_path()

        if path is None:
            logger.info("No saved model found; creating pre-trained model.")
            self.create_pretrained_model()
            return True

        try:
            bundle = joblib.load(str(path))
            self._pipeline = bundle["pipeline"]
            self.feature_names = bundle["feature_names"]
            self.metadata = bundle.get("metadata", {})
            self.is_trained = True
            logger.info("Model loaded from %s", path.name)
            return True
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", path, exc)
            return False

    def save(self, tag: Optional[str] = None) -> Path:
        """Save the current model with a UTC timestamp.

        Args:
            tag: Optional extra label appended to the filename.

        Returns:
            Path of the saved file.
        """
        if self._pipeline is None:
            raise RuntimeError("No model to save; train first.")

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"model_{ts}"
        if tag:
            name += f"_{tag}"
        name += ".joblib"
        path = self.models_dir / name

        bundle = {
            "pipeline": self._pipeline,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        joblib.dump(bundle, str(path))
        logger.info("Model saved to %s", path.name)
        return path

    def train(
        self,
        features_list: list[dict[str, float]],
        labels: list[int],
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Train the ensemble on labelled data.

        Args:
            features_list: List of feature dicts (one per sample).
            labels: Binary labels (0 = truth, 1 = lie).
            cv_folds: Number of cross-validation folds for accuracy estimate.

        Returns:
            Dict with ``accuracy``, ``cv_accuracy``, ``report``, and
            ``feature_count``.

        Raises:
            ValueError: If fewer than ``_MIN_SAMPLES_PER_CLASS`` per class.
        """
        if len(features_list) != len(labels):
            raise ValueError("features_list and labels must have the same length.")

        truth_n = labels.count(0)
        lie_n = labels.count(1)
        if truth_n < _MIN_SAMPLES_PER_CLASS or lie_n < _MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Need at least {_MIN_SAMPLES_PER_CLASS} samples per class; "
                f"got {truth_n} truth, {lie_n} lie."
            )

        # Collect union of all feature names from the training set
        all_names: list[str] = []
        seen: set[str] = set()
        for fd in features_list:
            for k in fd:
                if k not in seen:
                    all_names.append(k)
                    seen.add(k)

        self.feature_names = all_names
        X = self._features_to_matrix(features_list)
        y = np.array(labels, dtype=np.int8)

        pipeline = self._build_pipeline()
        pipeline.fit(X, y)
        self._pipeline = pipeline
        self.is_trained = True

        # Accuracy on training set
        y_pred = pipeline.predict(X)
        train_acc = float(accuracy_score(y, y_pred))

        # Cross-validation (only when enough samples)
        cv_acc = 0.0
        if len(labels) >= cv_folds * _MIN_SAMPLES_PER_CLASS:
            try:
                cv_scores = cross_val_score(
                    self._build_pipeline(),
                    X, y,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring="accuracy",
                )
                cv_acc = float(np.mean(cv_scores))
            except Exception as exc:
                logger.warning("CV failed: %s", exc)

        report = classification_report(
            y, y_pred,
            target_names=["Truth", "Lie"],
            output_dict=True,
            zero_division=0,
        )

        self.metadata = {
            "trained_on": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": len(labels),
            "n_truth": truth_n,
            "n_lie": lie_n,
            "train_accuracy": train_acc,
            "cv_accuracy": cv_acc,
            "n_features": len(self.feature_names),
            "synthetic": False,
        }

        self.save()
        return {
            "accuracy": train_acc,
            "cv_accuracy": cv_acc,
            "report": report,
            "feature_count": len(self.feature_names),
        }

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict truth/lie for a feature dict.

        Args:
            features: Feature dict from :class:`AudioFeatureExtractor`.

        Returns:
            Dict with:
              - ``truth_probability`` (float 0–1)
              - ``lie_probability`` (float 0–1)
              - ``prediction`` ("Truth", "Lie", or "Uncertain")
              - ``confidence`` (float 0–1)
              - ``is_pretrained`` (bool)
              - ``feature_analysis`` (dict of indicator strings)

        Raises:
            RuntimeError: If no model has been trained / loaded.
        """
        if self._pipeline is None or not self.is_trained:
            raise RuntimeError("Model is not trained. Call load() or train() first.")

        X = self._features_to_matrix([features])
        proba = self._pipeline.predict_proba(X)[0]

        # Class 0 = truth, class 1 = lie
        classes = list(self._pipeline.named_steps["clf"].classes_)
        truth_idx = classes.index(0) if 0 in classes else 0
        lie_idx = classes.index(1) if 1 in classes else 1

        truth_prob = float(proba[truth_idx])
        lie_prob = float(proba[lie_idx])

        # Dampen toward 0.5 for synthetic/pretrained models — they lack real data
        if self.metadata.get("synthetic", False):
            truth_prob = 0.5 + (truth_prob - 0.5) * 0.6
            lie_prob = 1.0 - truth_prob

        confidence = float(max(truth_prob, lie_prob))

        if truth_prob >= 0.58:
            prediction = "Truth"
        elif lie_prob >= 0.58:
            prediction = "Lie"
        else:
            prediction = "Uncertain"

        feature_analysis = self._analyze_indicators(features)

        return {
            "truth_probability": truth_prob,
            "lie_probability": lie_prob,
            "prediction": prediction,
            "confidence": confidence,
            "is_pretrained": self.metadata.get("synthetic", False),
            "feature_analysis": feature_analysis,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances averaged over RF and GB estimators.

        Returns:
            Dict mapping feature name → importance score.
        """
        if self._pipeline is None:
            return {}
        clf = self._pipeline.named_steps["clf"]
        importances = np.zeros(len(self.feature_names))
        count = 0
        for name, estimator in clf.estimators:
            if hasattr(estimator, "feature_importances_"):
                importances += estimator.feature_importances_
                count += 1
        if count > 0:
            importances /= count
        return {n: float(v) for n, v in zip(self.feature_names, importances)}

    def create_pretrained_model(self) -> None:
        """Generate synthetic data and train the initial model.

        Saves the model to disk.  After this call the model is ready for
        predictions.
        """
        logger.info("Generating synthetic training data (%d samples/class)...",
                    _SYNTHETIC_N_PER_CLASS)
        features_list, labels = _generate_synthetic_data(_SYNTHETIC_N_PER_CLASS)
        base_names = AudioFeatureExtractor.core_feature_names()
        self.feature_names = base_names

        X = self._features_to_matrix(features_list)
        y = np.array(labels, dtype=np.int8)

        pipeline = self._build_pipeline()
        pipeline.fit(X, y)
        self._pipeline = pipeline
        self.is_trained = True

        y_pred = pipeline.predict(X)
        acc = float(accuracy_score(y, y_pred))
        self.metadata = {
            "trained_on": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": len(labels),
            "n_truth": labels.count(0),
            "n_lie": labels.count(1),
            "train_accuracy": acc,
            "cv_accuracy": 0.0,
            "n_features": len(self.feature_names),
            "synthetic": True,
        }
        self.save(tag="pretrained")
        logger.info("Pre-trained model ready (train acc=%.2f).", acc)

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of current model state."""
        return {
            "is_trained": self.is_trained,
            "synthetic": self.metadata.get("synthetic", True),
            "n_features": len(self.feature_names),
            "n_samples": self.metadata.get("n_samples", 0),
            "n_truth": self.metadata.get("n_truth", 0),
            "n_lie": self.metadata.get("n_lie", 0),
            "train_accuracy": self.metadata.get("train_accuracy", 0.0),
            "cv_accuracy": self.metadata.get("cv_accuracy", 0.0),
            "trained_on": self.metadata.get("trained_on", "—"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pipeline() -> Pipeline:
        """Return a fresh (unfitted) sklearn Pipeline."""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        svm = SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42,
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
            voting="soft",
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ])

    def _features_to_matrix(
        self, features_list: list[dict[str, float]]
    ) -> np.ndarray:
        """Convert a list of feature dicts to a 2-D NumPy array.

        Missing features are filled with 0.
        """
        n = len(features_list)
        m = len(self.feature_names)
        X = np.zeros((n, m), dtype=np.float64)  # float64: sklearn StandardScaler precision
        for i, fd in enumerate(features_list):
            for j, name in enumerate(self.feature_names):
                val = fd.get(name, 0.0)
                X[i, j] = float(val) if np.isfinite(float(val)) else 0.0
        return X

    def _latest_model_path(self) -> Optional[Path]:
        """Return the most recently saved .joblib model file, or None."""
        candidates = sorted(self.models_dir.glob("model_*.joblib"), reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _analyze_indicators(features: dict[str, float]) -> dict[str, str]:
        """Generate human-readable indicator labels from key features."""
        def level(value: float, low: float, high: float) -> str:
            if value < low:
                return "Low"
            if value > high:
                return "High"
            return "Normal"

        indicators: dict[str, str] = {}

        f0_std = features.get("f0_std", 0.0)
        indicators["Pitch Variability"] = level(f0_std, 15.0, 45.0)

        jitter = features.get("jitter", 0.0)
        indicators["Voice Tremor (Jitter)"] = level(jitter, 0.005, 0.020)

        shimmer = features.get("shimmer", 0.0)
        indicators["Amplitude Instability (Shimmer)"] = level(shimmer, 0.03, 0.15)

        pause_count = features.get("pause_count", 0.0)
        indicators["Pause Frequency"] = level(pause_count, 1.0, 5.0)

        pause_dur = features.get("pause_mean_duration", 0.0)
        indicators["Pause Duration"] = level(pause_dur, 0.15, 0.40)

        speech_ratio = features.get("speech_ratio", 0.5)
        indicators["Speaking Ratio"] = level(speech_ratio, 0.50, 0.85)

        hnr = features.get("hnr", 0.0)
        indicators["Voice Clarity (HNR)"] = level(hnr, 0.2, 1.2)

        return indicators


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_per_class: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[dict[str, float]], list[int]]:
    """Create *n_per_class* synthetic feature vectors for each class.

    Statistical distributions are inspired by published deception-detection
    research (F0 increase under stress, elevated jitter/shimmer, more pauses,
    reduced HNR, etc.).

    Returns:
        (features_list, labels) where labels 0 = truth, 1 = lie.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    feature_names = AudioFeatureExtractor.core_feature_names()
    features_list: list[dict[str, float]] = []
    labels: list[int] = []

    for label in [0, 1]:
        for _ in range(n_per_class):
            fd = _sample_features(label, rng, feature_names)
            features_list.append(fd)
            labels.append(label)

    return features_list, labels


def _sample_features(
    label: int,
    rng: np.random.Generator,
    feature_names: list[str],
) -> dict[str, float]:
    """Sample one synthetic feature vector.

    label 0 = truth, label 1 = lie.
    """
    is_lie = label == 1

    def r(mu: float, sigma: float, lo: float = 0.0, hi: float = 1e9) -> float:
        return float(np.clip(rng.normal(mu, sigma), lo, hi))

    # Pitch (Hz) — wide overlap, subtle lie shift (~10-15 Hz)
    f0_mean = r(145, 30) if is_lie else r(130, 30)
    f0_std = r(30, 10, 1) if is_lie else r(22, 8, 1)
    f0_range = f0_std * 3.5
    f0_min = max(f0_mean - f0_range / 2, 60.0)
    f0_max = f0_mean + f0_range / 2

    # Jitter — subtle difference, realistic range 0.003-0.025
    jitter = r(0.015, 0.006, 0.002) if is_lie else r(0.008, 0.004, 0.002)

    # Voiced fraction
    voiced_frac = r(0.62, 0.10, 0.1, 1.0) if is_lie else r(0.76, 0.08, 0.1, 1.0)

    # Energy
    rms_mean = r(0.08, 0.02, 0.01)
    rms_std = r(0.04, 0.01, 0.001) if is_lie else r(0.025, 0.008, 0.001)
    shimmer = r(0.10, 0.03, 0.01) if is_lie else r(0.05, 0.015, 0.01)
    energy_entropy = r(5.5, 0.8) if is_lie else r(4.8, 0.7)

    # Speech rate
    speech_rate = r(2.8, 0.8, 0.5) if is_lie else r(3.6, 0.5, 0.5)
    speech_ratio = r(0.60, 0.10, 0.2, 1.0) if is_lie else r(0.74, 0.08, 0.2, 1.0)
    pause_count = float(max(0, int(rng.poisson(4.5 if is_lie else 2.0))))
    pause_mean_dur = r(0.38, 0.15, 0.0) if is_lie else r(0.20, 0.08, 0.0)
    pause_total = pause_count * pause_mean_dur
    duration_sec = r(12, 4, 2)

    # Spectral
    spec_centroid = r(1500, 300, 200) if is_lie else r(1350, 250, 200)
    spec_centroid_std = r(350, 80) if is_lie else r(250, 60)
    spec_rolloff = r(3500, 600)
    spec_bw = r(2200, 400)
    zcr_mean = r(0.08, 0.02, 0.001) if is_lie else r(0.065, 0.015, 0.001)
    zcr_std = r(0.04, 0.01, 0.001)
    hnr = r(0.2, 0.6, -1.0, 2.0) if is_lie else r(0.9, 0.5, -1.0, 2.0)

    fd: dict[str, float] = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "f0_median": f0_mean + r(0, 5, -20, 20),
        "voiced_fraction": voiced_frac,
        "jitter": jitter,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_mean + rms_std * 2.5,
        "shimmer": shimmer,
        "energy_entropy": energy_entropy,
        "speech_rate": speech_rate,
        "speech_ratio": speech_ratio,
        "pause_count": pause_count,
        "pause_mean_duration": pause_mean_dur,
        "pause_total_duration": pause_total,
        "duration_seconds": duration_sec,
        "spectral_centroid_mean": spec_centroid,
        "spectral_centroid_std": spec_centroid_std,
        "spectral_rolloff_mean": spec_rolloff,
        "spectral_bandwidth_mean": spec_bw,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "hnr": hnr,
    }

    # MFCCs — realistic librosa ranges; mfcc_1 is the dominant (DC-like) coefficient
    # Real speech: mfcc_1 ~= -200 to -100; mfcc_2+ decreasing in magnitude
    # Truth/lie differences are subtle (~5-10 units on a ±30-40 sigma)
    mfcc1_mu = -170.0 if is_lie else -180.0
    fd["mfcc_1_mean"] = r(mfcc1_mu, 35.0)
    fd["mfcc_1_std"] = r(18.0 if is_lie else 15.0, 5.0, 2.0)

    # Baseline means and subtle lie shifts for mfcc 2-13
    _mfcc_base  = [32, -12, 8, -5, 4, -3, 2, -2, 2, -1, 1, -1]
    _mfcc_shift = [ 5,  -3, 2, -1, 2, -1, 1,  0, 1, -1, 1,  0]
    for i in range(2, 14):
        base = _mfcc_base[i - 2]
        shift = _mfcc_shift[i - 2] if is_lie else 0.0
        sigma = max(16.0 - (i - 2) * 0.9, 8.0)
        fd[f"mfcc_{i}_mean"] = r(base + shift, sigma)
        fd[f"mfcc_{i}_std"] = r(18.0 if is_lie else 15.0, 5.0, 2.0)

    fd["delta_mfcc_mean"] = r(0.05 if is_lie else -0.02, 0.5)
    fd["delta_mfcc_std"] = r(4.0 if is_lie else 3.5, 0.8, 0.5)

    # Validate names match the spec
    for name in feature_names:
        if name not in fd:
            fd[name] = 0.0

    return fd
