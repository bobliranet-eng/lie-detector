"""Audio feature extraction using librosa and openSMILE.

Extracts:
  - Pitch (F0): mean, std, range, jitter, voiced fraction
  - Energy: RMS, shimmer, entropy
  - MFCCs (13 coefficients + delta)
  - Speech rate and pause statistics
  - Spectral features: centroid, rolloff, bandwidth, ZCR, HNR
  - eGeMAPS features via openSMILE (optional, gracefully degraded)
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import librosa.effects
import soundfile as sf

logger = logging.getLogger(__name__)

# Optional dependencies loaded at runtime.
_opensmile_available: Optional[bool] = None


def _check_opensmile() -> bool:
    global _opensmile_available
    if _opensmile_available is None:
        try:
            import opensmile  # noqa: F401
            _opensmile_available = True
        except ImportError:
            logger.warning("opensmile not available; eGeMAPS features will be skipped.")
            _opensmile_available = False
    return _opensmile_available


class AudioFeatureExtractor:
    """Extract acoustic features from an audio file.

    Usage::

        extractor = AudioFeatureExtractor()
        features = extractor.extract_all_features("recording.wav")
    """

    #: Target sample rate for all processing.
    SAMPLE_RATE: int = 16_000

    def __init__(self, use_opensmile: bool = True) -> None:
        """Initialise the extractor.

        Args:
            use_opensmile: Whether to attempt eGeMAPS extraction via openSMILE.
                           Falls back silently if the package is not installed.
        """
        self._use_opensmile = use_opensmile
        self._smile = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_features(self, file_path: str) -> dict[str, float]:
        """Extract the full feature set from *file_path*.

        Args:
            file_path: Path to an audio file (wav, mp3, m4a, ogg, flac).

        Returns:
            Dict mapping feature name → float value.

        Raises:
            ValueError: If the audio is too short (<0.5 s) or silent.
            RuntimeError: On unrecoverable extraction errors.
        """
        y, sr = self._load_audio(file_path)
        self._validate_audio(y, sr)

        features: dict[str, float] = {}

        features.update(self._pitch_features(y, sr))
        features.update(self._energy_features(y, sr))
        features.update(self._mfcc_features(y, sr))
        features.update(self._speech_rate_features(y, sr))
        features.update(self._spectral_features(y, sr))

        # openSMILE eGeMAPS (optional)
        if self._use_opensmile and _check_opensmile():
            tmp_wav = self._as_wav(file_path, y, sr)
            try:
                features.update(self._opensmile_features(tmp_wav))
            finally:
                if tmp_wav != file_path:
                    _safe_unlink(tmp_wav)

        return features

    @staticmethod
    def core_feature_names() -> list[str]:
        """Return the ordered list of non-openSMILE feature names.

        This list is stable across versions and is used by the model to align
        training and inference feature vectors.
        """
        names: list[str] = [
            # Pitch
            "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
            "f0_median", "voiced_fraction", "jitter",
            # Energy
            "rms_mean", "rms_std", "rms_max", "shimmer", "energy_entropy",
            # Speech rate
            "speech_rate", "speech_ratio", "pause_count",
            "pause_mean_duration", "pause_total_duration", "duration_seconds",
            # Spectral
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_rolloff_mean", "spectral_bandwidth_mean",
            "zcr_mean", "zcr_std", "hnr",
        ]
        # MFCC (1–13 mean + std)
        for i in range(1, 14):
            names.append(f"mfcc_{i}_mean")
            names.append(f"mfcc_{i}_std")
        names += ["delta_mfcc_mean", "delta_mfcc_std"]
        return names

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load *file_path* as mono 16 kHz float32 array.

        Non-WAV formats are converted via pydub if available, otherwise
        librosa's built-in decoder is used.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".mp3", ".m4a", ".ogg", ".flac", ".aac"}:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(str(path))
                audio = audio.set_channels(1).set_frame_rate(self.SAMPLE_RATE)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio.export(tmp.name, format="wav")
                    y, sr = librosa.load(tmp.name, sr=self.SAMPLE_RATE, mono=True)
                    _safe_unlink(tmp.name)
                return y, sr
            except ImportError:
                logger.debug("pydub not available; falling back to librosa decoder.")

        y, sr = librosa.load(str(path), sr=self.SAMPLE_RATE, mono=True)
        return y, sr

    @staticmethod
    def _validate_audio(y: np.ndarray, sr: int) -> None:
        duration = len(y) / sr
        if duration < 0.5:
            raise ValueError(f"Audio too short ({duration:.2f}s); minimum is 0.5 s.")
        if np.max(np.abs(y)) < 1e-5:
            raise ValueError("Audio appears to be silent.")

    @staticmethod
    def _as_wav(file_path: str, y: np.ndarray, sr: int) -> str:
        """Return a WAV path, writing a temp file if *file_path* is not .wav."""
        if Path(file_path).suffix.lower() == ".wav":
            return file_path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            return tmp.name

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _pitch_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """F0-based features including jitter."""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
        except Exception as exc:
            logger.warning("pyin failed: %s; returning zeros.", exc)
            return {k: 0.0 for k in [
                "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
                "f0_median", "voiced_fraction", "jitter",
            ]}

        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        voiced_fraction = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))

        if len(voiced_f0) < 3:
            return {
                "f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0, "f0_max": 0.0,
                "f0_range": 0.0, "f0_median": 0.0,
                "voiced_fraction": voiced_fraction,
                "jitter": 0.0,
            }

        # Jitter: mean absolute difference of consecutive periods / mean period
        periods = 1.0 / voiced_f0
        jitter = float(
            np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-8)
        )

        return {
            "f0_mean": float(np.mean(voiced_f0)),
            "f0_std": float(np.std(voiced_f0)),
            "f0_min": float(np.min(voiced_f0)),
            "f0_max": float(np.max(voiced_f0)),
            "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
            "f0_median": float(np.median(voiced_f0)),
            "voiced_fraction": voiced_fraction,
            "jitter": jitter,
        }

    def _energy_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """RMS energy and shimmer."""
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))
        shimmer = float(
            np.std(rms) / (mean_rms + 1e-8)
        )
        entropy = float(_spectral_entropy(rms))
        return {
            "rms_mean": mean_rms,
            "rms_std": float(np.std(rms)),
            "rms_max": float(np.max(rms)),
            "shimmer": shimmer,
            "energy_entropy": entropy,
        }

    def _mfcc_features(
        self, y: np.ndarray, sr: int, n_mfcc: int = 13
    ) -> dict[str, float]:
        """MFCC (mean + std) and first-order delta statistics."""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfccs)

        feats: dict[str, float] = {}
        for i in range(n_mfcc):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
        feats["delta_mfcc_mean"] = float(np.mean(delta))
        feats["delta_mfcc_std"] = float(np.std(delta))
        return feats

    def _speech_rate_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Speech rate, silence ratio, and pause statistics."""
        hop = 512
        frame_len = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
        threshold = float(np.mean(rms)) * 0.3

        is_speech = rms > threshold
        total_frames = len(is_speech)
        speech_frames = int(np.sum(is_speech))
        duration = total_frames * hop / sr
        speech_duration = speech_frames * hop / sr

        # Count speech bursts (for rate estimation)
        transitions = np.diff(is_speech.astype(np.int8))
        speech_starts = int(np.sum(transitions == 1))

        # Collect pause durations
        pause_durations: list[float] = []
        in_pause = False
        pause_start = 0
        for idx, active in enumerate(is_speech):
            if not active and not in_pause:
                in_pause = True
                pause_start = idx
            elif active and in_pause:
                in_pause = False
                pd = (idx - pause_start) * hop / sr
                if pd >= 0.1:
                    pause_durations.append(pd)

        return {
            "speech_rate": float(speech_starts / max(duration, 0.1)),
            "speech_ratio": float(speech_duration / max(duration, 0.1)),
            "pause_count": float(len(pause_durations)),
            "pause_mean_duration": float(np.mean(pause_durations)) if pause_durations else 0.0,
            "pause_total_duration": float(sum(pause_durations)),
            "duration_seconds": float(duration),
        }

    def _spectral_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Spectral centroid, rolloff, bandwidth, ZCR, and HNR approximation."""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # HNR: log ratio of harmonic to percussive energy
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = float(np.mean(y_harm ** 2))
        perc_energy = float(np.mean(y_perc ** 2))
        hnr = float(np.log10(harm_energy / (perc_energy + 1e-8) + 1e-8))

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "hnr": hnr,
        }

    def _opensmile_features(self, wav_path: str) -> dict[str, float]:
        """Extract eGeMAPS functionals via openSMILE.

        Returns an empty dict if extraction fails.
        """
        if self._smile is None:
            import opensmile
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        try:
            df = self._smile.process_file(wav_path)
            if df.empty:
                return {}
            result: dict[str, float] = {}
            for col in df.columns:
                val = float(df[col].iloc[0])
                if np.isfinite(val):
                    result[f"ems_{col}"] = val
            return result
        except Exception as exc:
            logger.warning("openSMILE extraction failed: %s", exc)
            return {}


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _spectral_entropy(signal: np.ndarray) -> float:
    """Shannon entropy of the power spectrum."""
    power = signal ** 2
    total = float(np.sum(power)) + 1e-8
    prob = power / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob + 1e-12)))


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
