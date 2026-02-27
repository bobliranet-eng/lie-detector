"""Linguistic feature extraction via faster-whisper transcription.

Transcribes audio to text (multilingual) and computes:
  - Hedging word density
  - Cognitive load markers (filler words)
  - First-person pronoun usage
  - Negation count
  - Lexical diversity (type-token ratio)
  - Sentence complexity proxies

faster-whisper is loaded lazily on first use to avoid slow startup.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word lists  (English-centric; the underlying features are language-agnostic)
# ---------------------------------------------------------------------------

_HEDGING_WORDS: set[str] = {
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "sort", "kind", "guess", "think", "suppose", "seem", "appear",
    "somewhat", "rather", "fairly", "quite", "generally", "typically",
    "usually", "sometimes", "often", "occasionally",
}

_FILLER_WORDS: set[str] = {
    "um", "uh", "er", "ah", "hmm", "like", "basically", "literally",
    "actually", "honestly", "truthfully", "frankly", "obviously",
    "you", "know", "mean", "right", "okay",
}

_FIRST_PERSON_SINGULAR: set[str] = {
    "i", "me", "my", "myself", "mine",
}

_NEGATIONS: set[str] = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nor", "nowhere", "dont", "cant", "wont", "isnt", "arent",
    "wasnt", "werent", "havent", "hasnt", "hadnt", "didnt",
    "doesnt", "shouldnt", "wouldnt", "couldnt", "mustn",
}

_COGNITIVE_QUALIFIERS: set[str] = {
    "remember", "think", "believe", "feel", "know", "recall",
    "forget", "understand", "realize", "wonder",
}


class TextFeatureExtractor:
    """Transcribe audio and derive linguistic deception markers.

    Example::

        extractor = TextFeatureExtractor()
        result = extractor.extract_features("interview.wav")
        # result["transcript"] – full transcription
        # result["hedging_ratio"] – fraction of hedging words
    """

    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        """Initialise the extractor.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large").
                        "base" gives a good accuracy/speed tradeoff.
            device: "cpu" or "cuda".
        """
        self._model_size = model_size
        self._device = device
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if faster-whisper can be imported."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(self, file_path: str, language: Optional[str] = None) -> str:
        """Transcribe *file_path* to text.

        Args:
            file_path: Path to the audio file.
            language: BCP-47 language code (e.g. "en", "es").  When ``None``
                      Whisper auto-detects the language.

        Returns:
            Full transcript as a single string, or "" on failure.
        """
        model = self._get_model()
        if model is None:
            return ""
        try:
            segments, _ = model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments)
        except Exception as exc:
            logger.warning("Transcription failed for %s: %s", file_path, exc)
            return ""

    def analyze_text(self, text: str) -> dict[str, float]:
        """Compute linguistic features from a transcript string.

        Args:
            text: Transcribed speech.

        Returns:
            Dict of feature name → float.
        """
        if not text.strip():
            return self._empty_features()

        words = re.findall(r"\b\w+\b", text.lower())
        n = len(words)
        if n == 0:
            return self._empty_features()

        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)

        # Hedging
        hedging_count = sum(1 for w in words if w in _HEDGING_WORDS)
        # Filler words (cognitive load)
        filler_count = sum(1 for w in words if w in _FILLER_WORDS)
        # First-person singular
        fp_count = sum(1 for w in words if w in _FIRST_PERSON_SINGULAR)
        # Negations
        neg_count = sum(1 for w in words if w in _NEGATIONS)
        # Cognitive qualifiers
        cog_count = sum(1 for w in words if w in _COGNITIVE_QUALIFIERS)
        # Unique word ratio (type-token ratio)
        ttr = len(set(words)) / n
        # Average sentence length
        avg_sent_len = n / n_sentences
        # Long-word ratio (>7 chars → complex vocabulary)
        long_words = sum(1 for w in words if len(w) > 7)

        return {
            "text_word_count": float(n),
            "text_sentence_count": float(n_sentences),
            "text_hedging_ratio": hedging_count / n,
            "text_filler_ratio": filler_count / n,
            "text_first_person_ratio": fp_count / n,
            "text_negation_ratio": neg_count / n,
            "text_cognitive_qualifier_ratio": cog_count / n,
            "text_type_token_ratio": ttr,
            "text_avg_sentence_length": avg_sent_len,
            "text_long_word_ratio": long_words / n,
        }

    def extract_features(
        self, file_path: str, language: Optional[str] = None
    ) -> dict[str, float | str]:
        """Transcribe *file_path* and return both transcript and features.

        Args:
            file_path: Path to the audio file.
            language: Optional BCP-47 language code.

        Returns:
            Dict with ``transcript`` (str) and all linguistic feature keys.
        """
        transcript = self.transcribe(file_path, language=language)
        features = self.analyze_text(transcript)
        return {"transcript": transcript, **features}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily load the Whisper model."""
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading Whisper '%s' on %s (first-time download may take a moment)...",
                self._model_size,
                self._device,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type="int8",
            )
            logger.info("Whisper model loaded.")
            return self._model
        except ImportError:
            logger.warning(
                "faster-whisper not installed; transcription disabled. "
                "Install with: pip install faster-whisper"
            )
            return None
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            return None

    @staticmethod
    def _empty_features() -> dict[str, float]:
        return {
            "text_word_count": 0.0,
            "text_sentence_count": 0.0,
            "text_hedging_ratio": 0.0,
            "text_filler_ratio": 0.0,
            "text_first_person_ratio": 0.0,
            "text_negation_ratio": 0.0,
            "text_cognitive_qualifier_ratio": 0.0,
            "text_type_token_ratio": 0.0,
            "text_avg_sentence_length": 0.0,
            "text_long_word_ratio": 0.0,
        }
