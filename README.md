# Lie Detector

A voice-based deception detection application that analyzes acoustic features of speech to distinguish truthful statements from deceptive ones.

## Features

- **Acoustic analysis** — extracts pitch, energy, MFCCs, speech rate, spectral features, and more from audio recordings
- **Ensemble ML model** — RandomForest + GradientBoosting + SVM voting classifier with StandardScaler
- **Trainable** — upload your own labeled recordings to retrain the model on real data
- **Microphone recording** — record directly from your laptop mic or upload existing audio files
- **Transcription** — optional speech-to-text via faster-whisper with linguistic feature extraction
- **Analysis history** — all predictions logged to a local SQLite database
- **Dashboard** — feature importances, dataset distribution, and accuracy charts
- **eGeMAPS support** — extended Geneva Minimalistic Acoustic Parameter Set via openSMILE (optional)

## Installation

```bash
pip install -r requirements.txt
```

For optional eGeMAPS features:
```bash
pip install opensmile
```

For optional transcription:
```bash
pip install faster-whisper
```

## Usage

```bash
python run.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `7860` | Port to listen on |
| `--host` | `127.0.0.1` | Host to bind to |
| `--share` | off | Create a public Gradio share link |
| `--debug` | off | Enable debug logging |

Then open `http://127.0.0.1:7860` in your browser.

## Screenshots

_Screenshots placeholder — add images to `docs/screenshots/` and link them here._

## Architecture

```
lie-detector/
├── run.py                   # Entry point
├── detector/
│   ├── audio_features.py    # Acoustic feature extraction (librosa + openSMILE)
│   ├── text_features.py     # Transcription + linguistic features (faster-whisper)
│   ├── model.py             # Ensemble classifier (RF + GB + SVM)
│   ├── database.py          # SQLite persistence
│   └── utils.py             # Shared utilities
├── ui/
│   ├── app.py               # Gradio Blocks application (4 tabs)
│   ├── components.py        # HTML/CSS components
│   └── visualizations.py   # Plotly charts
├── data/                    # SQLite database (auto-created)
├── models/                  # Saved model files (auto-created)
└── tests/
    └── test_detector.py     # pytest test suite
```

## How Training Works

The app ships with a **pre-trained model** built on synthetic data derived from published deception-detection research. It works out of the box but will give a pre-trained data warning on every result.

To train on real data:

1. Go to the **Training** tab
2. Upload a recording and label it as "Truth" or "Lie"
3. Repeat until you have at least 3 samples per class (more is better — aim for 20+)
4. Click **Train Model**

The model re-trains an ensemble of RandomForest, GradientBoosting, and SVM classifiers on your labeled data and saves a versioned `.joblib` file in `models/`. Subsequent analyses will use your custom model and the pre-trained warning will disappear.

## Supported Audio Formats

| Format | Notes |
|--------|-------|
| WAV | Natively supported |
| MP3 | Requires `pydub` + ffmpeg |
| M4A / AAC | Requires `pydub` + ffmpeg |
| OGG | Requires `pydub` + ffmpeg |
| FLAC | Requires `pydub` + ffmpeg |

Install ffmpeg with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

## Multilingual Support

The acoustic feature extraction pipeline is **language-agnostic** — pitch, energy, MFCCs, and spectral features are computed directly from the audio signal without reference to any language model.

Optional transcription (via faster-whisper) supports **99 languages** and automatically detects the language of each recording. Linguistic deception indicators (filler words, hedging language) are currently calibrated for English; contributions for other languages are welcome.

## Running Tests

```bash
pytest tests/
```
