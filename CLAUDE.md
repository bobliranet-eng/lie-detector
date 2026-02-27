# Lie Detector - Voice Deception Analysis System

Build a complete Python application for voice-based lie/truth detection with a trainable model and modern web UI.

## Requirements

### Core Engine (Python)
1. **Audio Analysis** using `librosa` + `openSMILE` (via `opensmile` Python package):
   - Extract features: pitch (F0), jitter, shimmer, speech rate, pauses, energy, MFCCs, formants
   - Normalize features per-speaker
   
2. **Trainable ML Model**:
   - Use `scikit-learn` ensemble (RandomForest + GradientBoosting + SVM voting classifier)
   - Store trained models with `joblib`
   - Users can label recordings as "truth" or "lie" to train/retrain
   - Include a pre-built feature set so the app works out of the box (ship a small pretrained model on synthetic data)
   - Model versioning: save each trained model with timestamp

3. **Speech-to-Text** (optional but nice): use `faster-whisper` for transcription, then run basic linguistic analysis (hedging words, pronoun usage, cognitive load markers) — multilingual support via whisper

### Web UI (Gradio)
Modern, clean interface with:
1. **Upload tab**: Upload audio file (wav, mp3, m4a, ogg, flac) → get truth/lie prediction with percentage + color gauge
   - Green (>70% truth) → Yellow (50-70%) → Red (>70% lie)
   - Show feature breakdown: which indicators triggered
   - Show waveform visualization

2. **Training tab**: 
   - Upload labeled recordings (select "truth" or "lie" for each)
   - Batch upload support
   - Show training progress and model accuracy
   - Button to retrain model
   - Show dataset stats (how many truth/lie samples)

3. **History tab**:
   - List of all analyzed recordings with results
   - SQLite backend for persistence

4. **Dashboard**:
   - Model accuracy stats
   - Feature importance chart
   - Dataset distribution

### Architecture
```
lie-detector/
├── README.md
├── requirements.txt
├── setup.py
├── run.py                  # Entry point
├── detector/
│   ├── __init__.py
│   ├── audio_features.py   # librosa + opensmile feature extraction
│   ├── text_features.py    # linguistic analysis via whisper
│   ├── model.py            # ML model training + prediction
│   ├── database.py         # SQLite storage
│   └── utils.py
├── ui/
│   ├── __init__.py
│   ├── app.py              # Gradio app
│   ├── components.py       # UI components
│   └── visualizations.py   # Charts, gauges, waveforms
├── models/                 # Saved trained models
├── data/                   # Training data storage
└── tests/
    └── test_detector.py
```

### Technical Constraints
- Python 3.10+
- All text/UI should be in English (the ANALYSIS works on any language audio)
- Use type hints everywhere
- Include proper error handling
- Make it pip-installable
- The app should work even without a trained model (show "untrained" state and guide user to train)

### Key Libraries
- `librosa` - audio analysis
- `opensmile` - professional audio features
- `scikit-learn` - ML models
- `gradio` - web UI
- `faster-whisper` - speech to text (multilingual)
- `plotly` - charts and visualizations
- `numpy`, `pandas` - data handling
- `joblib` - model serialization
- `pydub` - audio format conversion

### Run
```bash
pip install -r requirements.txt
python run.py
```
Opens Gradio UI on localhost.

## IMPORTANT
- Write ALL files completely, no placeholders
- Make it production-quality code
- Include comprehensive error handling
- The UI must look modern and professional
- Test that imports work
- When completely finished, run: openclaw system event --text "Done: Built lie detector with trainable model, audio analysis (librosa+opensmile), and Gradio UI" --mode now
