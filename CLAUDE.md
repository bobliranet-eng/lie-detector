# Complete the Lie Detector App

The project is 90% built. You need to:

## 1. Create `run.py` (missing)
Entry point that initializes all singletons and launches the Gradio app.
```python
# Pattern: see ui/app.py build_app() signature
from detector.audio_features import AudioFeatureExtractor
from detector.text_features import TextFeatureExtractor  
from detector.model import LieDetectorModel
from detector.database import Database
from ui.app import build_app
```
- Accept --port, --host, --share, --debug CLI args
- BASE_DIR = project root, DATA_DIR = data/, MODELS_DIR = models/, DB_PATH = data/lie_detector.db

## 2. Add MICROPHONE recording to the UI
In `ui/app.py`, the Analysis tab has:
```python
audio_input = gr.Audio(label="Upload Audio Recording", type="filepath", sources=["upload"])
```
Change `sources` to `["upload", "microphone"]` so users can record directly from their laptop mic.

Also in the Training tab, same change for `train_audio_input`.

## 3. Create `tests/test_detector.py`
Basic tests:
- Test AudioFeatureExtractor can be instantiated
- Test LieDetectorModel can be instantiated and has is_trained property
- Test Database can be created with temp path
- Test utils functions work

## 4. Update README.md
Professional README with:
- Project description
- Features list
- Installation instructions (pip install -r requirements.txt)
- Usage (python run.py)
- Screenshots placeholder
- Architecture overview
- How training works
- Supported audio formats
- Multilingual support explanation

## 5. Git commit everything
```bash
git add -A && git commit -m "feat: complete lie detector with trainable model, Gradio UI, and mic recording"
```

## IMPORTANT
- Read existing files before modifying to understand the codebase
- Do NOT rewrite files that already work — only create missing ones or make targeted edits
- The app must be functional
- When completely finished, run: openclaw system event --text "Done: Lie detector complete with mic recording, tests, and README" --mode now
