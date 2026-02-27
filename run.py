"""Entry point for the Lie Detector application.

Usage:
    python run.py [--port PORT] [--host HOST] [--share] [--debug]
"""

import argparse
import logging
from pathlib import Path

from detector.audio_features import AudioFeatureExtractor
from detector.text_features import TextFeatureExtractor
from detector.model import LieDetectorModel
from detector.database import Database
from detector.utils import configure_logging
from ui.app import build_app

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "lie_detector.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lie Detector — voice deception analysis")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising components...")

    audio_extractor = AudioFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    model = LieDetectorModel(models_dir=MODELS_DIR)
    database = Database(db_path=DB_PATH)

    # Load or create the model
    model.load()

    logger.info("Building Gradio app...")
    demo = build_app(audio_extractor, text_extractor, model, database)

    logger.info("Launching on %s:%d (share=%s)", args.host, args.port, args.share)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
