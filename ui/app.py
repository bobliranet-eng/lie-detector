"""Main Gradio application for the Lie Detector.

Four tabs:
  1. Analysis  – upload audio → prediction + waveform + indicators
  2. Training  – add labelled samples, retrain model
  3. History   – browsable SQLite log of all analyses
  4. Dashboard – model stats, feature importances, dataset distribution
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
import pandas as pd

from detector.audio_features import AudioFeatureExtractor
from detector.text_features import TextFeatureExtractor
from detector.model import LieDetectorModel
from detector.database import Database
from detector.utils import format_duration, get_prediction_label

from ui.components import (
    CUSTOM_CSS,
    HEADER_HTML,
    PRETRAINED_WARNING_HTML,
    ANALYSIS_INFO_HTML,
    TRAINING_INFO_HTML,
    result_html,
    stats_html,
)
from ui.visualizations import (
    create_gauge,
    create_waveform,
    create_feature_breakdown,
    create_feature_importance_chart,
    create_dataset_distribution,
    create_accuracy_chart,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (initialised in build_app)
# ---------------------------------------------------------------------------

_audio_extractor: Optional[AudioFeatureExtractor] = None
_text_extractor: Optional[TextFeatureExtractor] = None
_model: Optional[LieDetectorModel] = None
_db: Optional[Database] = None


def _get_singletons() -> tuple[
    AudioFeatureExtractor, TextFeatureExtractor, LieDetectorModel, Database
]:
    assert _audio_extractor and _text_extractor and _model and _db
    return _audio_extractor, _text_extractor, _model, _db


# ---------------------------------------------------------------------------
# Tab 1 — Analysis
# ---------------------------------------------------------------------------

def analyse_audio(
    file_path: Optional[str],
    enable_transcription: bool,
) -> tuple:
    """Run the full analysis pipeline on *file_path*.

    Returns a tuple matching the Gradio output components:
    (result_html, gauge_fig, waveform_fig, breakdown_fig,
     transcript_text, status_text, pretrained_warning)
    """
    audio_ext, text_ext, model, db = _get_singletons()
    empty_gauge = create_gauge(0.5)
    empty_waveform = create_waveform(np.zeros(1000), 16_000)
    empty_breakdown = create_feature_breakdown({}, 0.5)

    if file_path is None:
        return (
            "<div class='warning-banner'>⚠️ Please upload an audio file first.</div>",
            empty_gauge,
            empty_waveform,
            empty_breakdown,
            "",
            "No file uploaded.",
            "",
        )

    try:
        # ------------------------------------------------------------------
        # Feature extraction
        # ------------------------------------------------------------------
        logger.info("Extracting features from %s", file_path)
        features = audio_ext.extract_all_features(file_path)

        # Optional transcription + linguistic features
        transcript = ""
        if enable_transcription:
            try:
                text_result = text_ext.extract_features(file_path)
                transcript = text_result.pop("transcript", "")
                features.update(text_result)
            except Exception as exc:
                logger.warning("Text analysis failed: %s", exc)

        # ------------------------------------------------------------------
        # Model prediction
        # ------------------------------------------------------------------
        pred = model.predict(features)
        truth_prob = pred["truth_probability"]
        lie_prob = pred["lie_probability"]
        prediction = pred["prediction"]
        confidence = pred["confidence"]
        feature_analysis = pred.get("feature_analysis", {})
        is_synthetic = pred.get("is_pretrained", True)

        # ------------------------------------------------------------------
        # Persist to DB
        # ------------------------------------------------------------------
        filename = Path(file_path).name
        duration = features.get("duration_seconds")
        db.save_analysis(
            filename=filename,
            truth_prob=truth_prob,
            lie_prob=lie_prob,
            prediction=prediction,
            confidence=confidence,
            features=features,
            transcript=transcript or None,
            duration_sec=duration,
            model_version=model.metadata.get("trained_on"),
        )

        # ------------------------------------------------------------------
        # Build visualisations
        # ------------------------------------------------------------------
        gauge = create_gauge(truth_prob)

        # Load audio for waveform (use librosa, already loaded internally)
        try:
            import librosa
            y, sr = librosa.load(file_path, sr=16_000, mono=True)
            waveform = create_waveform(y, sr)
        except Exception:
            waveform = empty_waveform

        breakdown = create_feature_breakdown(feature_analysis, truth_prob)
        res_html = result_html(prediction, truth_prob, lie_prob)
        warning = PRETRAINED_WARNING_HTML if is_synthetic else ""

        dur_str = format_duration(duration) if duration else "—"
        status = (
            f"✅ Analysis complete — {filename} ({dur_str}) | "
            f"Confidence: {confidence*100:.1f}%"
        )

        return (res_html, gauge, waveform, breakdown, transcript, status, warning)

    except ValueError as exc:
        msg = f"<div class='warning-banner'>⚠️ {exc}</div>"
        return (msg, empty_gauge, empty_waveform, empty_breakdown, "", str(exc), "")
    except Exception as exc:
        logger.exception("Analysis failed.")
        msg = f"<div class='warning-banner'>❌ Analysis failed: {exc}</div>"
        return (msg, empty_gauge, empty_waveform, empty_breakdown, "", f"Error: {exc}", "")


# ---------------------------------------------------------------------------
# Tab 2 — Training
# ---------------------------------------------------------------------------

def add_training_sample(
    file_path: Optional[str],
    label_choice: str,
) -> tuple[str, str]:
    """Extract features from *file_path* and add to the training set.

    Returns:
        (status_message, updated_stats_html)
    """
    audio_ext, _, _, db = _get_singletons()

    if file_path is None:
        return "⚠️ No file uploaded.", _training_stats_html()

    try:
        features = audio_ext.extract_all_features(file_path)
        label = 0 if label_choice.lower() == "truth" else 1
        filename = Path(file_path).name
        duration = features.get("duration_seconds")
        db.add_training_sample(
            filename=filename,
            label=label,
            features=features,
            duration_sec=duration,
        )
        stats = db.get_training_stats()
        status = (
            f"✅ Added <b>{filename}</b> as <b>{label_choice}</b> "
            f"(total: {stats['truth']} truth, {stats['lie']} lie)"
        )
        return status, _training_stats_html()
    except Exception as exc:
        logger.exception("Failed to add training sample.")
        return f"❌ Error: {exc}", _training_stats_html()


def train_model() -> tuple[str, str]:
    """Retrain the model on all stored training samples.

    Returns:
        (status_message, updated_stats_html)
    """
    _, _, model, db = _get_singletons()

    samples = db.get_training_samples()
    if not samples:
        return "⚠️ No training samples found. Add some first.", _training_stats_html()

    features_list = [s["features"] for s in samples]
    labels = [s["label"] for s in samples]

    truth_n = labels.count(0)
    lie_n = labels.count(1)

    if truth_n < 3 or lie_n < 3:
        return (
            f"⚠️ Need ≥3 samples per class. Have: {truth_n} truth, {lie_n} lie.",
            _training_stats_html(),
        )

    try:
        result = model.train(features_list, labels)
        acc = result["accuracy"] * 100
        cv_acc = result["cv_accuracy"] * 100
        status = (
            f"✅ Model trained on {len(labels)} samples — "
            f"Train acc: {acc:.1f}%"
            + (f"  |  CV acc: {cv_acc:.1f}%" if cv_acc > 0 else "")
        )
        return status, _training_stats_html()
    except Exception as exc:
        logger.exception("Training failed.")
        return f"❌ Training failed: {exc}", _training_stats_html()


def clear_training_data() -> tuple[str, str]:
    """Delete all training samples from the database."""
    _, _, _, db = _get_singletons()
    n = db.clear_training_samples()
    return f"🗑️ Cleared {n} training samples.", _training_stats_html()


def _training_stats_html() -> str:
    _, _, model, db = _get_singletons()
    stats = db.get_training_stats()
    mstats = model.get_stats()
    return stats_html(
        model_trained=mstats["is_trained"],
        is_synthetic=mstats["synthetic"],
        n_truth=stats["truth"],
        n_lie=stats["lie"],
        accuracy=mstats["train_accuracy"],
        cv_accuracy=mstats["cv_accuracy"],
    )


def get_dataset_distribution_fig() -> Any:
    _, _, _, db = _get_singletons()
    stats = db.get_training_stats()
    return create_dataset_distribution(stats["truth"], stats["lie"])


# ---------------------------------------------------------------------------
# Tab 3 — History
# ---------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    """Return the analysis history as a DataFrame for gr.Dataframe."""
    _, _, _, db = _get_singletons()
    rows = db.get_analyses(limit=200)
    if not rows:
        return pd.DataFrame(
            columns=["ID", "File", "Time", "Prediction", "Truth %", "Lie %",
                     "Confidence %", "Duration", "Transcript"]
        )
    records = []
    for r in rows:
        records.append({
            "ID": r["id"],
            "File": r["filename"],
            "Time": r["timestamp"],
            "Prediction": r["prediction"],
            "Truth %": f"{r['truth_prob']*100:.1f}",
            "Lie %": f"{r['lie_prob']*100:.1f}",
            "Confidence %": f"{r['confidence']*100:.1f}",
            "Duration": format_duration(r["duration_sec"]) if r["duration_sec"] else "—",
            "Transcript": (r["transcript"] or "")[:80],
        })
    return pd.DataFrame(records)


def clear_history() -> tuple[pd.DataFrame, str]:
    _, _, _, db = _get_singletons()
    n = db.clear_analyses()
    return load_history(), f"🗑️ Cleared {n} analysis records."


# ---------------------------------------------------------------------------
# Tab 4 — Dashboard
# ---------------------------------------------------------------------------

def load_dashboard() -> tuple[Any, Any, Any, str]:
    """Refresh dashboard figures and stats.

    Returns:
        (feature_importance_fig, dataset_distribution_fig, accuracy_chart_fig, stats_html)
    """
    _, _, model, db = _get_singletons()
    importance = model.get_feature_importance()
    fi_fig = create_feature_importance_chart(importance, top_n=20)

    db_stats = db.get_training_stats()
    dist_fig = create_dataset_distribution(db_stats["truth"], db_stats["lie"])

    # Accuracy history: just show current model for now
    mstats = model.get_stats()
    history = [mstats] if mstats["is_trained"] else []
    acc_fig = create_accuracy_chart(history)

    s = mstats
    html = stats_html(
        model_trained=s["is_trained"],
        is_synthetic=s["synthetic"],
        n_truth=s["n_truth"],
        n_lie=s["n_lie"],
        accuracy=s["train_accuracy"],
        cv_accuracy=s["cv_accuracy"],
    )
    return fi_fig, dist_fig, acc_fig, html


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app(
    audio_extractor: AudioFeatureExtractor,
    text_extractor: TextFeatureExtractor,
    model: LieDetectorModel,
    database: Database,
) -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    Call :func:`build_app` once during startup, then ``.launch()`` the result.
    """
    global _audio_extractor, _text_extractor, _model, _db
    _audio_extractor = audio_extractor
    _text_extractor = text_extractor
    _model = model
    _db = database

    with gr.Blocks(css=CUSTOM_CSS, title="Lie Detector") as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        with gr.Tabs() as tabs:

            # ── Tab 1: Analysis ─────────────────────────────────────────
            with gr.Tab("🔍 Analysis", id="analysis"):
                gr.HTML(ANALYSIS_INFO_HTML)

                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            label="Upload Audio Recording",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        with gr.Row():
                            transcribe_toggle = gr.Checkbox(
                                label="Enable transcription (requires faster-whisper)",
                                value=False,
                            )
                        analyse_btn = gr.Button(
                            "🎙️ Analyse Recording",
                            variant="primary",
                            elem_classes=["btn-primary"],
                        )
                        analysis_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            container=False,
                            placeholder="Upload a file and click Analyse…",
                        )

                    with gr.Column(scale=3):
                        pretrained_warning = gr.HTML("")
                        result_display = gr.HTML(
                            "<div class='info-banner'>Result will appear here…</div>"
                        )
                        gauge_plot = gr.Plot(
                            label="Truth/Lie Score",
                            value=create_gauge(0.5),
                        )

                with gr.Row():
                    waveform_plot = gr.Plot(
                        label="Waveform",
                        value=create_waveform(np.zeros(1000), 16_000),
                    )

                with gr.Row():
                    breakdown_plot = gr.Plot(
                        label="Deception Indicators",
                        value=create_feature_breakdown({}, 0.5),
                    )

                with gr.Accordion("📝 Transcript", open=False):
                    transcript_box = gr.Textbox(
                        label="Speech Transcript",
                        interactive=False,
                        lines=4,
                        placeholder="Enable transcription to see text here…",
                    )

                analyse_btn.click(
                    fn=analyse_audio,
                    inputs=[audio_input, transcribe_toggle],
                    outputs=[
                        result_display,
                        gauge_plot,
                        waveform_plot,
                        breakdown_plot,
                        transcript_box,
                        analysis_status,
                        pretrained_warning,
                    ],
                )

            # ── Tab 2: Training ──────────────────────────────────────────
            with gr.Tab("🏋️ Training", id="training"):
                gr.HTML(TRAINING_INFO_HTML)

                with gr.Row():
                    with gr.Column():
                        train_audio_input = gr.Audio(
                            label="Upload Labelled Recording",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        label_radio = gr.Radio(
                            choices=["Truth", "Lie"],
                            value="Truth",
                            label="Label this recording as:",
                        )
                        add_btn = gr.Button(
                            "➕ Add to Dataset",
                            variant="secondary",
                        )
                        train_status = gr.HTML("")

                    with gr.Column():
                        train_stats = gr.HTML(_training_stats_html())
                        dist_fig = gr.Plot(
                            label="Dataset Distribution",
                            value=get_dataset_distribution_fig(),
                        )

                with gr.Row():
                    train_btn = gr.Button(
                        "🚀 Train Model",
                        variant="primary",
                        elem_classes=["btn-primary"],
                    )
                    clear_data_btn = gr.Button(
                        "🗑️ Clear Training Data",
                        variant="stop",
                        elem_classes=["btn-danger"],
                    )

                def _add_sample_and_refresh(fp, lbl):
                    status, stats = add_training_sample(fp, lbl)
                    dist = get_dataset_distribution_fig()
                    return f"<div class='info-banner'>{status}</div>", stats, dist

                add_btn.click(
                    fn=_add_sample_and_refresh,
                    inputs=[train_audio_input, label_radio],
                    outputs=[train_status, train_stats, dist_fig],
                )

                def _train_and_refresh():
                    status, stats = train_model()
                    return f"<div class='info-banner'>{status}</div>", stats

                train_btn.click(
                    fn=_train_and_refresh,
                    outputs=[train_status, train_stats],
                )

                def _clear_and_refresh():
                    status, stats = clear_training_data()
                    dist = get_dataset_distribution_fig()
                    return f"<div class='info-banner'>{status}</div>", stats, dist

                clear_data_btn.click(
                    fn=_clear_and_refresh,
                    outputs=[train_status, train_stats, dist_fig],
                )

            # ── Tab 3: History ───────────────────────────────────────────
            with gr.Tab("📜 History", id="history"):
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
                    clear_history_btn = gr.Button(
                        "🗑️ Clear History",
                        variant="stop",
                        elem_classes=["btn-danger"],
                    )
                    history_status = gr.Textbox(
                        label="",
                        interactive=False,
                        container=False,
                        placeholder="",
                    )

                history_table = gr.Dataframe(
                    value=load_history(),
                    label="Analysis History",
                    interactive=False,
                    wrap=True,
                )

                refresh_history_btn.click(
                    fn=load_history,
                    outputs=history_table,
                )
                clear_history_btn.click(
                    fn=clear_history,
                    outputs=[history_table, history_status],
                )

            # ── Tab 4: Dashboard ─────────────────────────────────────────
            with gr.Tab("📊 Dashboard", id="dashboard"):
                refresh_dash_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                dash_stats = gr.HTML(load_dashboard()[3])

                with gr.Row():
                    dash_dist_fig = gr.Plot(
                        label="Dataset Distribution",
                        value=load_dashboard()[1],
                    )
                    dash_acc_fig = gr.Plot(
                        label="Accuracy History",
                        value=load_dashboard()[2],
                    )

                dash_fi_fig = gr.Plot(
                    label="Feature Importances",
                    value=load_dashboard()[0],
                )

                def _refresh_dashboard():
                    fi, dist, acc, html = load_dashboard()
                    return html, dist, acc, fi

                refresh_dash_btn.click(
                    fn=_refresh_dashboard,
                    outputs=[dash_stats, dash_dist_fig, dash_acc_fig, dash_fi_fig],
                )

    return demo
