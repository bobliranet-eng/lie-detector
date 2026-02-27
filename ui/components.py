"""Reusable Gradio UI components and constants for the Lie Detector.

Contains CSS, HTML snippets, and factory functions that are assembled
into the full interface in :mod:`ui.app`.
"""

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ─── Global ─────────────────────────────────────────────────────────── */
body, .gradio-container {
    background-color: #0f0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ─── Header ─────────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1e1e3f 0%, #2d1b69 50%, #1a1a2e 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 16px;
    border: 1px solid rgba(139, 92, 246, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.app-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
}

/* ─── Result cards ────────────────────────────────────────────────────── */
.result-truth {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.result-lie {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.result-uncertain {
    background: linear-gradient(135deg, #422006, #713f12);
    border: 1px solid #eab308;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 4px 0;
}

.result-subtext {
    font-size: 0.9rem;
    opacity: 0.8;
    margin: 0;
}

/* ─── Info banner ─────────────────────────────────────────────────────── */
.info-banner {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #93c5fd;
}

.warning-banner {
    background: rgba(234, 179, 8, 0.12);
    border: 1px solid rgba(234, 179, 8, 0.4);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #fde68a;
}

/* ─── Tabs ────────────────────────────────────────────────────────────── */
.tab-nav {
    background-color: #1e1e2e !important;
    border-bottom: 1px solid #2d2d4e !important;
}

.tab-nav button {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}

.tab-nav button.selected {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
    background-color: transparent !important;
}

/* ─── Buttons ─────────────────────────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

.btn-primary:hover { opacity: 0.9 !important; }

.btn-danger {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
}

.btn-success {
    background: linear-gradient(135deg, #15803d, #16a34a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
}

/* ─── Cards / panels ──────────────────────────────────────────────────── */
.panel {
    background-color: #1e1e2e !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* ─── Stats box ───────────────────────────────────────────────────────── */
.stat-box {
    background: #1e1e2e;
    border: 1px solid #2d2d4e;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #a78bfa;
}

.stat-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ─── Input / label overrides ─────────────────────────────────────────── */
label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

.gr-textbox, textarea, input[type=text] {
    background-color: #1e1e2e !important;
    border: 1px solid #2d2d4e !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ─── Dataframe ───────────────────────────────────────────────────────── */
.gr-dataframe table {
    background-color: #1e1e2e !important;
    color: #e2e8f0 !important;
}

.gr-dataframe th {
    background-color: #2d2d4e !important;
    color: #a78bfa !important;
    font-weight: 600 !important;
}

.gr-dataframe td {
    border-color: #2d2d4e !important;
}
"""


# ---------------------------------------------------------------------------
# HTML snippets
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div class="app-header">
  <p class="app-title">🎙️ Lie Detector</p>
  <p class="app-subtitle">
    AI-powered voice deception analysis &mdash; pitch, jitter, shimmer, speech patterns &amp; more
  </p>
</div>
"""

PRETRAINED_WARNING_HTML = """
<div class="warning-banner">
  ⚠️ <strong>Using pre-trained synthetic model.</strong>
  For accurate results, add your own labelled recordings in the <em>Training</em> tab
  and click <em>Train Model</em>.
</div>
"""

ANALYSIS_INFO_HTML = """
<div class="info-banner">
  📂 Upload a voice recording (WAV, MP3, M4A, OGG, FLAC) and click
  <strong>Analyse</strong> to detect deception indicators.
  The analysis is language-independent.
</div>
"""

TRAINING_INFO_HTML = """
<div class="info-banner">
  🏋️ Upload recordings and label them as <strong>Truth</strong> or
  <strong>Lie</strong>.  Once you have at least 3 samples of each, click
  <strong>Train Model</strong> to personalise the detector.
</div>
"""


def result_html(prediction: str, truth_prob: float, lie_prob: float) -> str:
    """Build a styled HTML result card.

    Args:
        prediction: "Truth", "Lie", or "Uncertain".
        truth_prob: Truth probability 0–1.
        lie_prob: Lie probability 0–1.

    Returns:
        HTML string.
    """
    emoji = {"Truth": "✅", "Lie": "❌", "Uncertain": "⚠️"}.get(prediction, "❔")
    css_class = {
        "Truth": "result-truth",
        "Lie": "result-lie",
        "Uncertain": "result-uncertain",
    }.get(prediction, "result-uncertain")

    color = {
        "Truth": "#22c55e",
        "Lie": "#ef4444",
        "Uncertain": "#eab308",
    }.get(prediction, "#eab308")

    return f"""
<div class="{css_class}">
  <p class="result-label" style="color:{color};">{emoji} {prediction}</p>
  <p class="result-subtext">
    Truth: {truth_prob*100:.1f}% &nbsp;|&nbsp; Lie: {lie_prob*100:.1f}%
  </p>
</div>
"""


def stats_html(
    model_trained: bool,
    is_synthetic: bool,
    n_truth: int,
    n_lie: int,
    accuracy: float,
    cv_accuracy: float,
) -> str:
    """Build a model stats summary HTML block.

    Args:
        model_trained: Whether any model is loaded.
        is_synthetic: Whether the current model was trained on synthetic data.
        n_truth: Number of truth training samples used.
        n_lie: Number of lie training samples used.
        accuracy: Training set accuracy (0–1).
        cv_accuracy: Cross-validated accuracy (0–1, may be 0 if not computed).

    Returns:
        HTML string.
    """
    status = "Synthetic (pre-trained)" if is_synthetic else "Custom trained"
    acc_str = f"{accuracy*100:.1f}%" if model_trained else "—"
    cv_str = f"{cv_accuracy*100:.1f}%" if (model_trained and cv_accuracy > 0) else "—"

    return f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;">
  <div class="stat-box">
    <div class="stat-number">{n_truth + n_lie}</div>
    <div class="stat-label">Training Samples</div>
  </div>
  <div class="stat-box">
    <div class="stat-number" style="color:#22c55e;">{n_truth}</div>
    <div class="stat-label">Truth</div>
  </div>
  <div class="stat-box">
    <div class="stat-number" style="color:#ef4444;">{n_lie}</div>
    <div class="stat-label">Lie</div>
  </div>
  <div class="stat-box">
    <div class="stat-number" style="color:#60a5fa;">{acc_str}</div>
    <div class="stat-label">Train Accuracy</div>
  </div>
  <div class="stat-box">
    <div class="stat-number" style="color:#34d399;">{cv_str}</div>
    <div class="stat-label">CV Accuracy</div>
  </div>
  <div class="stat-box">
    <div class="stat-number" style="font-size:1.1rem;color:#a78bfa;">{status}</div>
    <div class="stat-label">Model Type</div>
  </div>
</div>
"""
