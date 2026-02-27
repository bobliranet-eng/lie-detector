"""Plotly-based chart builders for the Lie Detector UI.

All functions return :class:`plotly.graph_objects.Figure` objects that
can be passed directly to ``gr.Plot`` components.
"""

import logging
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_GREEN = "#22c55e"
_YELLOW = "#eab308"
_ORANGE = "#f97316"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_DARK_BG = "#1e1e2e"
_PANEL_BG = "#2a2a3e"
_TEXT = "#e2e8f0"

_PLOTLY_TEMPLATE = "plotly_dark"


def _truth_color(truth_prob: float) -> str:
    if truth_prob >= 0.70:
        return _GREEN
    if truth_prob >= 0.50:
        return _YELLOW
    if truth_prob >= 0.30:
        return _ORANGE
    return _RED


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def create_gauge(truth_probability: float) -> go.Figure:
    """Create a coloured gauge showing truth probability.

    Args:
        truth_probability: Float 0–1 (1 = certain truth).

    Returns:
        Plotly Figure containing the gauge indicator.
    """
    pct = truth_probability * 100
    color = _truth_color(truth_probability)
    lie_pct = (1 - truth_probability) * 100

    if truth_probability >= 0.70:
        verdict = "TRUTH"
        verdict_color = _GREEN
    elif truth_probability <= 0.30:
        verdict = "LIE"
        verdict_color = _RED
    else:
        verdict = "UNCERTAIN"
        verdict_color = _YELLOW

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 42, "color": color},
        },
        delta={
            "reference": 50,
            "increasing": {"color": _GREEN},
            "decreasing": {"color": _RED},
        },
        title={
            "text": f"<b>{verdict}</b><br><span style='font-size:14px;color:{_TEXT}'>Truth Score</span>",
            "font": {"size": 22, "color": verdict_color},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": _TEXT,
                "tickfont": {"color": _TEXT},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": _PANEL_BG,
            "borderwidth": 2,
            "bordercolor": _TEXT,
            "steps": [
                {"range": [0, 30], "color": "#3d1515"},   # Dark red zone
                {"range": [30, 50], "color": "#3d2b15"},  # Dark orange zone
                {"range": [50, 70], "color": "#3d3415"},  # Dark yellow zone
                {"range": [70, 100], "color": "#153d20"},  # Dark green zone
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))

    # Add lie probability annotation
    fig.add_annotation(
        x=0.5, y=-0.1,
        text=f"Lie Score: <b>{lie_pct:.1f}%</b>",
        showarrow=False,
        font={"size": 16, "color": _RED if lie_pct > 70 else _TEXT},
        xref="paper", yref="paper",
    )

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        height=320,
        margin={"t": 80, "b": 50, "l": 30, "r": 30},
    )
    return fig


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def create_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    max_points: int = 4000,
) -> go.Figure:
    """Plot the audio waveform.

    Args:
        audio_data: 1-D float32 waveform array.
        sample_rate: Samples per second.
        max_points: Maximum number of points to plot (downsampled for speed).

    Returns:
        Plotly Figure.
    """
    n = len(audio_data)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Waveform (no audio)",
            template=_PLOTLY_TEMPLATE,
            paper_bgcolor=_DARK_BG,
        )
        return fig

    # Downsample if needed
    if n > max_points:
        step = n // max_points
        y = audio_data[::step]
        t = np.linspace(0, n / sample_rate, len(y))
    else:
        y = audio_data
        t = np.linspace(0, n / sample_rate, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=y,
        mode="lines",
        line={"color": _BLUE, "width": 0.8},
        name="Waveform",
        hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": "Audio Waveform", "font": {"color": _TEXT}},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_PANEL_BG,
        height=200,
        margin={"t": 40, "b": 40, "l": 50, "r": 20},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Feature breakdown
# ---------------------------------------------------------------------------

def create_feature_breakdown(
    feature_analysis: dict[str, str],
    truth_prob: float,
) -> go.Figure:
    """Horizontal bar chart showing deception indicator levels.

    Args:
        feature_analysis: Dict from :meth:`LieDetectorModel.predict`
                          (e.g. {"Pitch Variability": "High", ...}).
        truth_prob: Overall truth probability (used for colouring).

    Returns:
        Plotly Figure.
    """
    if not feature_analysis:
        fig = go.Figure()
        fig.update_layout(
            title="No feature data",
            template=_PLOTLY_TEMPLATE,
            paper_bgcolor=_DARK_BG,
        )
        return fig

    labels = list(feature_analysis.keys())
    values_str = list(feature_analysis.values())

    # Convert Low/Normal/High to numeric and colour
    score_map = {"Low": 0.15, "Normal": 0.5, "High": 0.85}
    lie_indicator = {
        "Pitch Variability": True,
        "Voice Tremor (Jitter)": True,
        "Amplitude Instability (Shimmer)": True,
        "Pause Frequency": True,
        "Pause Duration": True,
        "Speaking Ratio": False,  # High is truth indicator
        "Voice Clarity (HNR)": False,  # High is truth indicator
    }

    colors = []
    scores = []
    for lbl, val_str in zip(labels, values_str):
        score = score_map.get(val_str, 0.5)
        scores.append(score)
        is_lie_ind = lie_indicator.get(lbl, True)
        if val_str == "Normal":
            colors.append(_BLUE)
        elif (is_lie_ind and val_str == "High") or (not is_lie_ind and val_str == "Low"):
            colors.append(_RED)
        else:
            colors.append(_GREEN)

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=values_str,
        textposition="outside",
        textfont={"color": _TEXT, "size": 12},
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": "Deception Indicators", "font": {"color": _TEXT}},
        xaxis={
            "range": [0, 1.2],
            "showticklabels": False,
            "showgrid": False,
        },
        yaxis={"tickfont": {"color": _TEXT, "size": 11}},
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_PANEL_BG,
        height=max(250, len(labels) * 38),
        margin={"t": 40, "b": 20, "l": 230, "r": 80},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------------------------

def create_feature_importance_chart(
    importances: dict[str, float],
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of the top-N feature importances.

    Args:
        importances: Dict mapping feature name → importance score.
        top_n: Number of features to display.

    Returns:
        Plotly Figure.
    """
    if not importances:
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importances (no model)",
            template=_PLOTLY_TEMPLATE,
            paper_bgcolor=_DARK_BG,
        )
        return fig

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:top_n]

    names = [item[0] for item in reversed(top)]
    values = [item[1] for item in reversed(top)]

    # Nice name mapping
    name_labels = [_pretty_feature_name(n) for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=name_labels,
        orientation="h",
        marker=dict(
            color=values,
            colorscale="Viridis",
            showscale=True,
            colorbar={"title": "Importance", "tickfont": {"color": _TEXT}},
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": f"Top {top_n} Feature Importances", "font": {"color": _TEXT}},
        xaxis_title="Importance Score",
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_PANEL_BG,
        height=max(350, top_n * 22),
        margin={"t": 50, "b": 50, "l": 200, "r": 80},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Dataset distribution
# ---------------------------------------------------------------------------

def create_dataset_distribution(
    truth_count: int,
    lie_count: int,
) -> go.Figure:
    """Pie chart showing truth vs lie sample distribution.

    Args:
        truth_count: Number of truth training samples.
        lie_count: Number of lie training samples.

    Returns:
        Plotly Figure.
    """
    total = truth_count + lie_count
    if total == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No training data yet",
            template=_PLOTLY_TEMPLATE,
            paper_bgcolor=_DARK_BG,
        )
        return fig

    fig = go.Figure(go.Pie(
        labels=["Truth", "Lie"],
        values=[truth_count, lie_count],
        marker={"colors": [_GREEN, _RED]},
        textinfo="label+percent+value",
        textfont={"size": 14, "color": "white"},
        hovertemplate="%{label}: %{value} samples (%{percent})<extra></extra>",
        hole=0.4,
    ))

    fig.update_layout(
        title={
            "text": f"Dataset Distribution ({total} samples)",
            "font": {"color": _TEXT},
        },
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        legend={"font": {"color": _TEXT}},
        height=320,
        margin={"t": 60, "b": 30, "l": 20, "r": 20},
    )
    return fig


# ---------------------------------------------------------------------------
# Accuracy over training history
# ---------------------------------------------------------------------------

def create_accuracy_chart(accuracy_history: list[dict]) -> go.Figure:
    """Line chart of model accuracy over training runs.

    Args:
        accuracy_history: List of dicts with keys ``trained_on`` and
                          ``train_accuracy``.

    Returns:
        Plotly Figure.
    """
    if not accuracy_history:
        fig = go.Figure()
        fig.update_layout(
            title="No training history",
            template=_PLOTLY_TEMPLATE,
            paper_bgcolor=_DARK_BG,
        )
        return fig

    dates = [h.get("trained_on", "")[:10] for h in accuracy_history]
    accs = [h.get("train_accuracy", 0.0) * 100 for h in accuracy_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=accs,
        mode="lines+markers",
        line={"color": _BLUE, "width": 2},
        marker={"size": 8, "color": _BLUE},
        name="Train Accuracy",
        hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title={"text": "Model Accuracy History", "font": {"color": _TEXT}},
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        yaxis={"range": [0, 105]},
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_PANEL_BG,
        height=280,
        margin={"t": 50, "b": 50, "l": 60, "r": 20},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty_feature_name(name: str) -> str:
    """Convert a snake_case feature name to a readable label."""
    clean = name.removeprefix("ems_").removeprefix("opensmile_")
    clean = clean.replace("_", " ").title()
    return clean
