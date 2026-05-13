"""
ui/charts.py
------------
Funciones de visualización con matplotlib/seaborn
Devuelven objetos Figure para que Streamlit los renderice con st.pyplot()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns


# Paleta y estilo base
BG   = "#0d0d0d"
BG2  = "#141414"
GRID = "#1f1f1f"
TEXT = "#aaaaaa"
ACC  = "#c8ff00"


def _base_fig(w: float = 8, h: float = 4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    return fig, ax


# Distribución de clases
def grafico_distribucion_clases(
    counts: pd.Series,
    lang_meta: dict,
) -> plt.Figure:
    fig, ax = _base_fig(6, 3.5)
    colors = [lang_meta[c]["color"] for c in counts.index]
    labels = [f"{lang_meta[c]['flag']} {lang_meta[c]['name']}" for c in counts.index]
    bars   = ax.barh(labels, counts.values, color=colors, edgecolor=BG)
    for bar, v in zip(bars, counts.values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2, str(v),
                va="center", color=TEXT, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Count", color=TEXT, fontsize=8)
    ax.set_title("Class Distribution", color=ACC, fontsize=10)
    ax.xaxis.grid(True, color=GRID)
    plt.tight_layout()
    return fig


# Histograma de longitud de texto
def grafico_longitud_texto(df: pd.DataFrame, lang_meta: dict) -> plt.Figure:
    fig, ax = _base_fig(6, 3.5)
    for code, meta in lang_meta.items():
        subset = df[df["language"] == code]["text_length"]
        ax.hist(subset, bins=20, alpha=0.5, color=meta["color"],
                label=meta["name"], edgecolor="none")
    ax.set_xlabel("Text length (chars)", color=TEXT, fontsize=8)
    ax.set_ylabel("Frequency",           color=TEXT, fontsize=8)
    ax.set_title("Text Length Distribution", color=ACC, fontsize=10)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#333", labelcolor=TEXT)
    ax.yaxis.grid(True, color=GRID)
    plt.tight_layout()
    return fig


# N-gramas más frecuentes
def grafico_ngrams(tokens: list[str], counts: list[int], title: str) -> plt.Figure:
    fig, ax = _base_fig(10, 3)
    cmap   = plt.cm.YlGn
    colors = [cmap(v / max(counts)) for v in counts]
    ax.bar(range(len(tokens)), counts, color=colors, edgecolor=BG)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([repr(t) for t in tokens], rotation=45, ha="right",
                        fontsize=7, color=TEXT, fontfamily="monospace")
    ax.set_title(title, color=ACC, fontsize=10)
    ax.yaxis.grid(True, color=GRID)
    plt.tight_layout()
    return fig


# Barras de comparación de modelos
def grafico_comparacion_modelos(
    names:  list[str],
    values: list[float],
    colors: list[str],
    title:  str,
    ylabel: str = "Accuracy",
) -> plt.Figure:
    fig, ax = _base_fig(8, 4)
    bars = ax.bar(names, values, color=colors, edgecolor=BG, linewidth=0.5, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom",
                color="#ccc", fontsize=8, fontfamily="monospace")
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.set_title(title,   color=ACC,  fontsize=11)
    ax.set_ylim(max(0, min(values) - 0.05), 1.02)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


# Matriz de confusión
def grafico_confusion_matrix(cm: np.ndarray, labels: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#1a1a1a",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_xlabel("Predicted", color="#888", fontsize=9)
    ax.set_ylabel("True",      color="#888", fontsize=9)
    ax.set_title("Confusion Matrix", color=ACC, fontsize=11, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    plt.tight_layout()
    return fig


# Métricas por clase (precision/recall/F1)
def grafico_metricas_por_clase(report_df: pd.DataFrame) -> plt.Figure:
    fig, ax = _base_fig(6, 4)
    x       = np.arange(len(report_df))
    w       = 0.28
    metrics = [("precision", "#4A90D9"), ("recall", "#5BAD6F"), ("f1-score", ACC)]
    for i, (metric, color) in enumerate(metrics):
        ax.bar(x + i * w, report_df[metric], w, label=metric.capitalize(),
               color=color, edgecolor=BG)
    ax.set_xticks(x + w)
    ax.set_xticklabels(report_df.index, rotation=20, ha="right", color=TEXT, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=TEXT)
    ax.set_title("Precision / Recall / F1 per Language", color=ACC, fontsize=10)
    ax.yaxis.grid(True, color=GRID)
    plt.tight_layout()
    return fig


# Otener grado de confianza de la predicción
def grafico_confianza(
    scores:    dict[str, float],
    pred_code: str,
    lang_meta: dict,
    ylabel:    str = "Probability",
) -> plt.Figure:
    items     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    codes     = [k for k, _ in items]
    vals      = [v for _, v in items]
    labels    = [f"{lang_meta[c]['flag']} {lang_meta[c]['name']}" for c in codes]
    colors    = [lang_meta[c]["color"] if c == pred_code else "#2a2a2a" for c in codes]

    fig, ax = _base_fig(8, 3.5)
    bars = ax.bar(labels, vals, color=colors, edgecolor=BG)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                color=TEXT, fontsize=8, fontfamily="monospace")
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.set_title("Prediction Confidence", color=ACC, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2 if vals else 1.15)
    ax.yaxis.grid(True, color=GRID)
    plt.tight_layout()
    return fig


# Mejora por tuning
def grafico_mejora_tuning(acc_base: float, acc_tuned: float) -> plt.Figure:
    return grafico_comparacion_modelos(
        names  = ["Baseline", "Tuned"],
        values = [acc_base, acc_tuned],
        colors = ["#4A90D9", ACC],
        title  = "Accuracy: Baseline vs Tuned",
    )
