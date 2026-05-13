"""
pages/evaluation.py
-------------------
Página 4: Evaluación del modelo seleccionado
"""

import matplotlib.pyplot as plt
import streamlit as st

from src.config import LANG_META
from src.models import evaluar_modelo
from src.ui import charts
from src.ui.styles import section_header, metric_card


def render() -> None:
    st.markdown(section_header("Step 4 — Evaluación del modelo"), unsafe_allow_html=True)

    pipelines = st.session_state.get("trained_pipelines") or {}
    if not pipelines:
        st.warning("Selecciona al menos un modelo de entrenamiento")
        st.stop()

    model_name = st.selectbox("Seleccionar modelo a evaluar:", list(pipelines.keys()))
    pipe   = pipelines[model_name]
    X_te   = st.session_state["X_test"]
    y_te   = st.session_state["y_test"]

    labels = sorted(set(y_te))
    names  = [LANG_META[l]["name"] for l in labels]

    ev = evaluar_modelo(pipe, X_te, y_te, labels, names)

    # ── Métricas globales (5 KPIs) ────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, label in [
        (k1, f"{ev['acc']:.4f}",       "TEST ACCURACY"),
        (k2, f"{ev['f1']:.4f}",        "MACRO F1"),
        (k3, f"{ev['precision']:.4f}", "MACRO PRECISION"),
        (k4, f"{ev['recall']:.4f}",    "MACRO RECALL"),
        (k5, str(len(X_te)),           "TEST SAMPLES"),
    ]:
        with col:
            st.markdown(metric_card(val, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Matriz de confusión + métricas por clase ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Matriz de confusión")
        fig = charts.grafico_confusion_matrix(ev["cm"], names)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown("#### Métricas por clase")
        fig = charts.grafico_metricas_por_clase(ev["report_df"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Reporte de clasificación")
    st.text(ev["report_str"])

    # ── Análisis de errores ───────────────────────────────────────────────────
    if len(ev["errors_idx"]) > 0:
        st.markdown("#### Muestras clasificadas de forma errónea")
        y_pred = ev["y_pred"]
        rows   = [
            {
                "Texto":          X_te[i][:90] + "…",
                "Etiqueta real":  LANG_META[y_te[i]]["name"],
                "Predicción":     LANG_META[y_pred[i]]["name"],
            }
            for i in ev["errors_idx"][:15]
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

