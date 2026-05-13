"""
pages/training.py
-----------------
Página 3: Selección y entrenamiento de modelos.
Actualizado para incluir MLP y XGBoost junto con nota de tiempos de entrenamiento.
"""

import matplotlib.pyplot as st_plt
import streamlit as st

from src.config import MODELS_CONFIG
from src.models import entrenar_modelos
from src.ui import charts
from src.ui.styles import section_header, info_box

# Modelos rápidos (recomendados para primera ejecución)
FAST_MODELS   = ["Logistic Regression", "Linear SVC", "Naive Bayes"]
# Modelos lentos (más potentes pero requieren más tiempo)
SLOW_MODELS   = ["Random Forest", "Gradient Boosting", "MLP Classifier", "XGBoost"]


def render() -> None:
    st.markdown(section_header("Step 3 — Selección y entrenamiento de modelos"), unsafe_allow_html=True)

    if st.session_state.get("X_train") is None:
        st.warning("Completa **Feature Engineering** primero para definir el split train/test.")
        st.stop()

    st.markdown("""
    **7 clasificadores** encapsulados en un `Pipeline` con TF-IDF de n-gramas de caracteres.
    Selecciona los modelos a entrenar y pulsa el botón.
    """)

    # Aviso de tiempos para modelos pesados
    st.markdown(info_box(
        "<strong>⏱️ Tiempos orientativos</strong> (dataset 594 frases, 80/20):<br>"
        "🟢 <strong>Rápidos (&lt;5s):</strong> Logistic Regression, Linear SVC, Naive Bayes<br>"
        "🟡 <strong>Moderados (5-30s):</strong> Random Forest, MLP Classifier<br>"
        "🔴 <strong>Lentos (&gt;30s):</strong> Gradient Boosting, XGBoost"
    ), unsafe_allow_html=True)

    # Selector de modelos con default = solo los rápidos
    all_models = list(MODELS_CONFIG.keys())
    models_to_train = st.multiselect(
        "Selecciona los modelos a entrenar:",
        all_models,
        default=FAST_MODELS,
        help="Los modelos de boosting y redes neuronales son más lentos pero pueden ofrecer mejor accuracy."
    )

    if st.button("🚀  Entrenar modelos seleccionados") and models_to_train:
        X_tr = st.session_state["X_train"]
        X_te = st.session_state["X_test"]
        y_tr = st.session_state["y_train"]
        y_te = st.session_state["y_test"]

        progress_bar = st.progress(0, text="Entrenando…")
        status_text  = st.empty()

        def _on_progress(i: int, total: int, name: str) -> None:
            progress_bar.progress(i / total, text=f"Entrenando **{name}**…")
            status_text.markdown(f"⏳ Entrenando **{name}**…")

        try:
            pipelines, results_df = entrenar_modelos(
                models_to_train, X_tr, X_te, y_tr, y_te,
                progress_callback=_on_progress,
            )
            progress_bar.progress(1.0, text="¡Completado!")
            status_text.empty()

            existing = st.session_state.get("trained_pipelines") or {}
            existing.update(pipelines)
            st.session_state["trained_pipelines"] = existing
            st.session_state["results_df"]        = results_df

            best_name = results_df.iloc[0]["Modelo"]
            st.session_state["best_model_name"] = best_name
            st.session_state["best_pipeline"]   = pipelines[best_name]

            st.success(f"✅ Entrenamiento completado · Mejor modelo: **{best_name}**")

        except ImportError as e:
            st.error(f"❌ Dependencia no instalada: {e}\nEjecuta `pip install xgboost` para usar XGBoost.")
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {e}")

    # ── Tabla de resultados ───────────────────────────────────────────────────
    if st.session_state.get("results_df") is not None:
        df_r = st.session_state["results_df"]
        st.markdown("### Resultados")
        st.dataframe(
            df_r.style.format({
                "Test Accuracy":        "{:.4f}",
                "Macro F1":             "{:.4f}",
                "CV Accuracy (5-fold)": "{:.4f}",
                "Train Time (s)":       "{:.2f}",
            }).highlight_max(
                subset=["Test Accuracy", "Macro F1", "CV Accuracy (5-fold)"],
                color="#c8ff0022",
            ),
            use_container_width=True, hide_index=True,
        )

        fig = charts.grafico_comparacion_modelos(
            names  = df_r["Modelo"].tolist(),
            values = df_r["Test Accuracy"].tolist(),
            colors = [MODELS_CONFIG[m]["color"] for m in df_r["Modelo"]],
            title  = "Test Accuracy por Modelo",
        )
        st.pyplot(fig, use_container_width=True)
        st_plt.close(fig)
