"""
pages/prediction.py
-------------------
Página 6: Predicción en tiempo real
"""

import matplotlib.pyplot as plt
import streamlit as st

from src.config import LANG_META
from src.models import predecir
from src.preprocessing import limpiar_texto
from src.ui import charts
from src.ui.styles import section_header, info_box


SAMPLE_TEXTS = {
    "English":    "The global economy is showing signs of recovery after the pandemic.",
    "Spanish":    "La tecnología está transformando todas las industrias del mundo.",
    "French":     "Les scientifiques ont fait une découverte remarquable cette année.",
    "German":     "Die Bundesregierung hat neue Maßnahmen zur Klimapolitik angekündigt.",
    "Italian":    "Il museo ospita una straordinaria collezione di arte rinascimentale.",
    "Portuguese": "A pesquisa científica brasileira tem alcançado resultados impressionantes.",
    "Mixed":      "Hello mundo! Bonjour friends, wie geht es euch oggi?",
}


def render() -> None:
    st.markdown(section_header("Step 6 — Realizar prediciones"), unsafe_allow_html=True)

    pipelines = st.session_state.get("trained_pipelines") or {}
    if not pipelines:
        st.warning("Selecciona al menos un modelo de entrenamiento")
        st.stop()

    model_name = st.selectbox("Modelo a usar:", list(pipelines.keys()))
    pipe       = pipelines[model_name]

    # Selector de texto de ejemplo
    preset = st.selectbox(
        "Load a sample text:",
        ["— type your own —"] + list(SAMPLE_TEXTS.keys()),
    )
    default_text = SAMPLE_TEXTS.get(preset, "")

    user_text = st.text_area(
        "Texto a clasificar:",
        value=default_text,
        height=130,
        placeholder="Introduce un texto en alguno de los idiomas soportados",
    )

    if not st.button("Detectar idioma"):
        return
    if not user_text.strip():
        st.warning("Introduce un texto")
        return

    # Predicción
    cleaned  = limpiar_texto(user_text)
    resultado = predecir(pipe, cleaned)
    pred_code = resultado["pred"]
    meta      = LANG_META[pred_code]

    # Tarjeta de resultado
    st.markdown(f"""
    <div class="pred-card" style="border:2px solid {meta['color']}">
      <div class="pred-flag">{meta['flag']}</div>
      <div class="pred-name" style="color:{meta['color']}">{meta['name']}</div>
      <div class="pred-model">{pred_code.upper()} · {model_name}</div>
    </div>
    """, unsafe_allow_html=True)

    # Gráfico de confianza
    if resultado["has_proba"] and resultado["proba"]:
        fig = charts.grafico_confianza(resultado["proba"], pred_code, LANG_META, ylabel="Probability")
    elif resultado["decision"]:
        fig = charts.grafico_confianza(resultado["decision"], pred_code, LANG_META, ylabel="Decision Score")
    else:
        fig = None

    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Estadísticas del texto
    st.markdown(info_box(
        f"<strong>Input stats:</strong> "
        f"{len(user_text)} carácteres · {len(user_text.split())} palabras · "
        f"{len(set(user_text.lower()))} carácteres únicos"
    ), unsafe_allow_html=True)
