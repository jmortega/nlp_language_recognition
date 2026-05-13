"""
pages/dataset.py
----------------
Página 1: Formulación del problema y exploración del dataset (EDA).
"""

import streamlit as st
import matplotlib.pyplot as plt

from src.config import LANG_META, DATASET_PATH
from src.preprocessing import cargar_dataset
from src.ui.styles import metric_card, info_box, section_header
from src.ui import charts


def render() -> None:
    st.markdown(section_header("Step 1 — Formulación del problema y exploración del dataset (EDA)"), unsafe_allow_html=True)

    # Definición del problema
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(info_box("""
          <strong>Tarea:</strong> Clasificación de texto en varias clases<br>
          <strong>Tipo:</strong> Aprendizahe supervisado<br>
          <strong>Entrada:</strong> Texto <br>
          <strong>Salida:</strong> Idioma (en, es, fr, de, it, pt)<br>
          <strong>Metrics:</strong> Accuracy + Macro F1-score<br>
        """), unsafe_allow_html=True)

        st.markdown("""
        ### n-gramas de caracteres
        Los n-gramas a nivel de caracteres (de 2 a 4 caracteres) capturan:
        - Patrones morfológicos únicos de cada idioma.
        - Distribuciones de frecuencia de caracteres
        - Límites de palabras cortas que diferencian las lenguas romances de las germánicas.
        - Combinaciones de caracteres acentuados (fr / it / es / pt)""")

    with col2:
        for code, meta in LANG_META.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        background:#141414;border:1px solid #1f1f1f;
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;">
              <span style="font-size:1.4rem">{meta['flag']}</span>
              <div>
                <span style="font-family:'Syne',sans-serif;font-weight:700;color:#fff">{meta['name']}</span>
                <br>
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;
                             color:{meta['color']};letter-spacing:2px">{code.upper()}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Carga del dataset
    st.markdown(section_header("Dataset Overview"), unsafe_allow_html=True)

    try:
        df = cargar_dataset(DATASET_PATH)
        st.session_state["df"] = df
    except FileNotFoundError:
        st.error(f"⚠️  `{DATASET_PATH}` not found. Place the CSV in the project root.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{len(df):,}",                     "Nº MUESTRAS"),
        (k2, str(df.language.nunique()),          "IDIOMAS"),
        (k3, str(int(df.text_length.mean())),     "AVG CARACTERES/MUESTRA"),
        (k4, str(int(df.word_count.mean())),      "AVG PALABRAS/MUESTRA"),
    ]:
        with col:
            st.markdown(metric_card(val, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos EDA
    col_a, col_b = st.columns(2)
    with col_a:
        fig = charts.grafico_distribucion_clases(df.language.value_counts(), LANG_META)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        fig = charts.grafico_longitud_texto(df, LANG_META)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Muestra de registros
    st.markdown("### Muestra de registros")
    sample = (
        df[["text", "language", "lang_name", "word_count"]]
        .sample(10, random_state=1)
        .rename(columns={"text": "Text", "language": "Code",
                          "lang_name": "Language", "word_count": "Words"})
    )
    st.dataframe(sample, use_container_width=True, hide_index=True)
