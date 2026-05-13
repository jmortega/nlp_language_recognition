"""
pages/features.py
-----------------
Página 2: Ingeniería de características (TF-IDF char n-grams)
"""

from __future__ import annotations
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LANG_META, TFIDF_DEFAULTS
from src.models import dividir_datos
from src.preprocessing import cargar_dataset, limpiar_texto, limpiar_serie, extraer_ngrams_char
from src.ui import charts
from src.ui.styles import section_header, metric_card


def render() -> None:
    st.markdown(section_header("Step 2 — Ingeniería de características (TF-IDF char n-grams)"), unsafe_allow_html=True)

    if st.session_state.get("df") is None:
        st.warning("Primero realiza la carga de los datos")
        st.stop()

    df = st.session_state["df"]

    st.markdown("""
    ### Character N-gram TF-IDF
    `TfidfVectorizer(analyzer='char_wb')` — separa cada palabra con espacios
    extrae todas las subcadenas de longitud N y aplica la ponderación TF-IDF
    `strip_accents=None` mantiene los acentos
    """)

    # Inspector de n-gramas
    col1, col2 = st.columns(2)
    with col1:
        ngram_min = st.slider("Min n-gram size", 1, 4, 2, key="feat_min")
    with col2:
        ngram_max = st.slider("Max n-gram size", 2, 6, 4, key="feat_max")

    example_text = st.text_area(
        "Inspector de n-gramas para un texto:",
        value="The quick brown fox jumps over the lazy dog.",
        height=80,
    )

    if example_text.strip():
        ngrams = extraer_ngrams_char(example_text, ngram_min, ngram_max)
        freq   = Counter(ngrams).most_common(30)
        if freq:
            tokens, counts = zip(*freq)
            fig = charts.grafico_ngrams(
                list(tokens), list(counts),
                f"Top-30 character {ngram_min}–{ngram_max} grams",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # N-gramas distintos por idioma
    st.markdown("---")
    st.markdown("### N-gramas distintos por idioma")
    st.markdown("Patrones exclusivos de cada idioma:")

    @st.cache_data
    def _calcular_ngrams_distintivos(textos, etiquetas):
        vec = TfidfVectorizer(**{**TFIDF_DEFAULTS, "max_features": 30_000})
        X   = vec.fit_transform(textos)
        return X, vec.get_feature_names_out()

    textos   = limpiar_serie(df["text"])
    X, feats = _calcular_ngrams_distintivos(tuple(textos), tuple(df["language"]))

    cols = st.columns(3)
    for idx, (code, meta) in enumerate(LANG_META.items()):
        mask    = (df["language"] == code).values
        diff    = X[mask].mean(axis=0).A1 - X[~mask].mean(axis=0).A1
        top_ng  = feats[np.argsort(diff)[-12:][::-1]]

        with cols[idx % 3]:
            pills = "".join(
                f'<span class="lang-pill" style="background:{meta["color"]}22;'
                f'color:{meta["color"]};border:1px solid {meta["color"]}44">{ng}</span>'
                for ng in top_ng
            )
            st.markdown(f"""
            <div style="background:#141414;border:1px solid #1f1f1f;
                        border-radius:10px;padding:14px;margin-bottom:12px;">
              <div style="font-family:'Syne',sans-serif;font-weight:700;
                          font-size:0.9rem;color:{meta['color']};margin-bottom:8px;">
                {meta['flag']} {meta['name']}
              </div>
              {pills}
            </div>
            """, unsafe_allow_html=True)

    # Train / Test split
    st.markdown("---")
    st.markdown("### % Entrenamiento / Pruebas")
    test_size = st.slider("% conjunto de pruebas", 0.10, 0.40, 0.20, 0.05)

    from src.preprocessing import obtener_xy
    X_all, y_all = obtener_xy(df)
    X_tr, X_te, y_tr, y_te = dividir_datos(X_all, y_all, test_size=test_size)

    # Guardar en sesión
    st.session_state.update({
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
    })

    c1, c2, c3 = st.columns(3)
    for col, val, label in [
        (c1, len(X_tr), "TRAINING SAMPLES"),
        (c2, len(X_te), "TEST SAMPLES"),
        (c3, f"{100*(1-test_size):.0f} / {100*test_size:.0f}", "TRAIN / TEST"),
    ]:
        with col:
            st.markdown(metric_card(str(val), label), unsafe_allow_html=True)
