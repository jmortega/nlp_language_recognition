"""
app.py
------
Punto de entrada de la aplicación Streamlit.
"""

import streamlit as st

st.set_page_config(
    page_title="Language Recognition v2.0",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.styles import inject_styles
from src.ui.pages import dataset, features, training, evaluation, tuning, prediction, comparison

inject_styles()

# ── Estado de sesión ──────────────────────────────────────────────────────────
_STATE_DEFAULTS: dict = {
    "df":                None,
    "X_train":           None,
    "X_test":            None,
    "y_train":           None,
    "y_test":            None,
    "trained_pipelines": {},
    "results_df":        None,
    "best_model_name":   None,
    "best_pipeline":     None,
    "tuning_results":    None,
    "tuning_cv_scores":  {},    # { "ModelName ✦ Tuned": cv_score }
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
  <p class="hero-sub">NLP · SUPERVISED LEARNING · SKLEARN</p>
  <h1 class="hero-title">Language <span class="accent">Recognition</span></h1>
  <p style="color:#555;font-size:0.85rem;margin-top:8px;font-family:'Inter',sans-serif;">
    Reconocimiento de idioma con aprendizaje supervisado — 7 modelos · 6 idiomas
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
        'color:#555;letter-spacing:3px;">ML PIPELINE</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-weight:800;'
        'font-size:1.2rem;color:#fff;margin-top:0">Language Recognition</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Formulación del problema y dataset",
            "🔬 Ingeniería de características",
            "🤖 Entrenamiento del modelo",
            "📈 Evaluación del modelo",
            "⚙️ Tuning de hiperparámetros",
            "🔁 Comparativa de modelos",
            "🌐 Realizar predicciones",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Indicador de modelos entrenados
    trained = st.session_state.get("trained_pipelines") or {}
    if trained:
        st.markdown(
            f'<div style="background:#141414;border:1px solid #c8ff0044;border-radius:8px;'
            f'padding:10px 14px;font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#aaa;">'
            f'<span style="color:#c8ff00">✓</span> <strong>{len(trained)}</strong> modelo(s) entrenado(s)<br>'
            + "".join(f'<span style="color:#555">· </span>{m}<br>' for m in trained.keys())
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("""
    <div style="background:#111;border-left:3px solid #c8ff00;
                border-radius:0 8px 8px 0;padding:14px 18px;
                font-family:'DM Mono',monospace;font-size:0.82rem;color:#aaa;">
      <span style="color:#c8ff00"><strong>Características</strong></span><br>
      TF-IDF char n-grams<br><br>
      <span style="color:#c8ff00"><strong>Modelos</strong></span><br>
      LR · SVC · RF · GB<br>NB · MLP · XGB<br><br>
      <span style="color:#c8ff00"><strong>Idiomas</strong></span><br>
      EN · ES · FR · DE · IT · PT
    </div>
    """, unsafe_allow_html=True)

# ── Enrutador ─────────────────────────────────────────────────────────────────
PAGES = {
    "📊 Formulación del problema y dataset":  dataset,
    "🔬 Ingeniería de características":        features,
    "🤖 Entrenamiento del modelo":             training,
    "📈 Evaluación del modelo":                evaluation,
    "⚙️ Tuning de hiperparámetros":            tuning,
    "🔁 Comparativa de modelos":               comparison,
    "🌐 Realizar predicciones":                prediction,
}

PAGES[page].render()
