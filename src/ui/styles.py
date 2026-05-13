"""
ui/styles.py
------------
Inyecta el CSS global en la aplicación Streamlit
"""

import streamlit as st


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #1f1f1f;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] .stRadio label { color: #b0b0b0 !important; font-size: 0.85rem; }
  section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 8px 12px; border-radius: 6px; transition: background 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio > div > label:hover { background: #1a1a1a; }

  /* Hero */
  .hero-title { font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; letter-spacing:-1px; color:#fff; line-height:1.1; margin-bottom:0; }
  .hero-sub   { font-family:'DM Mono',monospace; font-size:0.75rem; color:#555; letter-spacing:3px; text-transform:uppercase; margin-top:4px; }
  .accent     { color:#c8ff00; }

  /* Section header */
  .section-header {
    font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#fff;
    border-bottom:2px solid #c8ff00; padding-bottom:6px; margin-bottom:1.2rem;
  }

  /* Metric card */
  .metric-card { background:#141414; border:1px solid #222; border-radius:10px; padding:18px 22px; text-align:center; transition:border-color 0.2s; }
  .metric-card:hover { border-color:#c8ff00; }
  .metric-val   { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#c8ff00; }
  .metric-label { font-family:'DM Mono',monospace; font-size:0.68rem; color:#666; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

  /* Info box */
  .info-box {
    background:#111; border-left:3px solid #c8ff00; border-radius:0 8px 8px 0;
    padding:14px 18px; font-family:'DM Mono',monospace; font-size:0.82rem; color:#aaa; margin:1rem 0;
  }
  .info-box strong { color:#c8ff00; }

  /* Lang pill */
  .lang-pill {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-family:'DM Mono',monospace; font-size:0.75rem; font-weight:500; letter-spacing:1px; margin:3px;
  }

  /* Prediction card */
  .pred-card {
    border-radius:14px; padding:24px 30px; text-align:center; margin-bottom:1.5rem;
    background:#141414;
  }
  .pred-flag  { font-size:3.5rem; }
  .pred-name  { font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; margin:6px 0; }
  .pred-model { font-family:'DM Mono',monospace; font-size:0.75rem; color:#666; letter-spacing:3px; }

  /* Button */
  .stButton > button {
    background:#c8ff00 !important; color:#000 !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    border:none !important; border-radius:6px !important;
    padding:0.5rem 1.5rem !important; transition:opacity 0.2s !important;
  }
  .stButton > button:hover { opacity:0.85 !important; }

  /* Form controls */
  .stSelectbox label, .stSlider label, .stTextArea label, .stMultiSelect label { color:#888 !important; }
  .stDataFrame { border:1px solid #1f1f1f; border-radius:8px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#0d0d0d; }
  ::-webkit-scrollbar-thumb { background:#2a2a2a; border-radius:3px; }
</style>
"""


def inject_styles() -> None:
    """Inyecta el CSS global en la página de Streamlit."""
    st.markdown(CSS, unsafe_allow_html=True)


def metric_card(value: str, label: str) -> str:
    """Devuelve el HTML de una tarjeta de métrica."""
    return f"""
    <div class="metric-card">
      <div class="metric-val">{value}</div>
      <div class="metric-label">{label}</div>
    </div>
    """


def info_box(html_content: str) -> str:
    """Devuelve el HTML de un bloque informativo con borde verde."""
    return f'<div class="info-box">{html_content}</div>'


def section_header(title: str) -> str:
    """Devuelve el HTML de una cabecera de sección."""
    return f'<div class="section-header">{title}</div>'
