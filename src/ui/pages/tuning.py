"""
pages/tuning.py
---------------
Página 5: Ajuste interactivo de hiperparámetros con GridSearchCV.
Incluye explicación del proceso de Cross-Validation y editor de parámetros.
"""

from __future__ import annotations
import ast
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import MODELS_CONFIG
from src.tuning import ajustar_hiperparametros
from src.ui import charts
from src.ui.styles import section_header, info_box


# ── Helpers de UI ─────────────────────────────────────────────────────────────

def _pill(text: str, color: str) -> str:
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:20px;padding:2px 10px;font-family:\'DM Mono\',monospace;'
            f'font-size:0.75rem;margin:2px;">{text}</span>')


# ── Diagramas SVG de Cross-Validation ────────────────────────────────────────

def _cv_diagram_svg(n_folds: int) -> str:
    """Genera un SVG que ilustra K-Fold CV con los folds coloreados."""
    W, H    = 560, n_folds * 36 + 60
    fold_w  = 460 // n_folds
    colors  = ["#c8ff00", "#4A90D9", "#E8603C", "#5BAD6F", "#C8A84B",
                "#9B59B6", "#E91E8C", "#00BCD4", "#FF5722", "#78909C"]

    rows = ""
    for f in range(n_folds):
        y = 44 + f * 36
        # Label
        rows += f'<text x="12" y="{y+18}" fill="#aaa" font-size="11" font-family="monospace">fold {f+1}</text>'
        for b in range(n_folds):
            x     = 70 + b * (fold_w + 3)
            is_val = b == f
            fill  = colors[f] if is_val else "#1f1f1f"
            stroke = colors[f] if is_val else "#333"
            label  = "VAL" if is_val else "TRAIN"
            label_color = "#000" if is_val else "#555"
            rows += (f'<rect x="{x}" y="{y}" width="{fold_w}" height="28" '
                     f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            rows += (f'<text x="{x+fold_w//2}" y="{y+18}" fill="{label_color}" '
                     f'font-size="9" font-family="monospace" text-anchor="middle">{label}</text>')

    return f"""<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H}" fill="#0d0d0d" rx="10"/>
  <text x="12" y="22" fill="#c8ff00" font-size="12" font-family="monospace" font-weight="bold">
    {n_folds}-Fold Cross-Validation
  </text>
  {rows}
  <text x="12" y="{H-6}" fill="#555" font-size="10" font-family="monospace">
    Score final = media de los {n_folds} scores de validación
  </text>
</svg>"""


# ── Generador de widgets por tipo de parámetro ────────────────────────────────

def _widget_param(param_key: str, meta: dict, model_name: str) -> list:
    """
    Renderiza el widget apropiado según meta['type'] y devuelve
    la lista de valores seleccionados para el grid de búsqueda.
    """
    uid  = f"{model_name}__{param_key}"
    help_txt = meta.get("help", "")
    ptype    = meta.get("type", "float")
    defaults = meta.get("values", [])

    # ── Tipo: float ───────────────────────────────────────────────────────────
    if ptype == "float":
        mn, mx = meta.get("min", 0.0001), meta.get("max", 100.0)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            v_min = st.number_input("Min", value=float(min(defaults)),
                                     min_value=float(mn), max_value=float(mx),
                                     format="%.4f", key=f"{uid}_min")
        with c2:
            v_max = st.number_input("Max", value=float(max(defaults)),
                                     min_value=float(mn), max_value=float(mx),
                                     format="%.4f", key=f"{uid}_max")
        with c3:
            n = st.number_input("Nº valores", value=len(defaults),
                                  min_value=1, max_value=10, step=1, key=f"{uid}_n")
        n = max(int(n), 1)
        if n == 1:
            vals = [round(float(v_min), 6)]
        else:
            vals = [round(float(v_min) + i * (float(v_max) - float(v_min)) / (n - 1), 6)
                    for i in range(n)]
        st.caption(f"→ Se explorarán: `{vals}`")
        if help_txt:
            st.caption(f"💡 {help_txt}")
        return vals

    # ── Tipo: int ─────────────────────────────────────────────────────────────
    elif ptype == "int":
        mn, mx = meta.get("min", 1), meta.get("max", 500)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            v_min = st.number_input("Min", value=int(min(defaults)),
                                     min_value=int(mn), max_value=int(mx),
                                     step=1, key=f"{uid}_min")
        with c2:
            v_max = st.number_input("Max", value=int(max(defaults)),
                                     min_value=int(mn), max_value=int(mx),
                                     step=1, key=f"{uid}_max")
        with c3:
            n = st.number_input("Nº valores", value=len(defaults),
                                  min_value=1, max_value=10, step=1, key=f"{uid}_n")
        n = max(int(n), 1)
        vals = sorted(set(
            int(v_min + i * (v_max - v_min) / max(n - 1, 1))
            for i in range(n)
        ))
        st.caption(f"→ Se explorarán: `{vals}`")
        if help_txt:
            st.caption(f"💡 {help_txt}")
        return vals

    # ── Tipo: int con None (ej. max_depth) ────────────────────────────────────
    elif ptype == "int_none":
        mn, mx = meta.get("min", 1), meta.get("max", 100)
        include_none = st.checkbox("Incluir `None` (sin límite)", value=True, key=f"{uid}_none")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            v_min = st.number_input("Min", value=int(mn), min_value=int(mn), max_value=int(mx), step=1, key=f"{uid}_min")
        with c2:
            v_max = st.number_input("Max", value=int(mx), min_value=int(mn), max_value=int(mx), step=1, key=f"{uid}_max")
        with c3:
            n = st.number_input("Nº valores enteros", value=len(defaults),
                                  min_value=1, max_value=8, step=1, key=f"{uid}_n")
        n = max(int(n), 1)
        int_vals = sorted(set(
            int(v_min + i * (v_max - v_min) / max(n - 1, 1)) for i in range(n)
        ))
        vals = ([None] if include_none else []) + int_vals
        st.caption(f"→ Se explorarán: `{vals}`")
        if help_txt:
            st.caption(f"💡 {help_txt}")
        return vals

    # ── Tipo: select (cadenas) ────────────────────────────────────────────────
    elif ptype == "select":
        options = meta.get("options", defaults)
        selected = st.multiselect(
            "Valores a probar", options=options, default=defaults, key=uid
        )
        if help_txt:
            st.caption(f"💡 {help_txt}")
        return selected if selected else defaults

    # ── Tipo: tuple_select (arquitecturas MLP) ────────────────────────────────
    elif ptype == "tuple_select":
        options  = meta.get("values", ["(64,)", "(128,)", "(128, 64)"])
        selected = st.multiselect(
            "Arquitecturas de capas ocultas", options=options, default=options[:2], key=uid
        )
        if help_txt:
            st.caption(f"💡 {help_txt}")
        parsed = []
        for s in (selected if selected else options[:1]):
            try:
                parsed.append(ast.literal_eval(s))
            except Exception:
                pass
        return parsed if parsed else [(64,)]

    return defaults


# ── Página principal ──────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(section_header("Step 5 — Ajuste de Hiperparámetros"), unsafe_allow_html=True)

    pipelines = st.session_state.get("trained_pipelines") or {}
    if not pipelines:
        st.warning("Entrena al menos un modelo primero en **🤖 Model Training**.")
        st.stop()

    # ═════════════════════════════════════════════════════════════════════════
    # SECCIÓN A: ¿Qué es Cross-Validation?
    # ═════════════════════════════════════════════════════════════════════════
    with st.expander("📖 ¿Qué es Cross-Validation? — Explicación detallada", expanded=False):
        st.markdown("""
        ### Cross-Validation (CV) — Validación cruzada

        **El problema:** si evaluamos el modelo siempre sobre el mismo conjunto de validación,
        podemos estar sobreajustando la elección de hiperparámetros a ese subconjunto específico.

        **La solución:** **K-Fold Cross-Validation** divide el conjunto de entrenamiento en
        **K partes iguales (folds)**. En cada iteración usa K-1 folds para entrenar y 1 fold
        para validar, rotando el fold de validación. El score final es la **media de los K scores**.

        #### Ventajas
        - Estimación más fiable de la capacidad de generalización
        - Usa todos los datos tanto para entrenar como para validar
        - Reduce la varianza en la estimación del error

        #### Desventaja
        - Coste computacional: K veces más lento que una sola evaluación
        """)

        cv_demo = st.slider("Número de folds (demo visual):", 2, 8, 5, key="cv_demo")
        st.markdown(
            _cv_diagram_svg(cv_demo),
            unsafe_allow_html=True
        )

        st.markdown("""
        #### GridSearchCV = Grid de hiperparámetros × Cross-Validation

        ```
        Para cada combinación de hiperparámetros:
            Para cada fold (1..K):
                Entrenar con K-1 folds → evaluar en el fold restante
            Score_combinación = media(K scores)
        Elegir la combinación con mayor Score_combinación
        ```

        > **Coste total:** `Nº combinaciones × K folds` ejecuciones del modelo.
        > El resumen de combinaciones se muestra antes de lanzar el grid search.
        """)

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # SECCIÓN B: Configuración
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(section_header("Configuración del Grid Search"), unsafe_allow_html=True)

    # Solo modelos base (sin ✦ Tuned) son válidos para tunear
    base_models = [k for k in pipelines.keys() if "✦ Tuned" not in k]

    col_model, col_folds = st.columns([3, 1])
    with col_model:
        model_name = st.selectbox(
            "Modelo a ajustar:",
            base_models,
            help="Solo se muestran modelos base. Los modelos tuneados no se pueden re-tunear aquí."
        )
    with col_folds:
        cv_folds = st.slider(
            "CV folds (K)", 2, 10, 5,
            help="K=3 rápido · K=5 equilibrado · K=10 más preciso pero lento"
        )

    # Buscar config por nombre base (por si acaso llega con sufijo ✦ Tuned)
    base_model_name = model_name.replace(" ✦ Tuned", "").strip()
    cfg  = MODELS_CONFIG[base_model_name]
    meta = cfg.get("param_meta", {})

    # ═════════════════════════════════════════════════════════════════════════
    # SECCIÓN C: Editor interactivo de hiperparámetros
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(section_header("⚙️ Configurar parámetros a explorar"), unsafe_allow_html=True)

    st.markdown(info_box(
        f"<strong>Modelo seleccionado:</strong> {model_name}<br>"
        f"Ajusta los rangos de cada hiperparámetro. El grid search probará "
        f"<strong>todas las combinaciones posibles</strong> de los valores configurados."
    ), unsafe_allow_html=True)

    custom_grid: dict = {}
    param_keys = list(cfg["params"].keys())

    for param_key in param_keys:
        # Extraer nombre limpio (sin prefijo clf__)
        short_name = param_key.replace("clf__", "")
        param_meta = meta.get(param_key, {
            "type": "float" if isinstance(cfg["params"][param_key][0], float) else "int",
            "values": cfg["params"][param_key],
            "min": min(v for v in cfg["params"][param_key] if v is not None),
            "max": max(v for v in cfg["params"][param_key] if v is not None),
        })

        with st.container():
            st.markdown(f"**`{short_name}`**")
            selected_vals = _widget_param(param_key, param_meta, model_name)
            custom_grid[param_key] = selected_vals
            st.divider()

    # ── Resumen del grid ──────────────────────────────────────────────────────
    n_combos = 1
    for v in custom_grid.values():
        n_combos *= max(len(v) if v else 1, 1)

    total_fits = n_combos * cv_folds
    time_est   = "muy rápido (<1 min)" if total_fits < 20 else \
                 "moderado (1-5 min)"  if total_fits < 60 else \
                 "lento (>5 min, sé paciente)"

    st.markdown(info_box(
        f"<strong>Combinaciones:</strong> {n_combos} × {cv_folds} folds "
        f"= <strong>{total_fits} ajustes totales</strong><br>"
        f"<strong>Tiempo estimado:</strong> {time_est}"
    ), unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECCIÓN D: Ejecutar y mostrar resultados
    # ═════════════════════════════════════════════════════════════════════════
    if st.button("🔍  Run Grid Search"):
        if not all(v for v in custom_grid.values()):
            st.error("Selecciona al menos un valor por parámetro.")
            return

        with st.spinner(
            f"GridSearchCV: {model_name} · {n_combos} combinaciones × {cv_folds} folds…"
        ):
            resultado = ajustar_hiperparametros(
                model_name    = base_model_name,
                X_train       = st.session_state["X_train"],
                X_test        = st.session_state["X_test"],
                y_train       = st.session_state["y_train"],
                y_test        = st.session_state["y_test"],
                pipeline_base = pipelines[model_name],
                cv_folds      = cv_folds,
                custom_params = custom_grid,
            )

        # Guardar el modelo afinado con clave propia — NO sobreescribir el baseline
        # Así ambos (original y tuned) aparecen en la comparativa
        tuned_key = f"{base_model_name} ✦ Tuned"
        pipelines[tuned_key] = resultado["best_pipeline"]
        st.session_state["trained_pipelines"] = pipelines
        st.session_state["tuning_results"]    = resultado

        # Acumular CV scores por modelo para mostrarlos en la comparativa
        cv_scores = st.session_state.get("tuning_cv_scores") or {}
        cv_scores[tuned_key] = resultado["best_cv_score"]
        st.session_state["tuning_cv_scores"] = cv_scores

        st.success(
            f"✅ Grid Search completado · Mejor CV Accuracy: **{resultado['best_cv_score']:.4f}**"
        )

    # ── Resultados ────────────────────────────────────────────────────────────
    r = st.session_state.get("tuning_results")
    if r is None:
        st.info("Configura los parámetros y haz clic en **Run Grid Search**.")
        return

    if r["model"] != model_name:
        st.info(f"Resultados del último tuning: **{r['model']}**. Lanza un nuevo grid search para **{model_name}**.")

    st.markdown(section_header("📊 Resultados del tuning"), unsafe_allow_html=True)

    # Mejores parámetros encontrados
    st.markdown("**Mejores hiperparámetros encontrados:**")
    best_clean = {k.replace("clf__", ""): v for k, v in r["best_params"].items()}
    cols_params = st.columns(len(best_clean))
    for col, (k, v) in zip(cols_params, best_clean.items()):
        with col:
            st.markdown(
                f'<div style="background:#141414;border:1px solid #c8ff0055;border-radius:8px;'
                f'padding:12px 14px;text-align:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#555;'
                f'letter-spacing:2px;text-transform:uppercase">{k}</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:1.3rem;font-weight:800;'
                f'color:#c8ff00">{v}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparativa de métricas
    col1, col2 = st.columns(2)
    with col1:
        fig = charts.grafico_mejora_tuning(r["acc_base"], r["acc_tuned"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        delta_acc = (r["acc_tuned"] - r["acc_base"]) * 100
        delta_f1  = (r["f1_tuned"]  - r["f1_base"])  * 100
        s_acc = f"+{delta_acc:.2f}%" if delta_acc >= 0 else f"{delta_acc:.2f}%"
        s_f1  = f"+{delta_f1:.2f}%"  if delta_f1  >= 0 else f"{delta_f1:.2f}%"
        color = lambda d: "#c8ff00" if d >= 0 else "#E8603C"

        st.markdown(info_box(f"""
          <strong>Baseline Accuracy:</strong>  {r['acc_base']:.4f}<br>
          <strong>Tuned Accuracy:</strong>     {r['acc_tuned']:.4f}<br>
          <strong>Accuracy Δ:</strong>
            <span style="color:{color(delta_acc)}">{s_acc}</span><br><br>
          <strong>Baseline Macro F1:</strong>  {r['f1_base']:.4f}<br>
          <strong>Tuned Macro F1:</strong>     {r['f1_tuned']:.4f}<br>
          <strong>F1 Δ:</strong>
            <span style="color:{color(delta_f1)}">{s_f1}</span><br><br>
          <strong>Best CV Accuracy:</strong>   {r['best_cv_score']:.4f}
        """), unsafe_allow_html=True)

    # Tabla de todos los resultados del grid
    st.markdown("**Todos los resultados del Grid Search:**")
    st.dataframe(
        r["cv_results_df"].rename(columns={
            "mean_test_score": "CV Accuracy (media)",
            "std_test_score":  "Std Dev",
            "rank_test_score": "Rank",
        }),
        use_container_width=True, hide_index=True
    )
