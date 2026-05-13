"""
pages/comparison.py
--------------------
Página 6: Comparativa de todos los modelos entrenados.
Muestra tabla de métricas, gráfico de barras, radar chart y
análisis del mejor y peor modelo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_CONFIG, LANG_META
from src.models import evaluar_modelo
from src.ui import charts
from src.ui.styles import section_header, info_box, metric_card


# ── Constantes de presentación ────────────────────────────────────────────────
BG   = "#0d0d0d"
BG2  = "#141414"
GRID = "#1f1f1f"
TEXT = "#aaaaaa"
ACC  = "#c8ff00"


# ── Helper: radar chart ───────────────────────────────────────────────────────
def _radar_chart(df: pd.DataFrame) -> plt.Figure:
    """Spider/radar chart con Accuracy, F1, Precision, Recall por modelo."""
    categories  = ["Accuracy", "Macro F1", "Precision", "Recall"]
    n_cats      = len(categories)
    angles      = [n / n_cats * 2 * np.pi for n in range(n_cats)]
    angles     += angles[:1]   # cerrar el polígono

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    # Cuadrícula
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT, fontsize=9, fontfamily="monospace")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], color="#444", fontsize=7)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5)

    # Una línea por modelo
    palette = [
        ACC, "#4A90D9", "#E8603C", "#5BAD6F",
        "#C8A84B", "#9B59B6", "#E91E8C", "#00BCD4",
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        vals   = [row.get(c, 0) for c in categories]
        vals  += vals[:1]
        color  = palette[i % len(palette)]
        ax.plot(angles, vals, color=color, linewidth=2, label=row["Modelo"])
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=TEXT,
    )
    ax.set_title("Radar — métricas por modelo", color=ACC, fontsize=10, pad=20)
    plt.tight_layout()
    return fig


# ── Helper: tabla de errores entre modelos ────────────────────────────────────
def _tabla_errores(pipelines: dict, X_test, y_test) -> pd.DataFrame:
    """Cuenta cuántos samples falla cada modelo para detectar solapamientos."""
    rows = []
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        n_err  = int((y_pred != y_test).sum())
        rows.append({"Modelo": name, "Errores": n_err, "% Error": round(n_err / len(y_test) * 100, 2)})
    return pd.DataFrame(rows).sort_values("Errores")


# ── Página ─────────────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown(section_header("Step 6 — Comparativa de modelos"), unsafe_allow_html=True)

    pipelines  = st.session_state.get("trained_pipelines") or {}
    tuning_r   = st.session_state.get("tuning_results")

    # Contar modelos únicos (base + tuned cuentan por separado)
    n_models = len(pipelines)

    # Necesitamos al menos 2 entradas (ej. 1 base + 1 tuned, o 2 bases)
    if n_models < 2:
        st.info(
            "Necesitas al menos **2 modelos** para comparar.\n\n"
            "Opciones:\n"
            "- Entrena 2 o más modelos en **🤖 Entrenamiento del modelo**\n"
            "- O entrena 1 modelo y aplícale tuning en **⚙️ Tuning de hiperparámetros**"
        )
        st.stop()

    # Banner de estado en tiempo real
    base_count  = sum(1 for k in pipelines if "✦ Tuned" not in k)
    tuned_count = sum(1 for k in pipelines if "✦ Tuned" in k)
    status_parts = [f"<strong>{base_count}</strong> modelo(s) base"]
    if tuned_count:
        status_parts.append(f"<strong>{tuned_count}</strong> modelo(s) tuneado(s)")
    if tuning_r:
        last_tuned = tuning_r.get("model", "")
        status_parts.append(
            f"Último tuning: <strong>{last_tuned}</strong> · "
            f"CV Accuracy: <strong>{tuning_r.get('best_cv_score', 0):.4f}</strong>"
        )
    st.markdown(
        '<div class="info-box">✨ <strong>Estado actual:</strong> ' +
        " &nbsp;·&nbsp; ".join(status_parts) +
        '</div>',
        unsafe_allow_html=True,
    )

    X_te = st.session_state.get("X_test")
    y_te = st.session_state.get("y_test")

    # Calcular métricas completas para todos los modelos entrenados
    lang_labels = sorted(set(y_te))
    lang_names  = [LANG_META[l]["name"] for l in lang_labels]

    def _base_name(n: str) -> str:
        """Nombre del modelo sin el sufijo Tuned."""
        return n.replace(" ✦ Tuned", "").strip()

    def _is_tuned(n: str) -> bool:
        return "✦ Tuned" in n

    # CV scores por modelo tuneado (guardados durante el tuning)
    cv_scores = st.session_state.get("tuning_cv_scores") or {}

    rows = []
    for name, pipe in pipelines.items():
        ev = evaluar_modelo(pipe, X_te, y_te, lang_labels, lang_names)
        row = {
            "Modelo":    name,
            "Tuned":     _is_tuned(name),
            "Accuracy":  round(ev["acc"], 4),
            "Macro F1":  round(ev["f1"],  4),
            "Precision": None,
            "Recall":    None,
            "CV Score":  cv_scores.get(name, None),   # solo para modelos tuneados
            "Errores":   int((ev["y_pred"] != y_te).sum()),
        }
        # Extraer precision/recall del classification report
        # NOTA: iloc[:-0] == iloc[:0] (vacío) — usar .mean() directamente
        rd = ev["report_df"]
        if "precision" in rd.columns and len(rd) > 0:
            row["Precision"] = round(float(rd["precision"].mean()), 4)
            row["Recall"]    = round(float(rd["recall"].mean()),    4)
        rows.append(row)

    comp_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best    = comp_df.iloc[0]
    worst   = comp_df.iloc[-1]

    # ── KPIs destacados ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, str(len(pipelines)),          "MODELOS COMPARADOS"),
        (k2, f"{best['Accuracy']:.4f}",    "MEJOR ACCURACY"),
        (k3, best["Modelo"],               "MEJOR MODELO"),
        (k4, f"{best['Macro F1']:.4f}",    "MEJOR F1"),
    ]:
        with col:
            st.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.markdown(section_header("📋 Tabla de métricas"), unsafe_allow_html=True)

    # Banner informativo si hay modelos tuneados
    tuned_names = [r["Modelo"] for r in rows if r.get("Tuned")]
    if tuned_names:
        badge_list = " &nbsp;·&nbsp; ".join(
            f'<span style="color:#c8ff00">{n}</span>' for n in tuned_names
        )
        st.markdown(
            '<div class="info-box">'
            f'<strong>✦ Modelos con tuning aplicado:</strong> {badge_list}<br>'
            'Los modelos marcados con <strong>✦ Tuned</strong> han sido optimizados con GridSearchCV.<br>'
            '<strong>⚠️ CV Score ≠ Test Accuracy:</strong> el CV Score se calcula sobre el set de '
            'entrenamiento con validación cruzada y puede ser mayor que el Test Accuracy (que se mide '
            'sobre datos nunca vistos durante el entrenamiento).'
            '</div>',
            unsafe_allow_html=True,
        )

    def _color_acc(v):
        if isinstance(v, float):
            c = "#5BAD6F" if v >= 0.95 else "#C8A84B" if v >= 0.90 else "#E8603C"
            return f"color:{c};font-weight:700"
        return ""

    # Construir columnas a mostrar
    show_cv    = any(r.get("CV Score") is not None for r in rows)
    base_cols  = ["Modelo", "Accuracy", "Macro F1", "Precision", "Recall"]
    extra_cols = (["CV Score"] if show_cv else []) + ["Errores"]
    display_cols = base_cols + extra_cols

    # Formatear valores que pueden ser None
    disp_df = comp_df.drop(columns=["Tuned"], errors="ignore").copy()
    for col in ["Precision", "Recall"]:
        if col in disp_df.columns:
            disp_df[col] = disp_df[col].apply(
                lambda v: f"{v:.4f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "—"
            )
    if show_cv and "CV Score" in disp_df.columns:
        disp_df["CV Score"] = disp_df["CV Score"].apply(
            lambda v: f"{v:.4f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "—"
        )

    highlight_cols = [c for c in ["Accuracy", "Macro F1"] if c in display_cols]
    st.dataframe(
        disp_df[display_cols]
        .style
        .map(_color_acc, subset=highlight_cols)
        .format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}"})
        .highlight_max(subset=highlight_cols, color="#c8ff0022")
        .highlight_min(subset=["Errores"],    color="#E8603C22"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos ──────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(section_header("📊 Comparativa de Accuracy"), unsafe_allow_html=True)
        def _model_color(m: str) -> str:
            base = m.replace(" ✦ Tuned", "").strip()
            color = MODELS_CONFIG[base]["color"] if base in MODELS_CONFIG else ACC
            # Tuned models get a lighter/brighter variant
            return color if "✦ Tuned" not in m else color + "cc"

        model_colors = [_model_color(m) for m in comp_df["Modelo"]]
        fig = charts.grafico_comparacion_modelos(
            names  = comp_df["Modelo"].tolist(),
            values = comp_df["Accuracy"].tolist(),
            colors = model_colors,
            title  = "Test Accuracy por modelo",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown(section_header("🕸️ Radar de métricas"), unsafe_allow_html=True)
        fig2 = _radar_chart(comp_df)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico multi-métrica ─────────────────────────────────────────────────
    st.markdown(section_header("📈 Accuracy vs F1 por modelo"), unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG)
    for s in ax3.spines.values(): s.set_edgecolor(GRID)

    x   = np.arange(len(comp_df))
    w   = 0.38
    col_acc = [MODELS_CONFIG[m]["color"] if m in MODELS_CONFIG else ACC for m in comp_df["Modelo"]]
    col_f1  = [c + "99" for c in col_acc]   # misma color pero transparente

    b1 = ax3.bar(x - w/2, comp_df["Accuracy"], w, color=col_acc, edgecolor=BG, label="Accuracy")
    b2 = ax3.bar(x + w/2, comp_df["Macro F1"], w, color=col_f1,  edgecolor=BG, label="Macro F1")

    for bar, v in zip(list(b1) + list(b2), list(comp_df["Accuracy"]) + list(comp_df["Macro F1"])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=7, fontfamily="monospace")

    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_df["Modelo"], rotation=15, ha="right", color=TEXT, fontsize=8)
    ax3.set_ylim(max(0, comp_df[["Accuracy","Macro F1"]].min().min() - 0.05), 1.05)
    ax3.set_ylabel("Score", color=TEXT, fontsize=9)
    ax3.set_title("Accuracy vs Macro F1", color=ACC, fontsize=11)
    ax3.yaxis.grid(True, color=GRID); ax3.set_axisbelow(True)
    ax3.tick_params(axis="y", colors=TEXT, labelsize=8)
    ax3.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=TEXT)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── Análisis del mejor y peor modelo ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_best, col_worst = st.columns(2)

    with col_best:
        st.markdown(section_header("🏆 Mejor modelo"), unsafe_allow_html=True)
        _bname = best["Modelo"].replace(" ✦ Tuned", "").strip()
        color = MODELS_CONFIG[_bname]["color"] if _bname in MODELS_CONFIG else ACC
        best_acc = f"{best['Accuracy']:.4f}"
        best_f1  = f"{best['Macro F1']:.4f}"
        best_model_name = str(best['Modelo'])
        st.markdown(
            f'<div style="background:#141414;border:2px solid {color};border-radius:12px;'
            f'padding:20px 24px;text-align:center;">'
            f'<div style="font-size:2.5rem">🥇</div>'
            f'<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.5rem;'
            f'color:{color};margin:6px 0">{best_model_name}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#555;'
            f'letter-spacing:2px">ACCURACY {best_acc} · F1 {best_f1}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_worst:
        st.markdown(section_header("📉 Modelo con menor accuracy"), unsafe_allow_html=True)
        _wname = worst["Modelo"].replace(" ✦ Tuned", "").strip()
        color_w = MODELS_CONFIG[_wname]["color"] if _wname in MODELS_CONFIG else "#555"
        worst_acc = f"{worst['Accuracy']:.4f}"
        worst_f1  = f"{worst['Macro F1']:.4f}"
        worst_model_name = str(worst['Modelo'])
        st.markdown(
            f'<div style="background:#141414;border:2px solid {color_w}55;border-radius:12px;'
            f'padding:20px 24px;text-align:center;">'
            f'<div style="font-size:2.5rem">📊</div>'
            f'<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.5rem;'
            f'color:{color_w};margin:6px 0">{worst_model_name}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#555;'
            f'letter-spacing:2px">ACCURACY {worst_acc} · F1 {worst_f1}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Errores por modelo ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🔍 Análisis de errores"), unsafe_allow_html=True)

    err_df = _tabla_errores(pipelines, X_te, y_te)
    total  = len(y_te)

    fig4, ax4 = plt.subplots(figsize=(8, max(3, len(pipelines) * 0.55)))
    fig4.patch.set_facecolor(BG); ax4.set_facecolor(BG)
    for s in ax4.spines.values(): s.set_edgecolor(GRID)

    err_colors = [
        "#5BAD6F" if v == err_df["Errores"].min() else
        "#E8603C" if v == err_df["Errores"].max() else "#4A90D9"
        for v in err_df["Errores"]
    ]
    bars = ax4.barh(err_df["Modelo"], err_df["Errores"], color=err_colors, edgecolor=BG)
    for bar, v in zip(bars, err_df["Errores"]):
        ax4.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{v} / {total}  ({v/total*100:.1f}%)",
                 va="center", color=TEXT, fontsize=8, fontfamily="monospace")
    ax4.set_xlabel("Muestras mal clasificadas", color=TEXT, fontsize=9)
    ax4.set_title("Errores en test set por modelo", color=ACC, fontsize=10)
    ax4.tick_params(colors=TEXT, labelsize=8)
    ax4.xaxis.grid(True, color=GRID); ax4.set_axisbelow(True)
    patches = [
        mpatches.Patch(color="#5BAD6F", label="Menos errores"),
        mpatches.Patch(color="#E8603C", label="Más errores"),
        mpatches.Patch(color="#4A90D9", label="Resto"),
    ]
    ax4.legend(handles=patches, fontsize=7, facecolor="#111", edgecolor="#333", labelcolor=TEXT)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # ── Sección dedicada: Baseline vs Tuned ─────────────────────────────────
    pairs = []   # (base_name, tuned_name)
    tuned_keys = [k for k in pipelines if "✦ Tuned" in k]
    for tk in tuned_keys:
        bk = tk.replace(" ✦ Tuned", "").strip()
        if bk in pipelines:
            pairs.append((bk, tk))

    if pairs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("✦ Baseline vs Tuned — Impacto del ajuste"), unsafe_allow_html=True)

        for base_k, tuned_k in pairs:
            base_row  = comp_df[comp_df["Modelo"] == base_k].iloc[0]  if base_k  in comp_df["Modelo"].values else None
            tuned_row = comp_df[comp_df["Modelo"] == tuned_k].iloc[0] if tuned_k in comp_df["Modelo"].values else None

            if base_row is None or tuned_row is None:
                continue

            delta_acc = (tuned_row["Accuracy"] - base_row["Accuracy"]) * 100
            delta_f1  = (tuned_row["Macro F1"] - base_row["Macro F1"]) * 100
            sign_acc  = f"+{delta_acc:.2f}%" if delta_acc >= 0 else f"{delta_acc:.2f}%"
            sign_f1   = f"+{delta_f1:.2f}%"  if delta_f1  >= 0 else f"{delta_f1:.2f}%"
            col_delta = "#5BAD6F" if delta_acc >= 0 else "#E8603C"
            bname     = base_k.replace(" ✦ Tuned", "").strip()
            model_color = MODELS_CONFIG[bname]["color"] if bname in MODELS_CONFIG else ACC

            # Construir el HTML como string sin indentación relativa al código Python
            # para evitar que Markdown interprete los espacios como bloque de código
            acc_base_val  = f"{base_row['Accuracy']:.4f}"
            acc_tuned_val = f"{tuned_row['Accuracy']:.4f}"
            card_html = (
                f'<div style="background:#141414;border:1px solid {model_color}44;'
                f'border-radius:12px;padding:20px 24px;margin-bottom:14px;">'
                f'<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1rem;'
                f'color:{model_color};margin-bottom:14px;">{base_k}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">'

                # Card: Baseline
                f'<div style="background:#0d0d0d;border-radius:8px;padding:12px;text-align:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#555;'
                f'letter-spacing:2px;text-transform:uppercase">Baseline Accuracy</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:1.5rem;font-weight:800;'
                f'color:#aaa">{acc_base_val}</div></div>'

                # Card: Tuned
                f'<div style="background:#0d0d0d;border-radius:8px;padding:12px;text-align:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#555;'
                f'letter-spacing:2px;text-transform:uppercase">Tuned Accuracy</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:1.5rem;font-weight:800;'
                f'color:{model_color}">{acc_tuned_val}</div></div>'

                # Card: Mejora Δ
                f'<div style="background:#0d0d0d;border-radius:8px;padding:12px;text-align:center;'
                f'border:1px solid {col_delta}44;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#555;'
                f'letter-spacing:2px;text-transform:uppercase">Mejora</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:1.5rem;font-weight:800;'
                f'color:{col_delta}">{sign_acc}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                f'color:{col_delta}">F1 {sign_f1}</div></div>'

                f'</div></div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # Mini gráfico baseline vs tuned
            fig_bt, ax_bt = plt.subplots(figsize=(6, 2.2))
            fig_bt.patch.set_facecolor(BG); ax_bt.set_facecolor(BG)
            for s in ax_bt.spines.values(): s.set_edgecolor(GRID)
            metrics   = ["Accuracy", "Macro F1"]
            base_vals = [base_row["Accuracy"], base_row["Macro F1"]]
            tune_vals = [tuned_row["Accuracy"], tuned_row["Macro F1"]]
            x = np.arange(len(metrics)); w = 0.32
            ax_bt.bar(x - w/2, base_vals,  w, color="#4A90D9",   label="Baseline",  edgecolor=BG)
            ax_bt.bar(x + w/2, tune_vals,  w, color=model_color, label="✦ Tuned", edgecolor=BG)
            for xi, (bv, tv) in enumerate(zip(base_vals, tune_vals)):
                ax_bt.text(xi - w/2, bv + 0.002, f"{bv:.4f}", ha="center", va="bottom",
                           color=TEXT, fontsize=7, fontfamily="monospace")
                ax_bt.text(xi + w/2, tv + 0.002, f"{tv:.4f}", ha="center", va="bottom",
                           color=TEXT, fontsize=7, fontfamily="monospace")
            ax_bt.set_xticks(x); ax_bt.set_xticklabels(metrics, color=TEXT, fontsize=9)
            ax_bt.tick_params(axis="y", colors=TEXT, labelsize=7)
            min_val = min(base_vals + tune_vals)
            ax_bt.set_ylim(max(0, min_val - 0.04), 1.02)
            ax_bt.yaxis.grid(True, color=GRID); ax_bt.set_axisbelow(True)
            ax_bt.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=TEXT)
            plt.tight_layout()
            st.pyplot(fig_bt, use_container_width=True)
            plt.close(fig_bt)

    # ── Consejo final ─────────────────────────────────────────────────────────
    margin = round((best["Accuracy"] - worst["Accuracy"]) * 100, 2)
    st.markdown(info_box(
        f"<strong>Diferencia entre mejor y peor:</strong> {margin:.2f}% de accuracy<br>"
        f"<strong>Recomendación:</strong> Usa <strong>{best['Modelo']}</strong> para producción "
        f"o ajusta sus hiperparámetros en la página ⚙️ Tuning para intentar mejorar aún más."
    ), unsafe_allow_html=True)
