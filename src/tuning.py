"""
tuning.py
---------
Lógica de ajuste de hiperparámetros con GridSearchCV.
Acepta un param_grid personalizado (desde la UI) en lugar de leerlo siempre de config.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODELS_CONFIG, get_clf
from src.models import construir_pipeline


def ajustar_hiperparametros(
    model_name:   str,
    X_train:      np.ndarray,
    X_test:       np.ndarray,
    y_train:      np.ndarray,
    y_test:       np.ndarray,
    pipeline_base,
    cv_folds:     int = 5,
    custom_params: dict | None = None,   # ← grid personalizado desde la UI
) -> dict:
    """
    Ejecuta GridSearchCV y compara con el pipeline baseline.

    Args:
        custom_params: si se proporciona, sustituye al grid de MODELS_CONFIG.

    Returns dict con: model, best_params, best_cv_score, acc_base, acc_tuned,
                      f1_base, f1_tuned, cv_results_df, best_pipeline.
    """
    param_grid = custom_params if custom_params is not None else MODELS_CONFIG[model_name]["params"]
    pipe_nueva = construir_pipeline(model_name)

    gs = GridSearchCV(
        pipe_nueva,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_pipeline = gs.best_estimator_

    y_pred_base  = pipeline_base.predict(X_test)
    y_pred_tuned = best_pipeline.predict(X_test)

    acc_base  = accuracy_score(y_test, y_pred_base)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    f1_base   = f1_score(y_test, y_pred_base,  average="macro")
    f1_tuned  = f1_score(y_test, y_pred_tuned, average="macro")

    cv_results_df = pd.DataFrame(gs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    return {
        "model":         model_name,
        "best_params":   gs.best_params_,
        "best_cv_score": round(gs.best_score_, 4),
        "acc_base":      acc_base,
        "acc_tuned":     acc_tuned,
        "f1_base":       f1_base,
        "f1_tuned":      f1_tuned,
        "cv_results_df": cv_results_df,
        "best_pipeline": best_pipeline,
    }
