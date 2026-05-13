"""
models.py
---------
Construcción de pipelines sklearn, entrenamiento, validación cruzada y métricas de evaluación.
Actualizado para usar get_clf() y soportar MLP y XGBoost.
"""

from __future__ import annotations
import ast
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
)

from src.config import (
    MODELS_CONFIG, TFIDF_DEFAULTS,
    DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE,
    get_clf,
)


def construir_pipeline(
    model_name: str,
    ngram_range: tuple[int, int] = (2, 4),
    max_features: int = 50_000,
) -> Pipeline:
    """Crea TfidfVectorizer → [to_dense] → Clasificador pipeline no entrenado."""
    tfidf_params = {**TFIDF_DEFAULTS, "ngram_range": ngram_range, "max_features": max_features}
    clf = get_clf(model_name)

    # MLP y XGBoost requieren matrices densas (no sparse)
    needs_dense = model_name in ("MLP Classifier", "XGBoost")
    to_dense = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)

    # XGBoost necesita LabelEncoder para etiquetas string
    if model_name == "XGBoost":
        from sklearn.preprocessing import LabelEncoder
        from sklearn.base import BaseEstimator, ClassifierMixin

        class XGBWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, xgb):
                self.xgb = xgb
                self.le  = LabelEncoder()
            def fit(self, X, y):
                y_enc = self.le.fit_transform(y)
                self.xgb.fit(X, y_enc)
                self.classes_ = self.le.classes_
                return self
            def predict(self, X):
                return self.le.inverse_transform(self.xgb.predict(X))
            def predict_proba(self, X):
                return self.xgb.predict_proba(X)
            def get_params(self, deep=True):
                return self.xgb.get_params(deep=deep)
            def set_params(self, **params):
                self.xgb.set_params(**params)
                return self

        clf = XGBWrapper(clf)

    steps = [("tfidf", TfidfVectorizer(**tfidf_params))]
    if needs_dense:
        steps.append(("to_dense", to_dense))
    steps.append(("clf", clf))

    return Pipeline(steps)


def dividir_datos(
    X: np.ndarray, y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def entrenar_modelo(
    model_name: str,
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    ngram_range: tuple[int, int] = (2, 4),
    cv_folds: int = 5,
) -> dict:
    pipe = construir_pipeline(model_name, ngram_range=ngram_range)

    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    try:
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring="accuracy").mean()
    except Exception:
        cv_acc = 0.0

    return {
        "model": model_name, "pipeline": pipe,
        "acc": acc, "f1": f1, "cv_acc": cv_acc,
        "train_time": elapsed, "y_pred": y_pred,
    }


def entrenar_modelos(
    model_names: list[str],
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    ngram_range: tuple[int, int] = (2, 4),
    progress_callback=None,
) -> tuple[dict, pd.DataFrame]:
    pipelines: dict = {}
    rows: list[dict] = []

    for i, name in enumerate(model_names):
        if progress_callback:
            progress_callback(i, len(model_names), name)
        resultado = entrenar_modelo(name, X_train, X_test, y_train, y_test, ngram_range)
        pipelines[name] = resultado["pipeline"]
        rows.append({
            "Modelo":               name,
            "Test Accuracy":        resultado["acc"],
            "Macro F1":             resultado["f1"],
            "CV Accuracy (5-fold)": resultado["cv_acc"],
            "Train Time (s)":       resultado["train_time"],
        })

    results_df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
    return pipelines, results_df


def evaluar_modelo(
    pipeline, X_test, y_test,
    lang_labels: list[str], lang_names: list[str],
) -> dict:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_test, y_pred, labels=lang_labels)
    report_str  = classification_report(y_test, y_pred, labels=lang_labels, target_names=lang_names)
    report_dict = classification_report(y_test, y_pred, labels=lang_labels, target_names=lang_names, output_dict=True)
    report_df   = pd.DataFrame(report_dict).T.iloc[:-3]
    errors_idx  = np.where(y_test != y_pred)[0]
    return {
        "acc": acc, "f1": f1,
        "precision": precision, "recall": recall,
        "cm": cm,
        "report_str": report_str, "report_df": report_df,
        "y_pred": y_pred, "errors_idx": errors_idx,
    }


def predecir(pipeline, texto_limpio: str) -> dict:
    pred      = pipeline.predict([texto_limpio])[0]
    has_proba = hasattr(pipeline, "predict_proba") and callable(pipeline.predict_proba)
    if has_proba:
        probs   = pipeline.predict_proba([texto_limpio])[0]
        classes = pipeline.classes_
        return {"pred": pred, "has_proba": True, "proba": dict(zip(classes, probs)), "decision": None}
    else:
        scores  = pipeline.decision_function([texto_limpio])[0]
        classes = pipeline.classes_
        return {"pred": pred, "has_proba": False, "proba": None, "decision": dict(zip(classes, scores))}
