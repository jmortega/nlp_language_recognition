"""
preprocessing.py
----------------
Funciones de limpieza de texto y carga del dataset
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from functools import lru_cache

from src.config import DATASET_PATH, LANG_META


# Limpieza de texto
def limpiar_texto(text: str) -> str:
    """Normalización mínima: lowercase + colapsar espacios."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def limpiar_serie(serie: pd.Series) -> np.ndarray:
    """Aplica limpiar_texto a una Serie y devuelve np.ndarray de strings."""
    return np.array([limpiar_texto(t) for t in serie])


# Carga del dataset
def cargar_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Carga el CSV y añade columnas derivadas útiles para el EDA.

    Columnas añadidas:
        text_length  — número de caracteres
        word_count   — número de palabras
        lang_name    — nombre legible del idioma
    """
    df = pd.read_csv(path)
    df["text_length"] = df["text"].str.len()
    df["word_count"]  = df["text"].str.split().str.len()
    df["lang_name"]   = df["language"].map(lambda c: LANG_META.get(c, {}).get("name", c))
    return df


def obtener_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (X_texts, y_labels) listos para sklearn."""
    X = limpiar_serie(df["text"])
    y = np.array(df["language"].tolist())
    return X, y


# Análisis de n-gramas
def extraer_ngrams_char(text: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    """Extrae n-gramas de caracteres de un texto"""
    text = limpiar_texto(text)
    ngrams: list[str] = []
    for n in range(n_min, n_max + 1):
        ngrams += [text[i : i + n] for i in range(len(text) - n + 1)]
    return ngrams
