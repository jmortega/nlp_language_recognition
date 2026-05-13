"""
config.py
---------
Constantes globales: metadatos de idiomas y configuración de modelos.
Incluye MLP y XGBoost además de los modelos originales.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# Dataset
DATASET_PATH = "multilingual_dataset.csv"

# Idiomas soportados
LANG_META: dict[str, dict] = {
    "en": {"name": "English",    "flag": "🇬🇧", "color": "#4A90D9"},
    "es": {"name": "Spanish",    "flag": "🇪🇸", "color": "#E8603C"},
    "fr": {"name": "French",     "flag": "🇫🇷", "color": "#6B8EBF"},
    "de": {"name": "German",     "flag": "🇩🇪", "color": "#C8A84B"},
    "it": {"name": "Italian",    "flag": "🇮🇹", "color": "#5BAD6F"},
    "pt": {"name": "Portuguese", "flag": "🇵🇹", "color": "#9B59B6"},
}

# ── Modelos sklearn ────────────────────────────────────────────────────────────
# Cada entrada tiene:
#   clf     → instancia del clasificador (se recrea en cada entrenamiento)
#   params  → grid de búsqueda para GridSearchCV (prefijo clf__ por el Pipeline)
#   color   → color del modelo en gráficos
#   param_meta → metadata para generar widgets interactivos en la UI
#
# param_meta keys por parámetro:
#   type    → "float" | "int" | "select" | "tuple_select" | "bool_none"
#   values  → lista de valores por defecto
#   min/max → rango mínimo/máximo para sliders numéricos
#   options → opciones disponibles para multiselect

MODELS_CONFIG: dict[str, dict] = {
    "Logistic Regression": {
        "clf": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "clf__C":      [0.1, 1, 10],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "param_meta": {
            "clf__C": {
                "type": "float", "values": [0.1, 1.0, 10.0],
                "min": 0.001, "max": 100.0,
                "help": "Inverso de la regularización — valores altos = menos regularización",
            },
            "clf__solver": {
                "type": "select", "values": ["lbfgs", "liblinear"],
                "options": ["lbfgs", "liblinear", "saga"],
                "help": "Algoritmo de optimización. lbfgs es robusto; liblinear funciona bien con L1",
            },
        },
        "color": "#4A90D9",
    },

    "Linear SVC": {
        "clf": LinearSVC(max_iter=2000, random_state=42),
        "params": {"clf__C": [0.1, 1, 5, 10]},
        "param_meta": {
            "clf__C": {
                "type": "float", "values": [0.1, 1.0, 5.0, 10.0],
                "min": 0.001, "max": 100.0,
                "help": "Parámetro de margen — valores pequeños = margen más amplio (más regularización)",
            },
        },
        "color": "#E8603C",
    },

    "Random Forest": {
        "clf": RandomForestClassifier(random_state=42),
        "params": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth":    [None, 10, 20],
        },
        "param_meta": {
            "clf__n_estimators": {
                "type": "int", "values": [50, 100, 200],
                "min": 10, "max": 500,
                "help": "Número de árboles — más árboles = más robusto pero más lento",
            },
            "clf__max_depth": {
                "type": "int_none", "values": [10, 20],
                "min": 1, "max": 50,
                "help": "Profundidad máxima del árbol. None = sin límite (puede causar overfitting)",
            },
        },
        "color": "#5BAD6F",
    },

    "Gradient Boosting": {
        "clf": GradientBoostingClassifier(random_state=42),
        "params": {
            "clf__n_estimators":  [50, 100],
            "clf__learning_rate": [0.05, 0.1, 0.2],
        },
        "param_meta": {
            "clf__n_estimators": {
                "type": "int", "values": [50, 100],
                "min": 10, "max": 300,
                "help": "Número de boosting rounds — más = mejor ajuste pero más lento",
            },
            "clf__learning_rate": {
                "type": "float", "values": [0.05, 0.1, 0.2],
                "min": 0.001, "max": 1.0,
                "help": "Tasa de aprendizaje — valores pequeños + más estimadores = mejor generalización",
            },
        },
        "color": "#C8A84B",
    },

    "Naive Bayes": {
        "clf": MultinomialNB(),
        "params": {"clf__alpha": [0.1, 0.5, 1.0, 2.0]},
        "param_meta": {
            "clf__alpha": {
                "type": "float", "values": [0.1, 0.5, 1.0, 2.0],
                "min": 0.001, "max": 10.0,
                "help": "Suavizado de Laplace — evita probabilidades cero para n-gramas no vistos",
            },
        },
        "color": "#9B59B6",
    },

    "MLP Classifier": {
        "clf": MLPClassifier(max_iter=300, random_state=42),
        "params": {
            "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "clf__alpha":             [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.001, 0.01],
        },
        "param_meta": {
            "clf__hidden_layer_sizes": {
                "type": "tuple_select",
                "values": ["(64,)", "(128,)", "(128, 64)", "(256,)", "(256, 128)"],
                "help": "Arquitectura de capas ocultas — cada tupla define neuronas por capa",
            },
            "clf__alpha": {
                "type": "float", "values": [0.0001, 0.001, 0.01],
                "min": 0.00001, "max": 1.0,
                "help": "Regularización L2 — penaliza pesos grandes para evitar overfitting",
            },
            "clf__learning_rate_init": {
                "type": "float", "values": [0.001, 0.01],
                "min": 0.0001, "max": 0.1,
                "help": "Tasa de aprendizaje inicial del optimizador Adam",
            },
        },
        "color": "#E91E8C",
    },

    "XGBoost": {
        "clf": None,   # Se inicializa dinámicamente para evitar ImportError si no está instalado
        "params": {
            "clf__n_estimators":  [50, 100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth":     [3, 5, 7],
        },
        "param_meta": {
            "clf__n_estimators": {
                "type": "int", "values": [50, 100, 200],
                "min": 10, "max": 500,
                "help": "Número de árboles en el ensemble",
            },
            "clf__learning_rate": {
                "type": "float", "values": [0.05, 0.1, 0.2],
                "min": 0.001, "max": 1.0,
                "help": "Shrinkage — valores bajos necesitan más estimadores pero generalizan mejor",
            },
            "clf__max_depth": {
                "type": "int", "values": [3, 5, 7],
                "min": 1, "max": 15,
                "help": "Profundidad máxima de cada árbol — valores altos pueden causar overfitting",
            },
        },
        "color": "#00BCD4",
    },
}


def get_clf(model_name: str):
    """Devuelve una instancia fresca del clasificador (gestiona XGBoost de forma segura)."""
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder
            return XGBClassifier(
                n_estimators=100, random_state=42,
                use_label_encoder=False, eval_metric="mlogloss",
                verbosity=0,
            )
        except ImportError:
            raise ImportError("XGBoost no está instalado. Ejecuta: pip install xgboost")
    if model_name == "MLP Classifier":
        return MLPClassifier(max_iter=300, random_state=42)
    return MODELS_CONFIG[model_name]["clf"].__class__(
        **{k: v for k, v in MODELS_CONFIG[model_name]["clf"].get_params().items()}
    )


# Parámetros TF-IDF por defecto
TFIDF_DEFAULTS = {
    "analyzer":     "char_wb",
    "ngram_range":  (2, 4),
    "max_features": 50_000,
    "sublinear_tf": True,
    "strip_accents": None,
}

DEFAULT_TEST_SIZE    = 0.2
DEFAULT_RANDOM_STATE = 42
