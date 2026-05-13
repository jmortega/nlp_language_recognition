# NLP Language Recognition
### Pipeline completo de ML supervisado · Streamlit + scikit-learn

> Aplicación de reconocimiento de idioma en texto libre basada en **character n-gram TF-IDF** y múltiples clasificadores de scikit-learn.

---

## Tabla de contenidos

- [Descripción del problema](#-descripción-del-problema)
- [Requisitos del sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Ejecución](#-ejecución)
- [Dataset](#-dataset)
- [Pipeline ML](#-pipeline-ml)
- [Resultados](#-resultados)
- [Hiperparámetros](#️-ajuste-de-hiperparámetros)
- [Cómo navegar la aplicación](#-cómo-navegar-la-aplicación)

---

## Descripción del problema

| Campo | Detalle |
|---|---|
| **Tarea** | Clasificación de texto multiclase |
| **Tipo de aprendizaje** | Supervisado |
| **Entrada** | Texto en bruto (cualquier longitud) |
| **Salida** | Código de idioma: `en`, `es`, `fr`, `de`, `it`, `pt` |
| **Métrica principal** | Accuracy + Macro F1-score |

**Idiomas soportados:** 🇬🇧 English · 🇪🇸 Spanish · 🇫🇷 French · 🇩🇪 German · 🇮🇹 Italian · 🇵🇹 Portuguese

---

## 💻 Requisitos del sistema

- Python **3.9+**
- pip 21+
- ~200 MB de espacio en disco (modelos en memoria)

---

## 🔧 Instalación

### 1. Clonar el proyecto

```bash
git clone https://github.com/jmortega/nlp_language_recognition
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install streamlit==1.35.0 \
            scikit-learn==1.5.0 \
            pandas==2.2.0 \
            numpy==1.26.4 \
            matplotlib==3.9.0 \
            seaborn==0.13.2
```

O con el fichero de requisitos:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit>=1.30.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Estructura del proyecto

```
nlp_language_recognition/
├── app.py                    ← aplicación Streamlit principal
├── multilingual_dataset.csv  ← dataset de 594 frases en 6 idiomas
├── requirements.txt          ← dependencias Python
└── README.md                 ← este fichero
```

---

## Ejecución

```bash
streamlit run app.py
``` 

La aplicación se abre automáticamente en `http://localhost:8501`

> **Nota:** el fichero `multilingual_dataset.csv` debe estar en el mismo directorio que `app.py`.

---

## Dataset

| Característica | Valor |
|---|---|
| **Total de registros** | 594 filas |
| **Idiomas** | 6 (99 muestras por idioma) |
| **Longitud media** | ~68 caracteres / ~11 palabras por frase |
| **Fuente** | Generado con frases sobre ciencia, tecnología, cultura y sociedad |
| **Balance de clases** | Perfectamente balanceado (99 filas por clase) |

### Distribución por idioma

| Idioma | Código | Filas |
|---|---|---|
| 🇬🇧 English | `en` | 99 |
| 🇪🇸 Spanish | `es` | 99 |
| 🇫🇷 French | `fr` | 99 |
| 🇩🇪 German | `de` | 99 |
| 🇮🇹 Italian | `it` | 99 |
| 🇵🇹 Portuguese | `pt` | 99 |

---

## Pipeline ML

```
Texto bruto
    │
    ▼
┌─────────────────────────────────────┐
│  Preprocesamiento                   │
│  • Lowercase                        │
│  • Normalización de espacios        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Feature Engineering                │
│  TfidfVectorizer(                   │
│    analyzer='char_wb',              │  ← n-gramas de caracteres
│    ngram_range=(2, 4),              │  ← bi, tri y tetragrams
│    max_features=50_000,             │
│    sublinear_tf=True                │
│  )                                  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Clasificador (sklearn)             │
│  • Logistic Regression              │
│  • Linear SVC                       │
│  • Random Forest                    │
│  • Gradient Boosting                │
│  • Naive Bayes                      │
└────────────────┬────────────────────┘
                 │
                 ▼
         Código de idioma
         en / es / fr / de / it / pt
```

### Por qué character n-grams

Los n-gramas de caracteres son el estándar para detección de idioma porque capturan patrones morfológicos únicos sin necesidad de diccionarios:

| N-grama | Idioma asociado |
|---|---|
| `ão`, `ção`, `ões` | Portugués |
| `sch`, `ung`, `keit` | Alemán |
| `eau`, `eux`, `que` | Francés |
| `ción`, `mente`, `dad` | Español |
| `zione`, `mente`, `ità` | Italiano |
| `tion`, `ing`, `the` | Inglés |

---

## Resultados

Partición train/test: **80% / 20%** estratificada (475 train · 119 test)

### Comparativa de modelos (test set)

| Modelo | Test Accuracy | Macro F1 | CV Accuracy (5-fold) | Tiempo (s) |
|---|---|---|---|---|
| **Logistic Regression** | **0.9748** | **0.9750** | 0.9832 | 4.6 |
| Linear SVC | 0.9748 | 0.9748 | — | **0.4** |
| Random Forest | 0.9748 | 0.9746 | — | 2.0 |
| Naive Bayes | 0.9748 | 0.9748 | — | **0.1** |
| Gradient Boosting | 0.9580 | 0.9584 | — | 30.4 |

### Classification Report — Mejor modelo (Logistic Regression)

```
              precision    recall  f1-score   support

      German       1.00      1.00      1.00        19
     English       1.00      0.95      0.97        20
     Spanish       0.95      1.00      0.98        20
      French       0.95      1.00      0.98        20
     Italian       0.95      1.00      0.98        20
  Portuguese       1.00      0.90      0.95        20

    accuracy                           0.97       119
   macro avg       0.98      0.98      0.97       119
weighted avg       0.98      0.97      0.97       119
```

### Observaciones

- **Alemán** — clasificado con precision y recall perfectos (1.00). Los n-gramas tipo `sch`, `ung`, `keit` son extremadamente distintivos.
- **Portugués** — 2 falsos negativos. Comparte muchos patrones con el español (`ção` ↔ `ción`), lo que genera la mayor confusión del sistema.
- **Inglés** — 1 falso negativo. La ausencia de caracteres acentuados lo hace ambiguo en textos cortos.
- **Gradient Boosting** — menor accuracy (0.958) y tiempo de entrenamiento 7× superior.

---

## Ajuste de hiperparámetros

Grid Search con validación cruzada de 5 folds sobre Logistic Regression:

| Parámetro | Valores explorados | Mejor valor |
|---|---|---|
| `C` (regularización) | `[0.1, 1, 10]` | **10** |
| `solver` | `['lbfgs', 'liblinear']` | **lbfgs** |

| Métrica | Baseline | Tuned | Mejora |
|---|---|---|---|
| Test Accuracy | 0.9748 | 0.9748 | +0.00% |
| CV Accuracy (5-fold) | 0.9832 | **0.9895** | **+0.63%** |

> El ajuste de hiperparámetros mejoró la **generalización** del modelo (CV score) aunque el accuracy en test se mantuvo estable, lo que indica que el baseline ya estaba bien ajustado para este dataset.

---

## Estructura de la aplicación

La app está organizada en **6 secciones** accesibles desde la barra lateral:

| Sección | Contenido |
|---|---|
| 📊 **Problem & Dataset** | Formulación del problema, estadísticas del dataset, distribución de clases |
| 🔬 **Feature Engineering** | Inspector de n-gramas, n-gramas distintivos por idioma, configuración train/test split |
| 🤖 **Model Training** | Selección y entrenamiento de modelos, tabla comparativa, gráfico de barras |
| 📈 **Evaluation** | Matriz de confusión, métricas por clase, análisis de errores |
| ⚙️ **Hyperparameter Tuning** | Grid Search, comparativa baseline vs tuned, parámetros óptimos |
| 🌐 **Live Prediction** | Texto libre → predicción en tiempo real con barra de confianza |

---

## Referencias

- Cavnar, W. B. & Trenkle, J. M. (1994). *N-Gram-Based Text Categorization.*
- Scikit-learn: Machine Learning in Python — [scikit-learn.org](https://scikit-learn.org)
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
