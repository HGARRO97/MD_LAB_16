# LAB-S16: Aplicaciones de Machine Learning  
**Proyecto de Clasificación - Adult Income Dataset**

---

## Estructura del Proyecto

Laboratorio_16_Aplicaciones_ML/
│
├── data/
│ ├── raw/ # Datos originales (.data)
│ ├── processed/ # Datos limpios, transformados y de modelado
│
├── outputs/
│ ├── 01_diagnostico/ # Reportes y gráficos del análisis exploratorio
│ ├── 02_preparacion/ # Tablas HTML y gráficos de transformación, IV, etc.
│ ├── 03_modelado/ # Resultados de evaluación y ajustes
│
├── reports/ # Presentaciones teóricas
│
├── src/
│ ├── data/ # Carga y exploración de datos
│ ├── preprocessing/ # Imputación, transformación, outliers, IV, etc.
│ ├── model/ # Entrenamiento y evaluación de modelos
│
├── notebooks/
│ ├── 01_Analisis_exploratorio.ipynb
│ ├── 02_Preparacion_datos.ipynb
│ ├── 03_Modelado_ensamblaje.ipynb
│
├── requirements.txt
└── README.md


---

## Objetivo del Proyecto

Predecir si una persona gana más de $50K/año utilizando atributos demográficos del **Adult Income Dataset** (UCI). El flujo completo abarca desde la carga de datos hasta la selección del mejor modelo con búsqueda de hiperparámetros.

---

## Flujo de Trabajo

### 1. Exploración inicial (`01_Analisis_exploratorio.ipynb`)
- Resumen de tipos de variables, valores nulos, proporciones.
- Gráficos de barras para variables categóricas y distribuciones para numéricas.
- Guardado en: `outputs/01_diagnostico/`

### 2. Preparación de datos (`02_Preparacion_datos.ipynb`)
- Imputación de valores faltantes usando árboles de decisión.
- Simetrización según skew (log, sqrt, log1p).
- Escalamiento estándar.
- Detección y eliminación de outliers multivariados (Mahalanobis).
- Análisis de correlación.
- Cálculo de Information Value.
- Conversión de variables categóricas a dummies.
- División 80/20 y balanceo SMOTE.

### 3. Modelado (`03_Modelado_ensamblaje.ipynb`)
- Modelos entrenados: KNN, SVM, Regresión Logística, Árbol, Random Forest.
- Evaluación con `GridSearchCV` y `RandomizedSearchCV`.
- Métrica objetivo: `accuracy`.

---

## Modelos y Resultados

| Modelo             | Accuracy (Grid) | Accuracy (Random) | Hiperparámetros óptimos              |
|--------------------|------------------|--------------------|--------------------------------------|
| KNN                | 85.24%           | 85.24%             | `n_neighbors=7`, `weights='distance'` |
| SVM                | 84.92%           | 84.92%             | `C=10`, `kernel='rbf'`               |
| Regresión Logística| 82.77%           | 82.77%             | `C=10`, `penalty='l2'`               |
| Árbol de decisión  | 83.92%           | 83.92%             | `max_depth=10`, `criterion='gini'`   |
| **Random Forest**  | **86.61%**       | **86.61%**         | `n_estimators=100`, `max_depth=20`   |

> El modelo seleccionado fue **Random Forest**, por su mayor precisión y robustez.

---

## Librerías Principales

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn`, `imblearn`
- `scipy`, `logging`, `joblib`

---

## Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
