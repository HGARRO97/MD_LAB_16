import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

#### 1. CARGAR Y SEPARAR DATA DE ENTRENAMIENTO Y PRUEBA ####
def cargar_y_separar_datos(
    ruta_csv: str,
    target: str = "income",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Carga un CSV y separa los datos en conjuntos de entrenamiento y prueba.

    Args:
        ruta_csv (str): Ruta al archivo CSV procesado.
        target (str): Nombre de la variable objetivo.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para la separación aleatoria.

    Returns:
        X_train, X_test, y_train, y_test: Particiones de datos.
    """
    if not os.path.exists(ruta_csv):
        logging.error(f"No se encontró el archivo: {ruta_csv}")
        raise FileNotFoundError(f"No existe el archivo en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    logging.info(f"Archivo cargado correctamente: {ruta_csv}")
    logging.info(f"Dimensión total: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Separar X e y
    if target not in df.columns:
        logging.error(f"La columna objetivo '{target}' no existe en el DataFrame")
        raise KeyError(f"Columna '{target}' no encontrada")

    X = df.drop(columns=[target])
    y = df[target]

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info(f"Conjunto de entrenamiento: {X_train.shape[0]} filas")
    logging.info(f"Conjunto de prueba: {X_test.shape[0]} filas")

    return X_train, X_test, y_train, y_test

#### 2. BALANCEO DE CLASES ####

# Revisar balance de clases
def revisar_balance_clases(y_train):
    """
    Muestra el balance de clases (frecuencia y proporción) para el conjunto de entrenamiento.

    Args:
        y_train (pd.Series): Variable objetivo del conjunto de entrenamiento.
    """
    tabla = pd.concat([
        y_train.value_counts().rename("frecuencia"),
        y_train.value_counts(normalize=True).rename("proporcion")
    ], axis=1)
    tabla["proporcion"] = (tabla["proporcion"] * 100).round(2)

    logging.info("Balance de clases en el conjunto de entrenamiento:")
    logging.info(f"\n{tabla}")

# Realizar oversampling con SMOTE
from imblearn.over_sampling import SMOTE

def balancear_con_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logging.info(f"Aplicado SMOTE: {X_res.shape[0]} muestras totales después del balanceo")
    logging.info(f"Distribución posterior:\n{pd.Series(y_res).value_counts()}")

    return X_res, y_res

#### 3. BÚSQUEDA DE HIPERPARÁMETROS ####

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def entrenar_modelos_clasificacion(X_train, y_train, metodo="grid", cv=5, random_state=42, save_path=None):
    """
    Entrena modelos clásicos de clasificación utilizando búsqueda de hiperparámetros.

    Modelos: k-NN, SVM, Regresión Logística, Árbol de Decisión, Random Forest.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.
        metodo (str): "grid" para GridSearchCV, "random" para RandomizedSearchCV.
        cv (int): Número de particiones para validación cruzada.
        random_state (int): Semilla para reproducibilidad.
        save_path (str): Ruta para guardar el resumen de resultados en HTML.

    Returns:
        dict: Resultados por modelo, incluyendo mejor estimador, accuracy y mejores hiperparámetros.
    """
    resultados = {}
    resumen = []

    modelos = {
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        ),
        "SVM": (
            SVC(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"]
            }
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=random_state),
            {
                "max_depth": [3, 5, 10, None],
                "criterion": ["gini", "entropy"]
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state),
            {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20]
            }
        )
    }

    for nombre, (modelo, param_grid) in modelos.items():
        logging.info(f"Entrenando modelo: {nombre} ({metodo})")
        try:
            if metodo == "grid":
                searcher = GridSearchCV(modelo, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
            elif metodo == "random":
                searcher = RandomizedSearchCV(modelo, param_distributions=param_grid, n_iter=5, cv=cv,
                                              scoring="accuracy", n_jobs=-1, random_state=random_state)
            else:
                raise ValueError("El parámetro 'metodo' debe ser 'grid' o 'random'")

            searcher.fit(X_train, y_train)

            resultados[nombre] = {
                "mejor_estimador": searcher.best_estimator_,
                "mejor_accuracy": searcher.best_score_,
                "mejores_hiperparametros": searcher.best_params_
            }

            logging.info(f"Mejor accuracy para {nombre}: {searcher.best_score_:.4f}")
            logging.info(f"Hiperparámetros óptimos: {searcher.best_params_}")

            resumen.append({
                "modelo": nombre,
                "accuracy": round(searcher.best_score_, 4),
                "hiperparametros": searcher.best_params_
            })

        except Exception as e:
            logging.warning(f"No se pudo entrenar el modelo {nombre}: {e}")

    # Exportar resumen a HTML si se solicita
    if save_path:
        df_resumen = pd.DataFrame(resumen)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_resumen.to_html(save_path, index=False)
        logging.info(f"Resumen de resultados guardado en: {save_path}")

    return resultados
