import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

#### 1. IMPUTACIÓN DE DATOS ####

# Imputar registros de valores perdidos en  variables categóricas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def imputar_categoricas_por_arbol(df: pd.DataFrame, variables: list, f1_umbral: float = 0.7) -> pd.DataFrame:
    """
    Imputa múltiples variables categóricas con árbol de decisión si el F1-score supera un umbral.
    Si no, imputa con la categoría 'Desconocido'.

    Args:
        df (pd.DataFrame): DataFrame original
        variables (list): Lista de variables categóricas a imputar
        f1_umbral (float): Umbral mínimo de F1 para imputar con árbol

    Returns:
        pd.DataFrame: DataFrame con las variables imputadas
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    # Codificar variables categóricas
    le_dict = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le

    for var in variables:
        if var not in df.columns:
            logging.warning(f"La variable {var} no está en el DataFrame. Se omite.")
            continue

        df_known = df_encoded[df[var].notna()]
        df_missing = df_encoded[df[var].isna()]

        if df_missing.empty:
            logging.info(f"No hay valores faltantes en {var}.")
            continue

        # Seleccionar predictores sin nulos
        predictores_validos = df_known.drop(columns=[var]).columns[
            df_known.drop(columns=[var]).isna().sum() == 0
        ]

        if len(predictores_validos) == 0:
            logging.warning(f"No hay predictores válidos sin nulos para imputar {var}. Se asigna 'Desconocido'.")
            df[var] = df[var].fillna("Desconocido")
            continue

        X = df_known[predictores_validos]
        y = df_known[var]

        clf = DecisionTreeClassifier(max_depth=10, random_state=42)
        f1 = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted').mean()
        logging.info(f"{var} - F1-score promedio: {f1:.3f}")

        if f1 >= f1_umbral:
            clf.fit(X, y)
            X_missing = df_missing[predictores_validos]
            y_pred = clf.predict(X_missing)
            le_var = le_dict[var]
            imputados = le_var.inverse_transform(y_pred)
            df.loc[df[var].isna(), var] = imputados
            logging.info(f"{var}: Imputación completada con árbol (F1={f1:.3f})")
        else:
            df[var] = df[var].fillna("Desconocido")
            logging.warning(f"{var}: F1={f1:.3f} bajo. Imputado como 'Desconocido'")

    return df

# Eliminar variables específicas
def eliminar_variables(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Elimina variables específicas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        variables (list): Lista de nombres de columnas a eliminar.

    Returns:
        pd.DataFrame: DataFrame sin las variables indicadas.
    """
    df_filtrado = df.drop(columns=variables, errors='ignore')
    logging.info(f"Variables eliminadas: {variables}")
    return df_filtrado

# Separar variables numéricas y categóricas
def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa un DataFrame en variables numéricas y categóricas.

    Args:
        df (pd.DataFrame): DataFrame completo a procesar.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con solo variables numéricas.
            - pd.DataFrame: DataFrame con solo variables categóricas.
    """
    df_num = df.select_dtypes(include=["number"]).copy()
    df_cat = df.select_dtypes(exclude=["number"]).copy()

    logging.info("Variables numéricas: %s", list(df_num.columns))
    logging.info("Variables categóricas: %s", list(df_cat.columns))

    return df_num, df_cat

# Imputación de variables numéricas basada en porcentaje de nulos
def generar_diagnostico_knn(df: pd.DataFrame, columna: str, k: int = 3, save_dir: str = "outputs/02_preparacion") -> pd.DataFrame:
    """
    Genera un diagnóstico detallado de la imputación por KNN para una columna con valores nulos.
    Para cada fila con nulo:
        - Identifica los K vecinos más cercanos (distancia euclidiana ignorando nulos)
        - Obtiene sus distancias, valores y calcula el valor imputado (media)

    Args:
        df (pd.DataFrame): DataFrame original (preferentemente numérico).
        columna (str): Nombre de la columna a imputar.
        k (int): Número de vecinos más cercanos a considerar.
        save_dir (str): Ruta donde se guardará el archivo HTML con el diagnóstico.

    Returns:
        pd.DataFrame: DataFrame resumen del diagnóstico, también guardado como HTML.
    """
    df_work = df.copy()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    distancias = nan_euclidean_distances(df_work)
    diagnostico = []

    for idx in df_work[df_work[columna].isnull()].index:
        dists = pd.Series(distancias[idx], index=df_work.index).drop(idx)
        vecinos = dists.nsmallest(k)
        valores_vecinos = df_work.loc[vecinos.index, columna]
        promedio = valores_vecinos.mean()

        fila_resultado = {
            "fila_con_nan": idx,
            "valor_imputado": round(promedio, 2)
        }
        for i, (vec_idx, dist) in enumerate(vecinos.items(), start=1):
            fila_resultado[f"vecino_{i}"] = vec_idx
            fila_resultado[f"dist_{i}"] = round(dist, 2)
            fila_resultado[f"val_{i}"] = df_work.at[vec_idx, columna]
        diagnostico.append(fila_resultado)

    df_diag = pd.DataFrame(diagnostico)
    ruta_html = os.path.join(save_dir, f"diagnostico_knn_{columna}.html")
    df_diag.to_html(ruta_html, index=False)

    return df_diag

def imputar_nulos(df: pd.DataFrame, k: int = 3, umbrales: dict = None, save_dir: str = "outputs/02_preparacion") -> pd.DataFrame:
    """
    Imputa valores nulos según el porcentaje de faltantes:
    - < bajo: elimina registros
    - entre bajo y medio: imputa con KNN
    - > medio: elimina variable
    También guarda un diagnóstico detallado por variable imputada.

    Args:
        df (pd.DataFrame): DataFrame original.
        k (int): Número de vecinos para KNN.
        umbrales (dict): Diccionario con claves "bajo" y "medio".
        save_dir (str): Carpeta donde guardar los diagnósticos.

    Returns:
        pd.DataFrame: DataFrame imputado.
    """
    if umbrales is None:
        umbrales = {"bajo": 0.03, "medio": 0.15}

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    total = len(df)
    df_resultado = df.copy()

    for col in df.columns:
        nulos = df[col].isnull().sum()
        porcentaje = nulos / total

        if nulos == 0:
            continue
        elif porcentaje < umbrales["bajo"]:
            df_resultado = df_resultado[df_resultado[col].notnull()]
            logging.info(f"{col}: Se eliminaron {nulos} filas con nulos (<3%)")
        elif porcentaje <= umbrales["medio"]:
            diagnostico = generar_diagnostico_knn(df_resultado.select_dtypes(include='number'), col, k)
            diagnostico_path = os.path.join(save_dir, f"diagnostico_knn_{col}.html")
            diagnostico.to_html(diagnostico_path, index=False)
            logging.info(f"{col}: Imputado con KNN. Diagnóstico guardado en {diagnostico_path}")

            imputer = KNNImputer(n_neighbors=k)
            df_resultado[[col]] = imputer.fit_transform(df_resultado[[col]])
        else:
            df_resultado = df_resultado.drop(columns=col)
            logging.info(f"{col}: Eliminada por tener >15% de nulos")

    return df_resultado

#### 2. SIMETRIZACIÓN DE DATOS ####

from scipy.stats import skew

def detectar_asimetria(df_num: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula skewness de cada columna numérica y sugiere transformación.

    Args:
        df_num (pd.DataFrame): DataFrame solo con variables numéricas

    Returns:
        pd.DataFrame: Tabla con skewness y sugerencia de transformación
    """
    resultados = []
    for col in df_num.columns:
        skew_val = skew(df_num[col].dropna())
        if skew_val > 1:
            sugerencia = "log1p / sqrt"
        elif skew_val < -1:
            sugerencia = "exp / cuadrado"
        else:
            sugerencia = "sin transformación"
        resultados.append({
            "variable": col,
            "skewness": round(skew_val, 3),
            "sugerencia": sugerencia
        })
    resumen = pd.DataFrame(resultados)
    logging.info(f"Asimetría detectada:\n{resumen}")
    return resumen

def transformar_por_asimetria(df_num: pd.DataFrame, save_dir: str = None) -> tuple:
    """
    Aplica la mejor transformación a cada variable numérica según reducción de skew absoluto.
    Guarda un resumen HTML y gráficos de distribución transformada si se especifica `save_dir`.

    Args:
        df_num (pd.DataFrame): DataFrame solo con variables numéricas
        save_dir (str, optional): Ruta base para guardar resumen y gráficos

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame transformado
            - Tabla resumen de transformaciones
    """
    df_num = df_num.copy()
    resumen = []

    img_dir = os.path.join(save_dir) if save_dir else None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)

    for col in df_num.columns:
        x = df_num[col].dropna()
        skew_orig = skew(x)

        mejor_skew_abs = abs(skew_orig)
        mejor_skew_real = skew_orig
        mejor_trans = "ninguna"
        mejor_serie = df_num[col]

        if skew_orig > 1:
            try:
                skew_log = skew(np.log1p(x))
                skew_sqrt = skew(np.sqrt(x))

                if abs(skew_log) < mejor_skew_abs:
                    mejor_skew_abs = abs(skew_log)
                    mejor_skew_real = skew_log
                    mejor_trans = "log1p"
                    mejor_serie = np.log1p(df_num[col])
                if abs(skew_sqrt) < mejor_skew_abs:
                    mejor_skew_abs = abs(skew_sqrt)
                    mejor_skew_real = skew_sqrt
                    mejor_trans = "sqrt"
                    mejor_serie = np.sqrt(df_num[col])
            except Exception as e:
                logging.warning(f"{col}: error en transformación log/sqrt: {e}")

        elif skew_orig < -1:
            try:
                skew_exp = skew(np.exp(x))
                skew_square = skew(np.square(x))

                if abs(skew_exp) < mejor_skew_abs:
                    mejor_skew_abs = abs(skew_exp)
                    mejor_skew_real = skew_exp
                    mejor_trans = "exp"
                    mejor_serie = np.exp(df_num[col])
                if abs(skew_square) < mejor_skew_abs:
                    mejor_skew_abs = abs(skew_square)
                    mejor_skew_real = skew_square
                    mejor_trans = "cuadrado"
                    mejor_serie = np.square(df_num[col])
            except Exception as e:
                logging.warning(f"{col}: error en transformación exp/cuadrado: {e}")

        df_num[col] = mejor_serie

        # Graficar distribución
        if img_dir:
            try:
                plt.figure(figsize=(6, 4))
                df_num[col].plot(kind="hist", bins=30, density=True, alpha=0.6)
                df_num[col].plot(kind="kde", linewidth=2)
                plt.title(f"Distribución transformada: {col}")
                plt.xlabel(col)
                plt.ylabel("Densidad")
                plt.tight_layout()
                ruta_img = os.path.join(img_dir, f"{col}.png")
                plt.savefig(ruta_img)
                plt.close()
                logging.info(f"{col}: gráfico guardado en {ruta_img}")
            except Exception as e:
                logging.warning(f"{col}: error al guardar gráfico - {e}")

        # Logging resumen
        if mejor_trans == "ninguna":
            if abs(skew_orig) < 0.5:
                logging.info(f"{col}: sin transformación (simétrica, skew = {skew_orig:.2f})")
            else:
                logging.info(f"{col}: sin transformación (ninguna mejora el skew, original = {skew_orig:.2f})")
        else:
            logging.info(f"{col}: {mejor_trans} aplicada (skew original = {skew_orig:.2f}, final = {mejor_skew_real:.2f})")

        resumen.append({
            "variable": col,
            "skew_original": round(skew_orig, 3),
            "transformacion_aplicada": mejor_trans,
            "skew_final": round(mejor_skew_real, 3)
        })

    tabla_resumen = pd.DataFrame(resumen)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ruta_html = os.path.join(save_dir, "resumen_transformaciones.html")
        tabla_resumen.to_html(ruta_html, index=False)
        logging.info(f"Resumen de transformaciones guardado en HTML: {ruta_html}")

    return df_num, tabla_resumen

#### 3. NORMALIZACIÓN DE DATOS ####

# Estandarización de variables numéricas
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def estandarizar_variables(df_num: pd.DataFrame, save_dir: str = None) -> pd.DataFrame:
    """
    Aplica estandarización (media 0, desviación 1) a variables numéricas
    y guarda gráficos de distribución si se indica `save_dir`.

    Args:
        df_num (pd.DataFrame): DataFrame con variables numéricas.
        save_dir (str, optional): Carpeta para guardar gráficos PNG.

    Returns:
        pd.DataFrame: DataFrame con variables estandarizadas.
    """
    scaler = StandardScaler()
    df_estandarizado = pd.DataFrame(scaler.fit_transform(df_num),
                               columns=df_num.columns,
                               index=df_num.index)

    logging.info("Estandarización aplicada a variables numéricas")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for col in df_estandarizado.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df_estandarizado[col], kde=True, bins=30, color='steelblue')
            plt.title(f"Distribución estandarizada: {col}")
            plt.xlabel(col)
            plt.ylabel("Frecuencia")
            plt.tight_layout()
            ruta_img = os.path.join(save_dir, f"{col}_estandarizado.png")
            plt.savefig(ruta_img)
            plt.close()
            logging.info(f"Gráfico guardado: {ruta_img}")

    return df_estandarizado

#### 4. DETECCIÓN DE OUTLIERS ####

# Detección de outliers univariados
def detectar_outliers_univariado(
    df_num: pd.DataFrame,
    ruta_salida: str = None
) -> pd.DataFrame:
    """
    Detecta outliers univariados en variables numéricas usando IQR.
    Opcionalmente guarda un resumen en HTML.

    Args:
        df_num (pd.DataFrame): DataFrame numérico
        ruta_salida (str): Ruta de salida para guardar HTML (opcional)

    Returns:
        pd.DataFrame: DataFrame booleano con True en posiciones de outliers
    """
    outliers = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    resumen_outliers = []

    for col in df_num.columns:
        Q1 = df_num[col].quantile(0.25)
        Q3 = df_num[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_sup = Q3 + 1.5 * IQR
        limite_inf = Q1 - 1.5 * IQR

        outliers[col] = (df_num[col] < limite_inf) | (df_num[col] > limite_sup)
        cantidad = outliers[col].sum()

        logging.info(f"{col}: {cantidad} outliers univariados")
        resumen_outliers.append({"variable": col, "cantidad_outliers": cantidad})

    # Exportar resumen si se indica
    if ruta_salida:
        Path(ruta_salida).mkdir(parents=True, exist_ok=True)
        df_resumen = pd.DataFrame(resumen_outliers)
        df_resumen.to_html(os.path.join(ruta_salida, "outliers_univariados.html"), index=False)

    return outliers

# Detección de outliers multivariados usando Mahalanobis
def detectar_outliers_mahalanobis(
    df_num: pd.DataFrame,
    umbral: float = 0.99,
    ruta_salida: str = None
) -> pd.Series:
    """
    Detecta outliers multivariados usando la distancia de Mahalanobis.
    Guarda gráfico de distancias si se proporciona ruta_salida.

    Args:
        df_num (pd.DataFrame): DataFrame numérico
        umbral (float): Umbral de significancia (default: 0.99)
        ruta_salida (str): Carpeta para guardar gráficos (opcional)

    Returns:
        pd.Series: Serie booleana con outliers detectados
    """
    x = df_num.dropna().values
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    dist_maha = np.array([
        np.dot(np.dot((row - mu), inv_cov), (row - mu).T)
        for row in x
    ])

    chi2_limite = chi2.ppf(umbral, df_num.shape[1])
    outliers = dist_maha > chi2_limite

    logging.info(f"Outliers multivariados detectados: {np.sum(outliers)}")

    # Guardar gráficos si se especifica la ruta
    if ruta_salida:
        Path(ruta_salida).mkdir(parents=True, exist_ok=True)
        df_dist = pd.DataFrame({"dist_mahalanobis": dist_maha})

        # Histograma
        plt.figure()
        sns.histplot(df_dist["dist_mahalanobis"], kde=True)
        plt.axvline(chi2_limite, color='r', linestyle='--', label=f'Umbral {umbral}')
        plt.title("Distribución de distancias de Mahalanobis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_salida, "mahalanobis_dist_hist.png"))
        plt.close()

        # Boxplot
        plt.figure()
        sns.boxplot(x=df_dist["dist_mahalanobis"])
        plt.title("Boxplot de distancias de Mahalanobis")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_salida, "mahalanobis_dist_boxplot.png"))
        plt.close()

    return pd.Series(outliers, index=df_num.dropna().index)

#### 5. CORRELACIÓN ####

def generar_matriz_correlacion(df: pd.DataFrame, save_path: str = None, metodo: str = "pearson") -> pd.DataFrame:
    """
    Calcula y grafica la matriz de correlación entre variables numéricas.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas.
        save_path (str): Ruta para guardar la imagen PNG.
        metodo (str): Método de correlación ('pearson', 'spearman', 'kendall').

    Returns:
        pd.DataFrame: Matriz de correlación numérica.
    """
    # Calcular matriz
    corr_matrix = df.corr(method=metodo)

    # Configurar gráfico
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.75}
    )
    plt.title(f"Matriz de correlación ({metodo})", fontsize=14)
    plt.tight_layout()

    # Guardar imagen
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Matriz de correlación guardada en: {save_path}")

    plt.close()
    return corr_matrix

#### 6. INFORMATION VALUE ####

import pandas as pd
import numpy as np
import os
import logging

def calcular_information_value(
    df: pd.DataFrame,
    target: str,
    save_path: str = None,
    bins: int = 10
) -> pd.DataFrame:
    """
    Calcula el Information Value (IV) para todas las variables de un DataFrame, excepto la variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame con variables predictoras y objetivo.
        target (str): Nombre de la variable objetivo binaria (0/1).
        save_path (str, optional): Ruta para guardar el HTML con los resultados.
        bins (int): Número de bins para discretizar variables numéricas.

    Returns:
        pd.DataFrame: DataFrame con columnas 'variable' e 'information_value'.
    """
    resultado = []

    for var in df.drop(columns=target).columns:
        try:
            if df[var].nunique() <= 1 or df[var].isna().all():
                logging.warning(f"{var} excluida por ser constante o completamente nula.")
                continue

            serie = df[var]
            if pd.api.types.is_numeric_dtype(serie):
                serie = pd.qcut(serie, q=bins, duplicates="drop")

            tabla = pd.crosstab(serie, df[target], normalize='columns') + 1e-6
            dist_0 = tabla[0]
            dist_1 = tabla[1]
            woe = np.log(dist_0 / dist_1)
            iv = ((dist_0 - dist_1) * woe).sum()

            resultado.append({
                "variable": var,
                "information_value": round(iv, 4)
            })

        except Exception as e:
            logging.warning(f"No se pudo calcular IV para {var}: {e}")

    df_iv = pd.DataFrame(resultado)

    if df_iv.empty:
        logging.warning("No se pudo calcular IV para ninguna variable.")
    else:
        df_iv = df_iv.sort_values(by="information_value", ascending=False)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_iv.to_html(save_path, index=False)
            logging.info(f"Information Value exportado a HTML: {save_path}")

    logging.info("Cálculo de Information Value completado.")
    return df_iv

#### 7. DUMMIFICACIÓN ####

# Dumificación de variables categóricas con One-Hot Encoder

def dumificar_variables(df: pd.DataFrame, target: str = "income", save_path: str = None) -> pd.DataFrame:
    """
    Aplica codificación one-hot a las variables categóricas (excluyendo la variable objetivo) y guarda un resumen HTML.

    Args:
        df (pd.DataFrame): DataFrame completo con variables categóricas y numéricas.
        target (str): Nombre de la variable objetivo binaria (por defecto 'income').
        save_path (str, optional): Ruta para guardar el HTML con nombres de columnas dummificadas.

    Returns:
        pd.DataFrame: DataFrame con variables categóricas dummificadas y el resto sin modificar.
    """
    # Detectar columnas object a convertir (excluyendo target)
    columnas_a_convertir = [
        col for col in df.select_dtypes(include="object").columns if col != target
    ]

    if columnas_a_convertir:
        df[columnas_a_convertir] = df[columnas_a_convertir].astype("category")
        logging.info(f"Columnas convertidas a 'category': {columnas_a_convertir}")

    # Seleccionar variables categóricas excluyendo el target
    cat_vars = df.select_dtypes(include="category").drop(columns=[target], errors="ignore")

    # Aplicar one-hot encoding
    df_dummies = pd.get_dummies(cat_vars, drop_first=True)

    # Concatenar con el resto del DataFrame
    df_restante = df.drop(columns=cat_vars.columns)
    df_final = pd.concat([df_restante, df_dummies], axis=1)

    logging.info(f"Dumificación aplicada. Variables categóricas transformadas: {len(cat_vars.columns)}")

    # Guardar resumen HTML si se especifica
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_dummies.columns.to_frame(name="dummies").to_html(save_path, index=False)
        logging.info(f"Resumen de columnas dummificadas guardado en: {save_path}")

    return df_final



