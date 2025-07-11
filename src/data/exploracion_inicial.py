from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def diagnostico_inicial_numericas(df: pd.DataFrame, save_dir: str = "outputs/01_diagnostico") -> None:
    """
    Realiza un análisis exploratorio inicial del DataFrame.
    Genera gráficos, tabla de nulos, y resumen estadístico.

    Args:
        df (pd.DataFrame): Datos a analizar.
        save_dir (str): Carpeta donde guardar los gráficos y reportes.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Resumen estructural
    logging.info("Resumen general del dataset:")
    logging.info(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    logging.info(f"Columnas:\n{df.dtypes}")
    logging.info("Resumen estadístico:\n" + str(df.describe().round(2)))

    # Nulos con tipo de dato
    nulos = df.isnull().sum()
    total = len(df)
    porcentaje = ((nulos / total) * 100).round(2)
    tipos = df.dtypes
    df_nulos = pd.DataFrame({
        "nulos": nulos,
        "porcentaje": porcentaje,
        "tipo_dato": tipos
    }).sort_values("nulos", ascending=False)

    logging.info("Conteo de nulos por columna:\n" + str(df_nulos))

    # Exportar diagnósticos
    df_nulos.to_excel(os.path.join(save_dir, "diagnostico_nulos.xlsx"))
    df.describe().round(2).to_html(os.path.join(save_dir, "resumen_estadistico.html"))

    # Gráficos: histogramas y boxplots
    for col in df.select_dtypes(include="number"):
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribución: {col}")
        plt.savefig(os.path.join(save_dir, f"hist_{col}.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.savefig(os.path.join(save_dir, f"box_{col}.png"), bbox_inches="tight")
        plt.close()

def diagnostico_inicial_categoricas(df: pd.DataFrame, save_dir: str, top_n: int = 5) -> pd.DataFrame:
    """
    Diagnóstico de variables categóricas:
    - Tabla con número de clases únicas y lista de categorías
    - Gráficos de barras con frecuencia absoluta y etiquetas de porcentaje
    - Exporta tabla resumen como Excel
    - Guarda gráficos en disco

    Args:
        df (pd.DataFrame): Dataset de entrada
        save_dir (str): Ruta donde guardar los gráficos y resumen
        top_n (int): Número de clases más frecuentes a mostrar (resto se agrupa como 'Otros')

    Returns:
        pd.DataFrame: Tabla resumen de variables categóricas
    """
    os.makedirs(save_dir, exist_ok=True)
    categoricas = df.select_dtypes(include='object')

    resumen = pd.DataFrame({
        'variable': categoricas.columns,
        'n_clases': [df[col].nunique() for col in categoricas.columns],
        'categorias': [', '.join(map(str, df[col].unique())) for col in categoricas.columns]
    })

    # Exportar resumen a Excel
    ruta_excel = os.path.join(save_dir, "resumen_clases.xlsx")
    resumen.to_excel(ruta_excel, index=False)
    logging.info(f"Resumen de clases exportado a: {ruta_excel}")

    for col in categoricas.columns:
        conteo_abs = df[col].value_counts()
        conteo_rel = df[col].value_counts(normalize=True)
        top_abs = conteo_abs.head(top_n)
        top_rel = conteo_rel.head(top_n)

        otros_abs = conteo_abs[top_n:].sum()
        otros_rel = conteo_rel[top_n:].sum()

        if otros_abs > 0:
            top_abs['Otros'] = otros_abs
            top_rel['Otros'] = otros_rel

        # Gráfico
        plt.figure(figsize=(8, 5))
        bars = plt.bar(top_abs.index, top_abs.values, color="skyblue")
        plt.title(f'Distribución de {col} (Top {top_n})')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45, ha='right')

        for bar, prop in zip(bars, top_rel.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{prop*100:.1f}%',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        nombre_archivo = os.path.join(save_dir, f"{col}_top{top_n}.png")
        plt.savefig(nombre_archivo)
        plt.close()
        logging.info(f"Gráfico guardado: {nombre_archivo}")

    return resumen

import pandas as pd
import logging
import os

def calcular_proporcion_ceros(df: pd.DataFrame, columnas: list = None, save_path: str = None) -> pd.DataFrame:
    """
    Calcula la proporción de ceros por columna. Si no se especifican columnas, evalúa todas las numéricas.

    Args:
        df (pd.DataFrame): DataFrame original.
        columnas (list, optional): Lista de columnas a evaluar. Si es None, se usan todas las numéricas.
        save_path (str, optional): Ruta donde guardar el resumen en HTML.

    Returns:
        pd.DataFrame: Resumen con cantidad y proporción de ceros por variable.
    """
    if columnas is None:
        columnas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        logging.info(f"No se especificaron columnas. Se analizarán todas las numéricas: {columnas}")

    resultados = []
    for col in columnas:
        total = df[col].shape[0]
        n_ceros = (df[col] == 0).sum()
        prop = n_ceros / total
        resultados.append({
            "variable": col,
            "ceros": n_ceros,
            "proporcion_ceros": round(prop, 4)
        })

    resumen = pd.DataFrame(resultados)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        resumen.to_html(save_path, index=False)
        logging.info(f"Resumen de ceros guardado en HTML: {save_path}")

    return resumen

