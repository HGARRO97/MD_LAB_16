import os
import logging
import pandas as pd
from typing import List

def cargar_datos_tipo_data(ruta_csv: str, columnas: List[str]) -> pd.DataFrame:
    """
    Carga archivos .data sin encabezado, separador por coma y espacios iniciales.

    Args:
        ruta_csv (str): Ruta al archivo .data o .csv sin encabezado.
        columnas (List[str]): Lista con los nombres de las columnas.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        if not os.path.exists(ruta_csv):
            logging.error(f"No se encontró el archivo en la ruta: {ruta_csv}")
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

        df = pd.read_csv(
            ruta_csv,
            header=None,
            names=columnas,
            skipinitialspace=True
        )

        logging.info(f"Datos cargados correctamente desde: {ruta_csv}")
        return df

    except Exception as e:
        logging.error(f"Error al cargar datos desde {ruta_csv}: {e}")
        raise