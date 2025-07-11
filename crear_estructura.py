import os
import logging

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ruta base del proyecto (ajustar si es necesario)
base_dir = r"C:\Proyectos_Pycharm\Laboratorio_16_Aplicaciones_ML"

# Estructura del proyecto orientada al flujo profesional
estructura = [
    "data/raw",                   # Datos originales descargados
    "data/processed",            # Datos limpios y transformados
    "notebooks",                 # Contiene los notebooks por etapa
    "notebooks/01_AnalisisExploratorio",
    "notebooks/02_Preparacion",
    "notebooks/03_Modelado",
    "notebooks/04_Evaluacion",
    "src/data",                  # Funciones de carga, guardado
    "src/features",              # Funciones de ingeniería de variables
    "src/models",                # Algoritmos de modelado y modelos individuales
    "src/visualization",         # Funciones para gráficos
    "tests",                     # Pruebas de funciones
    "outputs/01_diagnostico",    # Gráficos y reportes del EDA
    "outputs/02_preparacion",    # Resultados de preparación de datos
    "outputs/03_modelado",       # Resultados de modelado
    "outputs/04_evaluacion",     # Resultados de métricas de evaluación
    "references",                # PDFs, papers, fuentes
    "reports/figures",           # Figuras para el informe
]

# Crear carpetas
for carpeta in estructura:
    ruta = os.path.join(base_dir, carpeta)
    os.makedirs(ruta, exist_ok=True)
    logging.info(f"Directorio creado o ya existente: {ruta}")

logging.info(f"Estructura del proyecto LAB-S14 creada en: {base_dir}")

# Crear carpetas
for carpeta in estructura:
    ruta = os.path.join(base_dir, carpeta)
    os.makedirs(ruta, exist_ok=True)
    logging.info(f"Directorio creado o ya existente: {ruta}")

logging.info(f"Estructura del proyecto creada en: {base_dir}")
