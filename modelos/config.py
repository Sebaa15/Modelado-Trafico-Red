"""Configuracion centralizada y unificada de parametros del sistema.

Homogeneiza los parametros para todos los modulos del proyecto segun
el marco teorico y banco de pruebas del informe (Elemento3/CasoDeEstudio/modelado-e2.md)
y el dataset de estadisticas pasivas de CAIDA (https://www.caida.org/catalog/datasets/trace_stats/).
"""
from pathlib import Path

# Rutas del proyecto
DIR_BASE = Path(__file__).resolve().parents[1]
DIR_DATOS = DIR_BASE / "datos"
DIR_RESULTADOS = DIR_BASE / "resultados"
DIR_DOCUMENTACION = DIR_BASE / "documentacion"

ARCHIVO_DATASET_CAIDA = DIR_DATOS / "caida_trace_stats.csv"

# Parametros temporales estandarizados
DT = 1.0                  # Paso de integracion discreto (segundos)
DURACION_ESTANDAR = 600   # Horizonte estandar de simulacion: 600 s (10 minutos)
N_PASOS = int(DURACION_ESTANDAR / DT)

# Parametros de red y enlace
BUFFER_MAX = 50000.0      # Buffer estandar de router B_max = 50,000 paquetes
CAPACIDAD_NOMINAL = 1000.0  # Capacidad nominal del enlace mu = 1000 paquetes/s
CAPACIDAD_DEGRADADA = 600.0 # Capacidad de enlace degradado mu = 600 paquetes/s

# Tasas de llegada de referencia (paquetes/s)
LAMBDA_VALLE = 800.0      # Trafico moderado / horario valle (u = 0.80)
LAMBDA_PICO = 1500.0      # Trafico intenso / hora pico con streaming (u = 1.50)

# Parametros estandar del Filtro de Kalman (Escenario D)
Q_ESTANDAR = 100.0        # Varianza del ruido de proceso
R_ESTANDAR = 2500.0       # Varianza del ruido de medicion (sigma_v = 50 paquetes)
P0_ESTANDAR = 1000.0      # Incertidumbre inicial de covarianza

# Parametros de toma de decisiones
UMBRAL_ALERTA_BUFFER = 0.80 # 80% del buffer (40,000 paquetes)
HORIZONTE_PRED = 15         # Proyeccion anticipatoria a 15 segundos
NIVEL_CONFIANZA_Z = 1.96    # Intervalo de credibilidad al 95%
