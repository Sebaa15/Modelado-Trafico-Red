"""Ejecutor integral de todos los componentes y escenarios del proyecto.

Integra colaborativamente:
  - Capa 1: Dinamica determinista Euler (Escenarios A, B, C)
  - Capa 2: Inferencia contextual con Red Bayesiana
  - Capa 3: Estimacion en linea con Filtros de Kalman y EKF (Escenario D)
  - Escenario E: Trafico real CAIDA con micro-rafagas y flujos
  - Pipeline integrado: Bayes -> Euler -> Kalman -> Decisiones de red
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

EXPERIMENTOS = [
    ("Pruebas automatizadas del sistema", ["-m", "pytest", "tests", "-v"], "tests/test_integracion.py"),
    ("Euler - escenarios A, B y C", ["-m", "euler.ejecutar_escenarios"], "euler/ejecutar_escenarios.py"),
    ("Red Bayesiana de contexto (Capa 2)", ["-m", "escenarios.inferencia_bayesiana"], "escenarios/inferencia_bayesiana.py"),
    ("Kalman - escenario D", ["-m", "escenarios.escenario_d"], "escenarios/escenario_d.py"),
    ("Calibracion Q/R inicial", ["-m", "escenarios.validacion_kalman"], "escenarios/validacion_kalman.py"),
    ("Integracion Euler–Kalman", ["-m", "escenarios.integracion_euler_kalman"], "escenarios/integracion_euler_kalman.py"),
    ("Resumen de escenarios", ["-m", "escenarios.resumen_experimentos"], "escenarios/resumen_experimentos.py"),
    ("Comparacion Kalman/EKF", ["-m", "escenarios.comparacion_kalman_ekf"], "escenarios/comparacion_kalman_ekf.py"),
    ("Validacion independiente", ["-m", "escenarios.validacion_independiente"], "escenarios/validacion_independiente.py"),
    ("Prediccion y decisiones", ["-m", "escenarios.prediccion_decisiones"], "escenarios/prediccion_decisiones.py"),
    ("Escenario E - Trafico CAIDA y micro-rafagas", ["-m", "escenarios.escenario_e"], "escenarios/escenario_e.py"),
    ("Pipeline integrado Bayes-Euler-Kalman", ["-m", "escenarios.pipeline_integrado"], "escenarios/pipeline_integrado.py"),
    ("Dashboards unificados CAIDA y analisis visual", ["-m", "escenarios.visualizacion_unificada"], "escenarios/visualizacion_unificada.py"),
    ("Generacion del informe Kalman", ["-m", "escenarios.generar_informe_kalman"], "escenarios/generar_informe_kalman.py"),
]


def main():
    print("SIMULACION DE TRAFICO DE RED - SISTEMA INTEGRADO DE 3 CAPAS", flush=True)
    print("=" * 65, flush=True)
    entorno = os.environ.copy()
    entorno["MPLBACKEND"] = "Agg"
    entorno["PYTHONPATH"] = str(BASE) + os.pathsep + entorno.get("PYTHONPATH", "")
    completados = 0
    for nombre, argumentos, archivo in EXPERIMENTOS:
        print(f"\nEjecutando: {nombre}", flush=True)
        if not (BASE / archivo).is_file():
            print(f"ERROR: falta {archivo}.", flush=True)
            return 1
        proceso = subprocess.run([sys.executable] + argumentos,
                                cwd=BASE, env=entorno, check=False)
        if proceso.returncode != 0:
            print(f"ERROR en: {nombre}", flush=True)
            return proceso.returncode
        completados += 1
        print(f"OK: {nombre}", flush=True)

    print("\n" + "=" * 65)
    print(f"Actividades finalizadas exitosamente: {completados}/{len(EXPERIMENTOS)}")
    print("Sistema integral completado colaborativamente:")
    print("  [x] Capa 1: Modelo dinámico Euler (Escenarios A, B, C)")
    print("  [x] Capa 2: Inferencia contextual con Red Bayesiana (Consulta Fase 3: 82.25%)")
    print("  [x] Capa 3: Estimacion de estado en linea con Kalman y EKF (Escenario D)")
    print("  [x] Escenario E: Traza backbone CAIDA, auto-similitud y flujos como leading indicator")
    print("  [x] Pipeline integral cerrado: Bayes -> Euler -> Kalman -> Decisiones de red")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
