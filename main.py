"""Ejecutor de componentes disponibles del proyecto.

No declara finalizado Bayes ni CAIDA; estos se integran cuando estan listos.
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

EXPERIMENTOS = [
    ("Pruebas automatizadas", ["-m", "pytest", "tests", "-v"], "tests/test_integracion.py"),
    ("Euler - escenarios A, B y C", ["euler/ejecutar_escenarios.py"], "euler/ejecutar_escenarios.py"),
    ("Kalman - escenario D", ["-m", "escenarios.escenario_d"], "escenarios/escenario_d.py"),
    ("Calibracion Q/R inicial", ["-m", "escenarios.validacion_kalman"], "escenarios/validacion_kalman.py"),
    ("Integracion Euler–Kalman", ["-m", "escenarios.integracion_euler_kalman"], "escenarios/integracion_euler_kalman.py"),
    ("Resumen de escenarios", ["-m", "escenarios.resumen_experimentos"], "escenarios/resumen_experimentos.py"),
    ("Comparacion Kalman/EKF", ["-m", "escenarios.comparacion_kalman_ekf"], "escenarios/comparacion_kalman_ekf.py"),
    ("Validacion independiente", ["-m", "escenarios.validacion_independiente"], "escenarios/validacion_independiente.py"),
    ("Prediccion y decisiones", ["-m", "escenarios.prediccion_decisiones"], "escenarios/prediccion_decisiones.py"),
    ("Generacion del informe Kalman", ["-m", "escenarios.generar_informe_kalman"], "escenarios/generar_informe_kalman.py"),
]


def main():
    print("SIMULACION DE TRAFICO DE RED", flush=True)
    print("=" * 45, flush=True)
    entorno = os.environ.copy()
    entorno["MPLBACKEND"] = "Agg"
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

    print("\n" + "=" * 45)
    print(f"Actividades locales finalizadas: {completados}/{len(EXPERIMENTOS)}")
    print("Pendientes del sistema integral: red bayesiana y escenario E (CAIDA).")
    print("La demostracion completa de Bayes -> Euler -> Kalman -> decision ")
    print("requiere incorporar el trabajo del integrante 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
