import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent

EXPERIMENTOS = [
    (
        "Pruebas automatizadas",
        ["-m", "pytest", "tests/test_integracion.py", "-v"],
        "tests/test_integracion.py"
    ),
    (
        "Euler - escenarios A, B y C",
        ["euler/ejecutar_escenarios.py"],
        "euler/ejecutar_escenarios.py"
    ),
    (
        "Kalman - escenario D",
        ["-m", "escenarios.escenario_d"],
        "escenarios/escenario_d.py"
    ),
    (
        "Calibracion de Kalman",
        ["-m", "escenarios.validacion_kalman"],
        "escenarios/validacion_kalman.py"
    ),
    (
        "Integracion Euler y Kalman",
        ["-m", "escenarios.integracion_euler_kalman"],
        "escenarios/integracion_euler_kalman.py"
    ),
    (
        "Resumen de escenarios",
        ["-m", "escenarios.resumen_experimentos"],
        "escenarios/resumen_experimentos.py"
    ),
    (
        "Comparacion Kalman y EKF",
        ["-m", "escenarios.comparacion_kalman_ekf"],
        "escenarios/comparacion_kalman_ekf.py"
    )
]


def main():
    print("SIMULACION DE TRAFICO DE RED")
    print("=" * 45)

    # Evita ventanas bloqueantes durante la ejecucion.
    entorno = os.environ.copy()
    entorno["MPLBACKEND"] = "Agg"

    completados = 0
    omitidos = 0

    for nombre, comando, archivo in EXPERIMENTOS:

        print(f"\nEjecutando: {nombre}", flush=True)

        if not (BASE / archivo).exists():
            print(f"PENDIENTE: falta {archivo}")
            omitidos += 1
            continue

        resultado = subprocess.run(
            [sys.executable] + comando,
            cwd=BASE,
            env=entorno
        )

        if resultado.returncode != 0:
            print(f"\nERROR en: {nombre}")
            sys.exit(resultado.returncode)

        completados += 1
        print(f"OK: {nombre}", flush=True)

    print("\n" + "=" * 45)
    print("RESUMEN DE EJECUCION")
    print(f"Actividades completadas: {completados}")
    print(f"Actividades pendientes: {omitidos}")
    print("Bayes y escenario E: pendientes de integracion.")

    if omitidos == 0:
        print("Experimentos disponibles ejecutados correctamente.")


if __name__ == "__main__":
    main()
