import csv
from pathlib import Path
import numpy as np

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola


def metricas(real, estimado):
    error = estimado - real
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    return mae, rmse


def ejecutar():
    dt = 1.0
    n = 1200
    tiempo = np.arange(n) * dt

    moderado = np.full(n, 800.0)
    pico = moderado.copy()

    pico[(tiempo >= 200) & (tiempo < 800)] = 1500.0

    escenarios = {
        "A": (moderado, 1000.0, None),
        "B": (pico, 1000.0, None),
        "C": (moderado, 600.0, None),
        "B_buffer": (pico, 1000.0, 100000.0)
    }

    generador = np.random.default_rng(42)
    filas = []

    for nombre, (llegadas, capacidad, buffer) in escenarios.items():

        resultado = modelo_euler(
            llegadas, capacidad,
            dt=dt, buffer_max=buffer
        )

        real = resultado["cola"]

        mediciones = real + generador.normal(
            0, 50, len(real)
        )

        estimado, sigma = estimar_cola(
            llegadas, mediciones, capacidad,
            dt=dt, Q=100, R=2500,
            buffer_max=buffer
        )

        mae_med, _ = metricas(real, mediciones)
        mae_k, rmse_k = metricas(real, estimado)

        cobertura = 100 * np.mean(
            np.abs(estimado - real) <= 1.96 * sigma
        )

        fila = {
            "Escenario": nombre,
            "Cola maxima": round(np.max(real), 2),
            "Perdidas": round(
                np.sum(resultado["perdidas"]), 2
            ),
            "MAE mediciones": round(mae_med, 2),
            "MAE Kalman": round(mae_k, 2),
            "RMSE Kalman": round(rmse_k, 2),
            "Cobertura aproximada %": round(cobertura, 2)
        }

        filas.append(fila)

        print(f"\nEscenario: {nombre}")
        for clave, valor in fila.items():
            print(f"{clave}: {valor}")

    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)

    with open(
        carpeta / "resumen_euler_kalman.csv",
        "w", newline="", encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo, fieldnames=filas[0].keys()
        )
        escritor.writeheader()
        escritor.writerows(filas)

    print("\nResumen guardado correctamente.")


if __name__ == "__main__":
    ejecutar()
