import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from modelos.kalman_extendido import estimar_cola_ekf


# Escenario B: hora pico con buffer limitado
dt = 1.0
n = 1200
mu = 1000.0
buffer_max = 100000.0

llegadas = np.full(n, 800.0)
llegadas[200:800] = 1500.0

# Estado de referencia mediante Euler
resultado = modelo_euler(
    llegadas, mu,
    dt=dt, buffer_max=buffer_max
)

cola_real = resultado["cola"]
tiempo = resultado["tiempo"]

# Mismas mediciones para ambos filtros
rng = np.random.default_rng(42)
mediciones = cola_real + rng.normal(
    0, 50, len(cola_real)
)

# Kalman lineal con restricciones
estimacion_kf, sigma_kf = estimar_cola(
    llegadas, mediciones, mu,
    dt=dt, Q=100, R=2500,
    buffer_max=buffer_max
)

# Kalman extendido
estimacion_ekf, sigma_ekf = estimar_cola_ekf(
    llegadas, mediciones, mu,
    dt=dt, Q=100, R=2500,
    buffer_max=buffer_max
)


def calcular_metricas(real, estimado):
    error = real - estimado

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2)))
    }


resultados = {
    "Mediciones": calcular_metricas(cola_real, mediciones),
    "Kalman": calcular_metricas(cola_real, estimacion_kf),
    "EKF": calcular_metricas(cola_real, estimacion_ekf)
}

print("\nCOMPARACION KALMAN VS EKF")
print("-" * 45)

for nombre, valores in resultados.items():
    print(
        f"{nombre}: "
        f"MAE={valores['MAE']:.2f}, "
        f"RMSE={valores['RMSE']:.2f}"
    )

print("\nCola maxima:", np.max(cola_real))
print("Paquetes perdidos:", np.sum(resultado["perdidas"]))

# Comprobar restricciones
for nombre, serie in [
    ("Kalman", estimacion_kf),
    ("EKF", estimacion_ekf)
]:
    assert len(serie) == len(cola_real)
    assert np.all(np.isfinite(serie))
    assert np.all(serie >= 0)
    assert np.all(serie <= buffer_max)

Path("resultados").mkdir(exist_ok=True)

# Guardar metricas
with open(
    "resultados/comparacion_kalman_ekf.csv",
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["Metodo", "MAE", "RMSE"])

    for nombre, valores in resultados.items():
        escritor.writerow([
            nombre, valores["MAE"], valores["RMSE"]
        ])

# Generar graficas
fig, ax = plt.subplots(2, 1, figsize=(11, 9))

ax[0].plot(tiempo, cola_real, label="Euler")
ax[0].plot(
    tiempo, estimacion_kf,
    label="Kalman", linestyle="--"
)
ax[0].plot(
    tiempo, estimacion_ekf,
    label="EKF", linestyle=":"
)
ax[0].axhline(
    buffer_max, linestyle="--", label="Limite buffer"
)
ax[0].set_title("Comparacion general")
ax[0].set_ylabel("Paquetes")
ax[0].legend()
ax[0].grid(True)

# Ampliacion del instante de saturacion
inicio, fin = 350, 460

ax[1].plot(
    tiempo[inicio:fin],
    cola_real[inicio:fin], label="Euler"
)
ax[1].plot(
    tiempo[inicio:fin],
    estimacion_kf[inicio:fin], label="Kalman"
)
ax[1].plot(
    tiempo[inicio:fin],
    estimacion_ekf[inicio:fin], label="EKF"
)

ax[1].set_title("Detalle cerca de la saturacion")
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_ylabel("Paquetes")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig(
    "resultados/comparacion_kalman_ekf.png",
    dpi=150
)
plt.show()

print("\nComparacion finalizada correctamente.")
