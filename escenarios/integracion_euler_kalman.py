
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola


# Parametros generales
dt = 1.0
duracion = 1200
tiempos = np.arange(duracion) * dt

# Escenario A: trafico moderado
llegadas_a = np.full(duracion, 800.0)

# Escenario B: hora pico
llegadas_b = np.full(duracion, 800.0)
pico = (tiempos >= 200) & (tiempos < 800)
llegadas_b[pico] = 1500.0

# Escenario C: capacidad reducida
llegadas_c = np.full(duracion, 800.0)

# Escenarios del componente Euler
escenarios = {
    "A": (llegadas_a, 1000.0, None),
    "B": (llegadas_b, 1000.0, None),
    "C": (llegadas_c, 600.0, None),
    "B_buffer": (llegadas_b, 1000.0, 100000.0)
}

Path("resultados").mkdir(exist_ok=True)

generador = np.random.default_rng(42)

for nombre, (llegadas, capacidad, buffer) in escenarios.items():

    # 1. Obtener el estado simulado con Euler
    resultado = modelo_euler(
        llegadas,
        capacidad,
        dt=dt,
        buffer_max=buffer
    )

    cola_real = resultado["cola"]
    tiempo = resultado["tiempo"]

    # 2. Simular mediciones ruidosas
    mediciones = cola_real + generador.normal(
        0, 50, len(cola_real)
    )

    # 3. Estimar utilizando nuestro filtro de Kalman
    estimacion, incertidumbre = estimar_cola(
        llegadas,
        mediciones,
        capacidad,
        dt=dt,
        Q=100,
        R=2500,
        buffer_max=buffer
    )

    # 4. Evaluar los errores
    mae = np.mean(np.abs(cola_real - estimacion))

    rmse = np.sqrt(
        np.mean((cola_real - estimacion) ** 2)
    )

    print(f"\nESCENARIO {nombre}")
    print(f"Cola maxima: {np.max(cola_real):.2f}")
    print(f"MAE Kalman: {mae:.2f}")
    print(f"RMSE Kalman: {rmse:.2f}")
    print(
        "Perdidas estimadas:",
        np.sum(resultado["perdidas"])
    )

    # 5. Graficar comparacion
    plt.figure(figsize=(11, 5))

    plt.plot(
        tiempo, cola_real,
        label="Cola simulada - Euler"
    )

    plt.plot(
        tiempo, mediciones,
        label="Mediciones con ruido",
        alpha=0.3
    )

    plt.plot(
        tiempo, estimacion,
        label="Cola estimada - Kalman"
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Paquetes en cola")
    plt.title(f"Integracion Euler-Kalman: {nombre}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"resultados/integracion_{nombre}.png",
        dpi=150
    )

    plt.close()

print("\nIntegracion finalizada correctamente.")
