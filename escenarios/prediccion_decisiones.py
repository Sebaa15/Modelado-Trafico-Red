"""Escenario B: prediccion de 120 s antes de la saturacion.

El perfil futuro de llegadas se asume conocido para comparar la
prediccion con el estado generado por Euler. No usa CAIDA ni Bayes.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from modelos.prediccion import predecir_cola
from modelos.decisiones import comparar_capacidades, evaluar_congestion


def ejecutar():
    dt = 1.0
    n = 1200
    instante_corte = 350
    horizonte = 120
    capacidad = 1000.0
    buffer = 100000.0

    llegadas = np.full(n, 800.0)
    llegadas[200:800] = 1500.0
    referencia = modelo_euler(llegadas, capacidad, dt=dt, buffer_max=buffer)

    rng = np.random.default_rng(42)
    observadas = referencia["cola"][:instante_corte + 1] + rng.normal(
        0, 50, instante_corte + 1
    )
    estimaciones, _ = estimar_cola(
        llegadas[:instante_corte], observadas, capacidad,
        dt=dt, Q=100, R=2500, buffer_max=buffer
    )
    estado_estimado = float(estimaciones[-1])
    perfil_futuro = llegadas[instante_corte:instante_corte + horizonte]

    # Capacidad alternativa: cubre el maximo supuesto de llegadas
    nueva_capacidad = float(max(capacidad, np.max(perfil_futuro)))
    comparacion = comparar_capacidades(
        estado_estimado, perfil_futuro, capacidad, nueva_capacidad,
        dt=dt, buffer_max=buffer
    )
    pronostico = comparacion["actual"]
    contrafactual = comparacion["alternativa"]
    realidad_futura = referencia["cola"][instante_corte:instante_corte + horizonte + 1]

    error_mae = float(np.mean(np.abs(pronostico["cola"] - realidad_futura)))
    alerta = evaluar_congestion(pronostico)
    alerta_alternativa = evaluar_congestion(contrafactual)
    tiempo_absoluto = instante_corte + pronostico["tiempo"]

    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    resumen = {
        "corte_s": instante_corte,
        "horizonte_s": horizonte * dt,
        "cola_estimada_corte": round(estado_estimado, 3),
        "capacidad_actual_pps": capacidad,
        "capacidad_alternativa_pps": nueva_capacidad,
        "alerta_actual": alerta["nivel"],
        "tiempo_buffer_lleno_desde_corte_s": alerta["tiempo_buffer_lleno_s"],
        "perdidas_previstas_actual": round(alerta["paquetes_perdidos"], 3),
        "perdidas_previstas_alternativa": round(alerta_alternativa["paquetes_perdidos"], 3),
        "mae_prediccion_vs_euler": round(error_mae, 3),
    }
    with open(carpeta / "prediccion_decisiones.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resumen.keys())
        writer.writeheader()
        writer.writerow(resumen)

    plt.figure(figsize=(11, 6))
    plt.plot(tiempo_absoluto, realidad_futura, label="Referencia Euler")
    plt.plot(tiempo_absoluto, pronostico["cola"], label="Pronostico desde Kalman")
    plt.plot(tiempo_absoluto, contrafactual["cola"],
             label="Con capacidad ampliada (hipotetica)")
    plt.axhline(buffer, linestyle="--", label="Limite del buffer")
    plt.axvline(instante_corte, linestyle=":", label="Corte de observaciones")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Paquetes en cola")
    plt.title("Prediccion y decision de capacidad: escenario B")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta / "prediccion_decisiones.png", dpi=150)
    plt.close()

    print("\nPREDICCION Y DECISIONES")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")
    print("Advertencia: el perfil futuro es un supuesto de simulacion, no una prediccion de CAIDA.")
    return resumen


if __name__ == "__main__":
    ejecutar()
