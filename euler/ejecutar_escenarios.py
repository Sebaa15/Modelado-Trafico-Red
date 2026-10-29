"""Ejecuta el modelo determinista de Euler para los Escenarios A, B y C.

Homogeneizado con los parametros unificados del proyecto y el dataset CAIDA:
  - Duracion estandar: 600 s (10 minutos), dt = 1.0 s.
  - Buffer estandar de router: B_max = 50,000 paquetes.
  - Capacidad nominal: mu = 1000 paquetes/s.
  - Capacidad degradada: mu = 600 paquetes/s.
"""
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.config import (
    DIR_RESULTADOS,
    DT,
    DURACION_ESTANDAR,
    BUFFER_MAX,
    CAPACIDAD_NOMINAL,
    CAPACIDAD_DEGRADADA,
    LAMBDA_VALLE,
    LAMBDA_PICO,
)


def ejecutar():
    print("=" * 60)
    print("CAPA 1: MODELO DETERMINISTA DE EULER (ESCENARIOS A, B, C)")
    print("=" * 60)

    n = int(DURACION_ESTANDAR / DT)
    tiempo = np.arange(n) * DT

    # 1. Definicion homogenea de perfiles
    # Escenario A: Trafico moderado (800 pkt/s, mu = 1000)
    llegadas_A = np.full(n, LAMBDA_VALLE)

    # Escenario B: Hora pico con streaming (1500 pkt/s entre t=100s y t=400s, mu = 1000)
    llegadas_B = np.full(n, LAMBDA_VALLE)
    llegadas_B[(tiempo >= 100) & (tiempo < 400)] = LAMBDA_PICO

    # Escenario C: Enlace degradado (800 pkt/s, mu = 600)
    llegadas_C = np.full(n, LAMBDA_VALLE)

    # 2. Simulacion con buffer finito B_max = 50,000
    res_A = modelo_euler(llegadas_A, CAPACIDAD_NOMINAL, dt=DT, buffer_max=BUFFER_MAX)
    res_B = modelo_euler(llegadas_B, CAPACIDAD_NOMINAL, dt=DT, buffer_max=BUFFER_MAX)
    res_C = modelo_euler(llegadas_C, CAPACIDAD_DEGRADADA, dt=DT, buffer_max=BUFFER_MAX)

    # Simulacion adicional de B sin limite para comparar impacto del buffer
    res_B_sin_buffer = modelo_euler(llegadas_B, CAPACIDAD_NOMINAL, dt=DT, buffer_max=None)

    escenarios = {
        "A": res_A,
        "B": res_B,
        "C": res_C,
        "B_sin_buffer": res_B_sin_buffer
    }

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    metricas = []

    for nombre, res in escenarios.items():
        cola = res["cola"]
        util = res["utilizacion"]
        perd = res["perdidas"]

        idx_sat = np.where(cola >= (res["buffer_max"] or np.inf) - 1e-3)[0]
        t_sat = float(res["tiempo"][idx_sat[0]]) if len(idx_sat) > 0 else None

        fila = {
            "Escenario": nombre,
            "Capacidad_pps": res["capacidad"],
            "Buffer_Max_pkt": res["buffer_max"] if res["buffer_max"] is not None else "Infinito",
            "Utilizacion_Max_pct": round(float(np.max(util)) * 100.0, 2),
            "Utilizacion_Media_pct": round(float(np.mean(util)) * 100.0, 2),
            "Cola_Maxima_pkt": round(float(np.max(cola)), 1),
            "Cola_Final_pkt": round(float(cola[-1]), 1),
            "Perdidas_Totales_pkt": round(float(np.sum(perd)), 1),
            "Instante_Saturacion_s": t_sat if t_sat is not None else "N/A"
        }
        metricas.append(fila)

        print(f"\n[{fila['Escenario']}]")
        print(f"  Capacidad: {fila['Capacidad_pps']} pkt/s | Buffer: {fila['Buffer_Max_pkt']}")
        print(f"  Utilizacion maxima: {fila['Utilizacion_Max_pct']}% | Cola maxima: {fila['Cola_Maxima_pkt']} pkt")
        print(f"  Perdidas acumuladas: {fila['Perdidas_Totales_pkt']} pkt | Saturacion en t = {fila['Instante_Saturacion_s']}")

    # Exportar CSV
    csv_path = DIR_RESULTADOS / "metricas_escenarios_euler.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metricas[0].keys()))
        writer.writeheader()
        writer.writerows(metricas)
    print(f"\nMetricas guardadas en: {csv_path}")

    # 3. Grafica clara y homogenea
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 1: Tasas de llegada vs capacidad
    axes[0].plot(tiempo, llegadas_A, label=r"Esc. A y C ($\lambda=800$)", color="#1f77b4", lw=1.8)
    axes[0].plot(tiempo, llegadas_B, label=r"Esc. B ($\lambda=1500$ pico)", color="#d62728", lw=1.8)
    axes[0].axhline(CAPACIDAD_NOMINAL, color="gray", linestyle="--", label=r"$\mu$ Nominal (1000 pkt/s)")
    axes[0].axhline(CAPACIDAD_DEGRADADA, color="orange", linestyle=":", label=r"$\mu$ Degradado (600 pkt/s)")
    axes[0].set_ylabel("Llegadas (pkt/s)", fontsize=10)
    axes[0].set_title("Capa 1: Dinámica Determinista de Euler (Escenarios A, B y C)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(alpha=0.3)

    # Panel 2: Evolucion de la cola con buffer B_max = 50,000
    t_cola = res_A["tiempo"]
    axes[1].plot(t_cola, res_A["cola"], label="Escenario A (Estable, cola = 0)", color="#1f77b4", lw=1.8)
    axes[1].plot(t_cola, res_B["cola"], label="Escenario B (Saturación a t = 200s)", color="#d62728", lw=1.8)
    axes[1].plot(t_cola, res_C["cola"], label="Escenario C (Degradado, satura a t = 250s)", color="#ff7f0e", lw=1.8, linestyle="--")
    axes[1].axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.8, label=r"Límite de Buffer ($B_{\max} = 50\mathrm{k}$)")
    axes[1].set_ylabel("Cola (paquetes)", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(alpha=0.3)

    # Panel 3: Perdidas acumuladas por desbordamiento (tail-drop)
    axes[2].plot(tiempo, np.cumsum(res_A["perdidas"]), label="Escenario A (0 pérdidas)", color="#1f77b4", lw=1.8)
    axes[2].plot(tiempo, np.cumsum(res_B["perdidas"]), label="Escenario B (Pérdidas acumuladas)", color="#d62728", lw=1.8)
    axes[2].plot(tiempo, np.cumsum(res_C["perdidas"]), label="Escenario C (Pérdidas crónicas)", color="#ff7f0e", lw=1.8, linestyle="--")
    axes[2].set_xlabel("Tiempo (s)", fontsize=10)
    axes[2].set_ylabel("Pérdidas acumuladas (pkt)", fontsize=10)
    axes[2].legend(loc="upper left", fontsize=8.5)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    png_path = DIR_RESULTADOS / "fig1_escenarios_euler_caida.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {png_path}")
    print("=" * 60)
    return metricas


if __name__ == "__main__":
    ejecutar()
